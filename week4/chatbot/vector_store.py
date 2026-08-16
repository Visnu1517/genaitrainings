"""
vector_store.py
---------------
A small FAISS-backed vector store. This is our version of the class RAG template,
with three deliberate differences:

1. Correct retrieval. The class template's queryDB() appends `lst[i]` using the
   loop counter instead of `lst[indices[i]]`, so it returns the first 3 chunks no
   matter what you ask. We index by the position FAISS actually returns.
2. Incremental writes. The template only ingests files from a folder at startup.
   Long-term memory needs to save a single new fact mid-conversation, so add_text()
   exists alongside ingest_folder().
3. Cosine similarity on normalized vectors (IndexFlatIP) instead of raw L2, which
   behaves better for text, plus a pure-numpy fallback if faiss isn't installed.

Files written into `store_dir`:
    index.bin    - the FAISS index (only when faiss is installed)
    chunks.json  - every chunk: {doc, id, content}
    state.json   - which source files are already ingested + the embedding backend
"""

from __future__ import annotations

import json
import os

import numpy as np

try:
    import faiss

    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    _HAS_FAISS = False


# Same chunking parameters as the class template.
WORD_COUNT = 512
OVERLAP = 50


def chunk_text(text: str, word_count: int = WORD_COUNT, overlap: int = OVERLAP) -> list[str]:
    """Split text into overlapping word windows so context isn't cut mid-thought."""
    words = text.split()
    if not words:
        return []
    step = max(1, word_count - overlap)
    chunks = []
    for i in range(0, len(words), step):
        window = words[i: i + word_count]
        if window:
            chunks.append(" ".join(window))
        if i + word_count >= len(words):
            break
    return chunks


class VectorStore:
    """Stores text chunks + their embeddings, and finds the most relevant ones."""

    def __init__(self, store_dir: str, embedder):
        self.store_dir = store_dir
        self.embedder = embedder
        self.dim = embedder.dim
        self.chunks: list[dict] = []      # position i here == vector i in the index
        self._vectors: np.ndarray | None = None   # numpy fallback storage
        self._index = None                        # faiss index

        os.makedirs(self.store_dir, exist_ok=True)
        self._load()

    # ---------------- paths ---------------- #
    @property
    def _index_path(self) -> str:
        return os.path.join(self.store_dir, "index.bin")

    @property
    def _chunks_path(self) -> str:
        return os.path.join(self.store_dir, "chunks.json")

    @property
    def _state_path(self) -> str:
        return os.path.join(self.store_dir, "state.json")

    @property
    def _vectors_path(self) -> str:
        return os.path.join(self.store_dir, "vectors.npy")

    # ---------------- setup / persistence ---------------- #
    def _new_index(self):
        # Inner product on unit-length vectors == cosine similarity.
        return faiss.IndexFlatIP(self.dim) if _HAS_FAISS else None

    def _load(self) -> None:
        self.state = {"ingested": {}, "backend": self.embedder.backend, "dim": self.dim}
        if os.path.exists(self._state_path):
            with open(self._state_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # If the embedding backend or size changed, old vectors are meaningless.
            if saved.get("backend") == self.embedder.backend and saved.get("dim") == self.dim:
                self.state = saved
            else:
                self._reset_files()

        if os.path.exists(self._chunks_path):
            with open(self._chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

        self._index = self._new_index()
        if _HAS_FAISS and os.path.exists(self._index_path):
            self._index = faiss.read_index(self._index_path)
        elif not _HAS_FAISS and os.path.exists(self._vectors_path):
            self._vectors = np.load(self._vectors_path)

        # Safety: if metadata and vectors disagree, start clean rather than
        # returning wrong chunks for a query.
        if self._count() != len(self.chunks):
            self._reset_files()
            self.chunks = []
            self._index = self._new_index()
            self._vectors = None

    def _reset_files(self) -> None:
        for p in (self._index_path, self._chunks_path, self._state_path, self._vectors_path):
            if os.path.exists(p):
                os.remove(p)
        self.state = {"ingested": {}, "backend": self.embedder.backend, "dim": self.dim}

    def _count(self) -> int:
        if _HAS_FAISS:
            return self._index.ntotal if self._index is not None else 0
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    def _save(self) -> None:
        with open(self._chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        if _HAS_FAISS and self._index is not None:
            faiss.write_index(self._index, self._index_path)
        elif self._vectors is not None:
            np.save(self._vectors_path, self._vectors)

    # ---------------- writing ---------------- #
    def _add_vectors(self, vecs: np.ndarray) -> None:
        if vecs.size == 0:
            return
        if _HAS_FAISS:
            self._index.add(vecs)
        else:
            self._vectors = vecs if self._vectors is None else np.vstack([self._vectors, vecs])

    def add_text(self, text: str, doc: str = "runtime") -> None:
        """Add one piece of text immediately (used for long-term memory writes)."""
        text = text.strip()
        if not text:
            return
        self._add_vectors(self.embedder.embed(text).reshape(1, -1))
        self.chunks.append({"doc": doc, "id": len(self.chunks), "content": text})
        self._save()

    def ingest_folder(self, docs_path: str) -> int:
        """
        Read every .md/.txt file in `docs_path`, chunk it, embed it, and index it.
        Already-ingested files are skipped (tracked in state.json), so re-running
        is cheap. Returns the number of new chunks added.
        """
        if not os.path.isdir(docs_path):
            return 0

        added = 0
        for filename in sorted(os.listdir(docs_path)):
            if not filename.lower().endswith((".md", ".txt")):
                continue
            full = os.path.join(docs_path, filename)
            # Re-ingest if the file changed since last time.
            stamp = f"{os.path.getmtime(full)}:{os.path.getsize(full)}"
            if self.state["ingested"].get(filename) == stamp:
                continue

            with open(full, "r", encoding="utf-8") as f:
                text = f.read()

            pieces = chunk_text(text)
            if not pieces:
                continue

            self._add_vectors(self.embedder.embed_many(pieces))
            for piece in pieces:
                self.chunks.append(
                    {"doc": filename, "id": len(self.chunks), "content": piece}
                )
            self.state["ingested"][filename] = stamp
            added += len(pieces)

        if added:
            self._save()
        return added

    # ---------------- reading ---------------- #
    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[dict]:
        """
        Return up to k chunks most similar to `query`, best first.
        Each result is {doc, id, content, score}. `min_score` drops weak matches
        so we don't inject irrelevant text into the prompt.
        """
        if not query.strip() or not self.chunks:
            return []

        qvec = self.embedder.embed(query).reshape(1, -1)
        k = min(k, len(self.chunks))

        if _HAS_FAISS:
            scores, positions = self._index.search(qvec, k)
            scores, positions = scores[0], positions[0]
        else:
            sims = (self._vectors @ qvec[0])
            positions = np.argsort(-sims)[:k]
            scores = sims[positions]

        results = []
        for score, pos in zip(scores, positions):
            # NOTE: index by the position FAISS returned, not the loop counter.
            if pos == -1 or pos >= len(self.chunks):
                continue
            if float(score) < min_score:
                continue
            item = dict(self.chunks[int(pos)])
            item["score"] = float(score)
            results.append(item)
        return results

    def clear(self) -> None:
        """Wipe this store completely."""
        self._reset_files()
        self.chunks = []
        self._index = self._new_index()
        self._vectors = None
