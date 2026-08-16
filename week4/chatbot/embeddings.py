"""
embeddings.py
-------------
Turns text into vectors (embeddings) so we can measure meaning-similarity.

Two backends:

1. "ollama"  (preferred, matches the class RAG template)
   Calls a locally running Ollama server at http://localhost:11434 using the
   `nomic-embed-text` model, which produces 768-dimensional vectors.
   Setup:  install Ollama, then run:  ollama pull nomic-embed-text

2. "hashing" (automatic fallback)
   A dependency-free bag-of-words hashing embedder. It needs no server and no
   downloads, so the project always runs. Retrieval quality is weaker than real
   embeddings (it matches on shared words rather than meaning), but the whole
   pipeline behaves identically.

The backend actually in use is recorded in the vector store, so switching
backends rebuilds the index instead of mixing incompatible vectors.
"""

from __future__ import annotations

import hashlib

import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_DIM = 768
HASHING_DIM = 768  # keep the same width so the two backends are drop-in swappable


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Scale a vector to length 1 so dot product == cosine similarity."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class Embedder:
    """Chooses the best available embedding backend and exposes embed()."""

    def __init__(self, prefer_ollama: bool = True, timeout: float = 5.0):
        self.timeout = timeout
        self.backend = "hashing"
        self.dim = HASHING_DIM

        if prefer_ollama and self._ollama_available():
            self.backend = "ollama"
            self.dim = OLLAMA_DIM

    # ------------------------------------------------------------------ #
    def _ollama_available(self) -> bool:
        """One cheap probe request; if anything fails we use the fallback."""
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": "ping"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return len(resp.json()["embedding"]) == OLLAMA_DIM
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    def _embed_ollama(self, text: str) -> np.ndarray:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embedding"], dtype="float32")

    # ------------------------------------------------------------------ #
    def _embed_hashing(self, text: str) -> np.ndarray:
        """
        Map each word to a bucket via a stable hash and count it.
        Same words -> same buckets -> similar vectors. No model needed.
        """
        vec = np.zeros(HASHING_DIM, dtype="float32")
        for word in text.lower().split():
            word = word.strip(".,!?;:'\"()[]")
            if not word:
                continue
            digest = hashlib.md5(word.encode("utf-8")).hexdigest()
            vec[int(digest, 16) % HASHING_DIM] += 1.0
        return vec

    # ------------------------------------------------------------------ #
    def embed(self, text: str) -> np.ndarray:
        """Return a normalized 1-D float32 vector for `text`."""
        if not text.strip():
            return np.zeros(self.dim, dtype="float32")
        raw = (
            self._embed_ollama(text)
            if self.backend == "ollama"
            else self._embed_hashing(text)
        )
        return _normalize(raw.astype("float32"))

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into a 2-D array of shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return np.vstack([self.embed(t) for t in texts])
