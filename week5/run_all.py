"""
run_all.py
----------
Run the same request through all three architectures and save the outputs side
by side, so the difference in answer quality is easy to inspect.

    python run_all.py                 # all three tasks
    python run_all.py --task 3        # just one
    python run_all.py --prompt "..."  # your own question

Writing your own prompt is the interesting experiment: TASK-2's three branches
are hardcoded for the AT&T comparison question, so a different question exposes
exactly why runtime planning matters.

Results are written to outputs/.
"""

from __future__ import annotations

import argparse
import os
import time

import task1_llm_with_tools as task1
import task2_workflow as task2
import task3_planning_reasoning as task3

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

TASKS = {
    1: ("TASK-1 (one-shot)", task1),
    2: ("TASK-2 (hardcoded workflow)", task2),
    3: ("TASK-3 (planning + reasoning)", task3),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the week-5 agentic workflow tasks.")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], help="run only one task")
    parser.add_argument("--prompt", type=str, help="use a custom user prompt")
    parser.add_argument("--quiet", action="store_true", help="hide step-by-step output")
    args = parser.parse_args()

    prompt = args.prompt or task1.USER_PROMPT
    selected = [args.task] if args.task else [1, 2, 3]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []

    for number in selected:
        label, module = TASKS[number]
        started = time.time()

        try:
            result = module.run(prompt, verbose=not args.quiet)
            elapsed = time.time() - started

            path = os.path.join(OUTPUT_DIR, f"task{number}_output.md")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(f"# {label}\n\n**Prompt:** {prompt}\n\n---\n\n{result}\n")

            summary.append((label, len(result), elapsed, path))
        except Exception as exc:  # noqa: BLE001 — one task failing shouldn't stop the rest
            print(f"\n[{label} failed: {exc}]")
            summary.append((label, 0, time.time() - started, "failed"))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for label, length, elapsed, path in summary:
        print(f"  {label:<34} {length:>6} chars  {elapsed:>6.1f}s  -> {path}")
    print("\nCompare the files in outputs/ to see the quality difference.")


if __name__ == "__main__":
    main()
