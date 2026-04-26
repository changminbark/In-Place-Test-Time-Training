"""Offline smoke test: exercise generate → evaluate → aggregate end-to-end.

Generates a tiny RULER batch (one task, one short length, 3 samples) and runs
the EchoPredictor over it. Useful for catching plumbing breaks without burning
real-model compute.

Requires `third_party/RULER` initialised and PG-essay JSON downloaded — see
benchmark/spec.md for the one-time setup.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..data_gen import generate_examples
from ..eval.predictor import EchoPredictor
from ..eval.runner import run_benchmark


SMOKE_TASK = "niah_single_1"   # noise haystack → no PG-essay dep
SMOKE_LENGTH = 1024
SMOKE_N = 3
TOKENIZER = "google/gemma-3-1b-it"


def main() -> None:
    out_root = Path("benchmark/data/smoke")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    out_path = out_root / f"{SMOKE_TASK}_{SMOKE_LENGTH}.jsonl"
    with out_path.open("w") as f:
        for ex in generate_examples(
            task=SMOKE_TASK,
            target_tokens=SMOKE_LENGTH,
            num_samples=SMOKE_N,
            tokenizer_model_id=TOKENIZER,
            seed=7,
        ):
            f.write(json.dumps(ex) + "\n")
    print(f"[smoke] wrote {SMOKE_N} to {out_path}")

    results_root = Path("benchmark/results/raw/smoke")
    if results_root.exists():
        shutil.rmtree(results_root)
    predictor = EchoPredictor(model_name="echo", mode="icl")

    summary = run_benchmark(
        dataset_path=out_path,
        results_path=results_root / "echo__icl" / f"{SMOKE_TASK}_{SMOKE_LENGTH}.jsonl",
        predictor=predictor,
        max_new_tokens=16,
    )
    print(f"[smoke] {SMOKE_TASK}  acc={summary['accuracy']:.3f}  n={summary['n']}")
    assert summary["accuracy"] == 1.0, f"Echo should score 100%, got {summary['accuracy']}"
    print("[smoke] OK")


if __name__ == "__main__":
    main()
