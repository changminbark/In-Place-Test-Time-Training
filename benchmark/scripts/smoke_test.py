"""Offline smoke test: generate, evaluate (echo), aggregate, all with a fake tokenizer.

Does not require HF access or a real model. Writes to benchmark/data/smoke and
benchmark/results/raw/smoke to avoid stomping on real runs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..data_gen import gen_multi_needle, gen_single_needle, gen_variable_tracking
from ..eval.predictor import EchoPredictor
from ..eval.runner import run_benchmark


class WhitespaceTokenizer:
    """~1 token per whitespace-split word. Good enough to verify token budgeter logic."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return text.split()

    def decode(self, ids, skip_special_tokens: bool = True):
        return " ".join(ids)


def main() -> None:
    tok = WhitespaceTokenizer()
    root = Path("benchmark/data/smoke")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    configs = [
        ("single_needle", 256, lambda i: gen_single_needle(
            tokenizer=tok, target_tokens=256, position="middle", seed=7, example_idx=i)),
        ("multi_needle", 256, lambda i: gen_multi_needle(
            tokenizer=tok, target_tokens=256, position_1="early", position_2="late",
            seed=7, example_idx=i)),
        ("variable_tracking", 256, lambda i: gen_variable_tracking(
            tokenizer=tok, target_tokens=256, seed=7, example_idx=i)),
    ]

    n_per = 5
    for task, ctx, gen in configs:
        path = root / f"{task}_{ctx}.jsonl"
        with path.open("w") as f:
            for i in range(n_per):
                f.write(json.dumps(gen(i)) + "\n")
        print(f"[smoke] wrote {n_per} to {path}")

    # Evaluate with EchoPredictor (should score 100%).
    results_root = Path("benchmark/results/raw/smoke")
    if results_root.exists():
        shutil.rmtree(results_root)
    predictor = EchoPredictor(model_name="echo", mode="icl")

    for task, ctx, _ in configs:
        dataset_path = root / f"{task}_{ctx}.jsonl"
        results_path = results_root / "echo__icl" / f"{task}_{ctx}.jsonl"
        summary = run_benchmark(
            dataset_path=dataset_path,
            results_path=results_path,
            predictor=predictor,
            max_new_tokens=16,
        )
        print(f"[smoke] {task:18s} acc={summary['accuracy']:.3f} n={summary['n']}")
        assert summary["accuracy"] == 1.0, f"EchoPredictor should score 100%, got {summary['accuracy']}"

    print("[smoke] OK")


if __name__ == "__main__":
    main()
