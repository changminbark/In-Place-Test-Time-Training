"""Generate benchmark JSONL files for each (task, context length).

Each task is dispatched through `benchmark.data_gen.ruler_runner`, which
shells out to NVIDIA/RULER's synthetic generators (under `third_party/RULER`)
and remaps the output to our schema.

Usage:
    uv run python -m benchmark.scripts.generate --profile dev
    uv run python -m benchmark.scripts.generate --profile full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..data_gen import generate_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmark/configs/benchmark.yaml")
    parser.add_argument("--profile", choices=["dev", "full"], default="dev")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Override the config's task list (subset for quick runs).")
    parser.add_argument("--lengths", nargs="*", type=int, default=None,
                        help="Override the config's context lengths.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    n_examples = cfg["profiles"][args.profile]["examples_per_task_length"]
    tasks = args.tasks or cfg["tasks"]
    lengths = args.lengths or cfg["context_lengths"]
    seed = cfg["seed"]
    tokenizer_model_id = cfg["tokenizer_model_id"]

    out_root = Path(args.out_root or f"benchmark/data/{args.profile}")
    out_root.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        for target_tokens in lengths:
            out_path = out_root / f"{task}_{target_tokens}.jsonl"
            print(f"[gen] {task:18s} @ {target_tokens:6d}  →  {out_path}")
            with out_path.open("w") as f:
                for ex in generate_examples(
                    task=task,
                    target_tokens=target_tokens,
                    num_samples=n_examples,
                    tokenizer_model_id=tokenizer_model_id,
                    seed=seed,
                ):
                    f.write(json.dumps(ex) + "\n")

    print(f"[done] profile={args.profile} examples_per_task_length={n_examples}")


if __name__ == "__main__":
    main()
