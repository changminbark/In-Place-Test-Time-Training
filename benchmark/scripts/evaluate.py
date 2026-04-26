"""Run a predictor over benchmark data and write per-example result rows.

This script intentionally does NOT import the TTT model. It accepts a predictor
factory path via --predictor (python module:function). The factory receives the
loaded config dict and returns a Predictor instance. This keeps the eval harness
decoupled from the model implementation.

Two built-in predictor factories:
  benchmark.eval.factories:echo_icl_factory
  benchmark.eval.factories:echo_ttt_factory

Usage:
    python -m benchmark.scripts.evaluate --profile dev --predictor benchmark.eval.factories:echo_icl_factory
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import yaml

from ..eval.runner import run_benchmark


def _load_predictor_factory(spec: str):
    module_name, fn_name = spec.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmark/configs/benchmark.yaml")
    parser.add_argument("--profile", choices=["dev", "full"], default="dev")
    parser.add_argument("--predictor", required=True,
                        help="module:function returning a Predictor (e.g. benchmark.eval.factories:echo_icl_factory)")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="max examples per file; useful for smoke tests")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(args.data_root or f"benchmark/data/{args.profile}")
    results_root = Path(args.results_root or f"benchmark/results/raw/{args.profile}")
    results_root.mkdir(parents=True, exist_ok=True)

    factory = _load_predictor_factory(args.predictor)
    predictor = factory(cfg)

    max_new_tokens = cfg["generation"]["max_new_tokens"]
    scoring_opts = {
        "lowercase": cfg["scoring"]["normalize_case"],
        "strip_whitespace": cfg["scoring"]["strip_whitespace"],
        "strip_punctuation": cfg["scoring"]["strip_punctuation"],
    }

    all_summaries = []
    for task in cfg["tasks"]:
        for target_tokens in cfg["context_lengths"]:
            dataset_path = data_root / f"{task}_{target_tokens}.jsonl"
            if not dataset_path.exists():
                print(f"[skip] missing {dataset_path}")
                continue
            results_path = (
                results_root
                / f"{predictor.model_name}__{predictor.mode}"
                / f"{task}_{target_tokens}.jsonl"
            )
            summary = run_benchmark(
                dataset_path=dataset_path,
                results_path=results_path,
                predictor=predictor,
                max_new_tokens=max_new_tokens,
                limit=args.limit,
                scoring_opts=scoring_opts,
            )
            print(
                f"[eval] {task:18s} @ {target_tokens:6d} tok   "
                f"n={summary['n']:4d}  acc={summary['accuracy']:.3f}  "
                f"mean_latency_ms={summary['mean_latency_ms']:.1f}"
            )
            all_summaries.append({
                "task": task,
                "context_length_target": target_tokens,
                "mode": predictor.mode,
                "model_name": predictor.model_name,
                **summary,
            })

    out = Path("benchmark/results/summary") / f"{predictor.model_name}__{predictor.mode}__{args.profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_summaries, indent=2))
    print(f"[done] summary written to {out}")


if __name__ == "__main__":
    main()
