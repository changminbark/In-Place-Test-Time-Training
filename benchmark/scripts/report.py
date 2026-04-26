"""Produce a readable text summary of benchmark results.

Reads per-example JSONL result rows under --raw-root and prints:
  - overall table: task x context_length, rows per (mode, model)
  - needle-position breakdown for single_needle / multi_needle
  - latency summary if non-zero

Usage:
    python -m benchmark.scripts.report
    python -m benchmark.scripts.report --raw-root benchmark/results/raw/dev
    python -m benchmark.scripts.report --models gemma-3-1b-it --modes icl ttt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _iter_rows(root: Path):
    for path in sorted(root.rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _fmt_pct(v):
    return f"{100 * v:5.1f}%"


def _fmt_ms(v):
    if v is None or v == 0:
        return "    —"
    return f"{v:7.1f}"


def _table(title, rows, header, col_widths):
    lines = []
    lines.append(title)
    lines.append("-" * len(title))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    lines.append(fmt.format(*header))
    lines.append("  ".join("-" * w for w in col_widths))
    for r in rows:
        lines.append(fmt.format(*r))
    return "\n".join(lines)


def accuracy_table(data, models, modes, tasks, ctx_lens):
    rows = []
    for model in models:
        for mode in modes:
            for task in tasks:
                row = [f"{model} / {mode}", task]
                for ctx in ctx_lens:
                    bucket = data.get((task, ctx, mode, model), [])
                    if not bucket:
                        row.append("  —  ")
                    else:
                        acc = sum(int(r["correct"]) for r in bucket) / len(bucket)
                        row.append(_fmt_pct(acc))
                rows.append(row)
    header = ["model/mode", "task"] + [f"{c}" for c in ctx_lens]
    widths = [max(24, max((len(r[0]) for r in rows), default=0))] + [18] + [8] * len(ctx_lens)
    return _table("Accuracy by (model, mode, task) vs context length", rows, header, widths)


def latency_table(data, models, modes, tasks, ctx_lens):
    rows = []
    any_nonzero = False
    for model in models:
        for mode in modes:
            for task in tasks:
                row = [f"{model} / {mode}", task]
                for ctx in ctx_lens:
                    bucket = data.get((task, ctx, mode, model), [])
                    if not bucket:
                        row.append("  —  ")
                        continue
                    lats = [r["latency_ms"] for r in bucket if r.get("latency_ms") is not None]
                    if not lats:
                        row.append("  —  ")
                        continue
                    m = mean(lats)
                    if m > 0:
                        any_nonzero = True
                    row.append(_fmt_ms(m))
                rows.append(row)
    if not any_nonzero:
        return "Latency: all zero (stub predictor)"
    header = ["model/mode", "task"] + [f"{c}" for c in ctx_lens]
    widths = [max(24, max((len(r[0]) for r in rows), default=0))] + [18] + [9] * len(ctx_lens)
    return _table("Mean latency (ms) by (model, mode, task) vs context length", rows, header, widths)


def needle_position_table(data, models, modes, ctx_lens):
    """For single_needle only (clean 1:1 position mapping)."""
    rows = []
    for model in models:
        for mode in modes:
            for pos in ["early", "middle", "late"]:
                row = [f"{model} / {mode}", pos]
                for ctx in ctx_lens:
                    bucket = data.get(("single_needle", ctx, mode, model), [])
                    subset = [r for r in bucket if r.get("metadata", {}).get("needle_position") == pos]
                    if not subset:
                        row.append("  —  ")
                    else:
                        acc = sum(int(r["correct"]) for r in subset) / len(subset)
                        row.append(_fmt_pct(acc))
                rows.append(row)
    header = ["model/mode", "position"] + [f"{c}" for c in ctx_lens]
    widths = [max(24, max((len(r[0]) for r in rows), default=0))] + [10] + [8] * len(ctx_lens)
    return _table("single_needle accuracy by needle position", rows, header, widths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="benchmark/results/raw/dev")
    parser.add_argument("--models", nargs="*", default=None, help="filter models")
    parser.add_argument("--modes", nargs="*", default=None, help="filter modes")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        print(f"no results under {raw_root}")
        return

    # Group rows
    data = defaultdict(list)
    all_models, all_modes, all_tasks, all_ctx = set(), set(), set(), set()
    for r in _iter_rows(raw_root):
        key = (r["task"], r["context_length_target"], r["mode"], r["model_name"])
        data[key].append(r)
        all_models.add(r["model_name"])
        all_modes.add(r["mode"])
        all_tasks.add(r["task"])
        all_ctx.add(r["context_length_target"])

    models = sorted(args.models or all_models)
    modes = sorted(args.modes or all_modes)
    tasks = sorted(all_tasks)
    ctx_lens = sorted(all_ctx)

    total = sum(len(v) for v in data.values())
    print(f"# Benchmark report — {raw_root}")
    print(f"  rows: {total}  |  models: {models}  |  modes: {modes}  |  tasks: {tasks}")
    print(f"  context lengths: {ctx_lens}")
    print()
    print(accuracy_table(data, models, modes, tasks, ctx_lens))
    print()
    print(needle_position_table(data, models, modes, ctx_lens))
    print()
    print(latency_table(data, models, modes, tasks, ctx_lens))


if __name__ == "__main__":
    main()
