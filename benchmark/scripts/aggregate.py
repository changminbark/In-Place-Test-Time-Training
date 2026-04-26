"""Aggregate per-example results into a summary CSV grouped by
(task, context_length_target, mode, model_name).

Reads every *.jsonl under a raw-results root and writes one CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _iter_result_rows(root: Path):
    for path in sorted(root.rglob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="benchmark/results/raw")
    parser.add_argument("--out", default="benchmark/results/summary/aggregate.csv")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    buckets = defaultdict(list)

    for row in _iter_result_rows(raw_root):
        key = (
            row["task"],
            row["context_length_target"],
            row["mode"],
            row["model_name"],
        )
        buckets[key].append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "task", "context_length_target", "mode", "model_name",
            "n", "accuracy", "mean_latency_ms", "mean_ingest_ms",
            "mean_answer_ms", "peak_gpu_memory_mb",
        ])
        for key, rows in sorted(buckets.items()):
            n = len(rows)
            acc = sum(int(r["correct"]) for r in rows) / n
            lat = mean(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
            ing = [r["ingest_latency_ms"] for r in rows if r.get("ingest_latency_ms") is not None]
            ans = [r["answer_latency_ms"] for r in rows if r.get("answer_latency_ms") is not None]
            peaks = [r["peak_gpu_memory_mb"] for r in rows if r.get("peak_gpu_memory_mb") is not None]
            w.writerow([
                *key,
                n,
                f"{acc:.4f}",
                f"{lat:.2f}",
                f"{mean(ing):.2f}" if ing else "",
                f"{mean(ans):.2f}" if ans else "",
                f"{max(peaks):.1f}" if peaks else "",
            ])
    print(f"[agg] wrote {out_path}")


if __name__ == "__main__":
    main()
