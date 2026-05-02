"""Plot benchmark results directly from aggregate.csv.

Useful when you only have the summary CSV (e.g. copied off Colab) and not the
raw per-example JSONLs. Emits:
  - accuracy_vs_context__<task>.png  (one per task; lines per model/mode)
  - accuracy_grid.png                (small-multiples; all tasks at once)
  - latency_vs_context.png           (mean latency, lines per model/mode)
  - peak_memory_vs_context.png       (mean peak GPU MB, lines per model/mode)
  - ingest_vs_answer__strict.png     (strict-mode ingest vs answer breakdown)

Usage:
    python -m benchmark.scripts.plot_csv \
        --csv benchmark/results/summary/aggregate.csv \
        --out-dir benchmark/results/plots
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def _load_rows(csv_path: Path):
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            r["context_length_target"] = int(r["context_length_target"])
            for k in ("accuracy", "mean_latency_ms", "mean_ingest_ms",
                      "mean_answer_ms", "peak_gpu_memory_mb"):
                v = r.get(k)
                r[k] = float(v) if v not in (None, "") else None
            r["n"] = int(r["n"])
            yield r


def _series(rows, value_key, model, mode, task=None):
    pts = []
    for r in rows:
        if r["model_name"] != model or r["mode"] != mode:
            continue
        if task is not None and r["task"] != task:
            continue
        v = r[value_key]
        if v is None:
            continue
        pts.append((r["context_length_target"], v))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_accuracy_per_task(rows, out_dir, tasks, model_modes, ctx_lens):
    import matplotlib.pyplot as plt
    for task in tasks:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, mode in model_modes:
            xs, ys = _series(rows, "accuracy", model, mode, task)
            if xs:
                ax.plot(xs, ys, marker="o", label=f"{model} / {mode}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ctx_lens)
        ax.set_xticklabels([str(c) for c in ctx_lens])
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Accuracy vs. context length — {task}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out = out_dir / f"accuracy_vs_context__{task}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {out}")


def plot_accuracy_grid(rows, out_dir, tasks, model_modes, ctx_lens):
    import matplotlib.pyplot as plt
    n = len(tasks)
    cols = 3
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.2, rows_n * 3.0),
                             sharex=True, sharey=True)
    axes = axes.flatten() if n > 1 else [axes]
    for ax, task in zip(axes, tasks):
        for model, mode in model_modes:
            xs, ys = _series(rows, "accuracy", model, mode, task)
            if xs:
                ax.plot(xs, ys, marker="o", label=f"{model}/{mode}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ctx_lens)
        ax.set_xticklabels([str(c) for c in ctx_lens], rotation=45, fontsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(task, fontsize=10)
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Accuracy vs. context length (all tasks)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = out_dir / "accuracy_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_metric_avg(rows, out_dir, model_modes, ctx_lens, value_key, ylabel,
                    title, fname, log_y=False):
    """Average a metric across tasks, plot vs. context length."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for model, mode in model_modes:
        bucket = defaultdict(list)
        for r in rows:
            if r["model_name"] != model or r["mode"] != mode:
                continue
            v = r[value_key]
            if v is None:
                continue
            bucket[r["context_length_target"]].append(v)
        if not bucket:
            continue
        xs = sorted(bucket)
        ys = [sum(bucket[c]) / len(bucket[c]) for c in xs]
        ax.plot(xs, ys, marker="o", label=f"{model} / {mode}")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(ctx_lens)
    ax.set_xticklabels([str(c) for c in ctx_lens])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = out_dir / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_strict_breakdown(rows, out_dir, ctx_lens):
    import matplotlib.pyplot as plt
    strict = [r for r in rows if r["mode"] == "ttt_strict"]
    if not strict:
        return
    bucket_ing = defaultdict(list)
    bucket_ans = defaultdict(list)
    for r in strict:
        if r["mean_ingest_ms"] is not None:
            bucket_ing[r["context_length_target"]].append(r["mean_ingest_ms"])
        if r["mean_answer_ms"] is not None:
            bucket_ans[r["context_length_target"]].append(r["mean_answer_ms"])
    xs = sorted(set(bucket_ing) | set(bucket_ans))
    if not xs:
        return
    ing = [sum(bucket_ing[c]) / len(bucket_ing[c]) if bucket_ing[c] else 0 for c in xs]
    ans = [sum(bucket_ans[c]) / len(bucket_ans[c]) if bucket_ans[c] else 0 for c in xs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    idx = range(len(xs))
    ax.bar([i - width / 2 for i in idx], ing, width, label="ingest")
    ax.bar([i + width / 2 for i in idx], ans, width, label="answer")
    ax.set_xticks(list(idx))
    ax.set_xticklabels([str(c) for c in xs])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("mean latency (ms)")
    ax.set_title("ttt_strict: ingest vs. answer latency (avg over tasks)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "ingest_vs_answer__strict.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="benchmark/results/summary/aggregate.csv")
    p.add_argument("--out-dir", default="benchmark/results/plots")
    p.add_argument("--drop-icl-zero", action="store_true",
                   help="drop any (task, context) cell where in_context accuracy is 0 "
                        "(suspected scoring/format bug)")
    args = p.parse_args()

    rows = list(_load_rows(Path(args.csv)))
    if not rows:
        print(f"no rows in {args.csv}")
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.drop_icl_zero:
        bad = {(r["task"], r["context_length_target"])
               for r in rows
               if r["mode"] == "in_context" and r["accuracy"] is not None and r["accuracy"] == 0.0}
        before = len(rows)
        rows = [r for r in rows if (r["task"], r["context_length_target"]) not in bad]
        print(f"[filter] dropped {len(bad)} (task, ctx) cells where in_context=0; rows {before} -> {len(rows)}")

    tasks = sorted({r["task"] for r in rows})
    ctx_lens = sorted({r["context_length_target"] for r in rows})
    model_modes = sorted({(r["model_name"], r["mode"]) for r in rows})

    plot_accuracy_per_task(rows, out_dir, tasks, model_modes, ctx_lens)
    plot_accuracy_grid(rows, out_dir, tasks, model_modes, ctx_lens)
    plot_metric_avg(rows, out_dir, model_modes, ctx_lens,
                    "mean_latency_ms", "mean latency (ms)",
                    "Mean latency vs. context length (avg over tasks)",
                    "latency_vs_context.png", log_y=True)
    plot_metric_avg(rows, out_dir, model_modes, ctx_lens,
                    "peak_gpu_memory_mb", "peak GPU memory (MB)",
                    "Peak GPU memory vs. context length (avg over tasks)",
                    "peak_memory_vs_context.png")
    plot_strict_breakdown(rows, out_dir, ctx_lens)
    print(f"[done] plots in {out_dir}")


if __name__ == "__main__":
    main()
