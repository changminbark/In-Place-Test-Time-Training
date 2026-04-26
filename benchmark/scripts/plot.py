"""Plot benchmark results.

Reads per-example JSONL rows under --raw-root and emits PNGs under --out-dir:
  - accuracy_vs_context__<task>.png (one per task, lines per model/mode)
  - latency_vs_context.png            (lines per model/mode)
  - needle_heatmap__<model>__<mode>.png  (single_needle only)

Usage:
    python -m benchmark.scripts.plot
    python -m benchmark.scripts.plot --raw-root benchmark/results/raw/dev
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


def _load(raw_root: Path):
    data = defaultdict(list)
    for r in _iter_rows(raw_root):
        data[(r["task"], r["context_length_target"], r["mode"], r["model_name"])].append(r)
    return data


def plot_accuracy(data, out_dir: Path, tasks, ctx_lens, model_modes):
    import matplotlib.pyplot as plt

    for task in tasks:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (model, mode) in model_modes:
            xs, ys = [], []
            for ctx in ctx_lens:
                bucket = data.get((task, ctx, mode, model), [])
                if not bucket:
                    continue
                xs.append(ctx)
                ys.append(sum(int(r["correct"]) for r in bucket) / len(bucket))
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
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        out_path = out_dir / f"accuracy_vs_context__{task}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {out_path}")


def plot_latency(data, out_dir: Path, ctx_lens, model_modes):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_line = False
    for (model, mode) in model_modes:
        xs, ys = [], []
        for ctx in ctx_lens:
            # aggregate across all tasks at this ctx
            vals = []
            for key, bucket in data.items():
                task, c, m, md = key
                if c == ctx and m == mode and md == model:
                    vals.extend([r["latency_ms"] for r in bucket if r.get("latency_ms") is not None])
            if vals and mean(vals) > 0:
                xs.append(ctx)
                ys.append(mean(vals))
        if xs:
            any_line = True
            ax.plot(xs, ys, marker="o", label=f"{model} / {mode}")
    if not any_line:
        print("[plot] skip latency: all zero (stub predictor)")
        plt.close(fig)
        return
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ctx_lens)
    ax.set_xticklabels([str(c) for c in ctx_lens])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("mean latency (ms)")
    ax.set_title("Mean latency vs. context length")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "latency_vs_context.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_needle_heatmap(data, out_dir: Path, ctx_lens, model_modes):
    import matplotlib.pyplot as plt
    import numpy as np

    positions = ["early", "middle", "late"]
    for (model, mode) in model_modes:
        grid = np.full((len(positions), len(ctx_lens)), np.nan)
        for i, pos in enumerate(positions):
            for j, ctx in enumerate(ctx_lens):
                bucket = data.get(("single_needle", ctx, mode, model), [])
                subset = [r for r in bucket if r.get("metadata", {}).get("needle_position") == pos]
                if subset:
                    grid[i, j] = sum(int(r["correct"]) for r in subset) / len(subset)
        if np.all(np.isnan(grid)):
            continue
        fig, ax = plt.subplots(figsize=(7, 3.2))
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ctx_lens)))
        ax.set_xticklabels(ctx_lens)
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels(positions)
        for i in range(len(positions)):
            for j in range(len(ctx_lens)):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.5 else "black", fontsize=9)
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("needle position")
        ax.set_title(f"single_needle accuracy — {model} / {mode}")
        fig.colorbar(im, ax=ax, shrink=0.8, label="accuracy")
        fig.tight_layout()
        safe_model = model.replace("/", "_")
        out_path = out_dir / f"needle_heatmap__{safe_model}__{mode}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="benchmark/results/raw/dev")
    parser.add_argument("--out-dir", default="benchmark/results/plots")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load(raw_root)
    if not data:
        print(f"no rows under {raw_root}")
        return

    tasks = sorted({k[0] for k in data})
    ctx_lens = sorted({k[1] for k in data})
    model_modes = sorted({(k[3], k[2]) for k in data})

    plot_accuracy(data, out_dir, tasks, ctx_lens, model_modes)
    plot_latency(data, out_dir, ctx_lens, model_modes)
    plot_needle_heatmap(data, out_dir, ctx_lens, model_modes)
    print(f"[done] plots in {out_dir}")


if __name__ == "__main__":
    main()
