"""Run a predictor over a benchmark JSONL file and write a results JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .predictor import Predictor
from .scoring import score_example


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def run_benchmark(
    dataset_path: Path,
    results_path: Path,
    predictor: Predictor,
    max_new_tokens: int = 16,
    limit: Optional[int] = None,
    scoring_opts: Optional[dict] = None,
) -> dict:
    """Run `predictor` over every example in `dataset_path`.

    Writes one JSONL row per example to `results_path`. Returns a small in-memory
    summary (n, n_correct, mean_latency_ms).
    """
    dataset_path = Path(dataset_path)
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    scoring_opts = scoring_opts or {}

    n = 0
    n_correct = 0
    total_latency_ms = 0.0

    with results_path.open("w") as out:
        for example in _iter_jsonl(dataset_path):
            if limit is not None and n >= limit:
                break
            result = predictor.predict(example, max_new_tokens=max_new_tokens)
            correct = score_example(example, result.prediction, **scoring_opts)
            row = {
                "example_id": example["id"],
                "task": example["task"],
                "mode": predictor.mode,
                "model_name": predictor.model_name,
                "context_length_target": example["context_length_target"],
                "prediction": result.prediction,
                "ground_truth": example["answer"],
                "correct": bool(correct),
                "latency_ms": result.latency_ms,
                "ingest_latency_ms": result.ingest_latency_ms,
                "answer_latency_ms": result.answer_latency_ms,
                "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
                "metadata": example.get("metadata", {}),
            }
            out.write(json.dumps(row) + "\n")
            n += 1
            n_correct += int(correct)
            total_latency_ms += result.latency_ms

    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": (n_correct / n) if n else 0.0,
        "mean_latency_ms": (total_latency_ms / n) if n else 0.0,
        "results_path": str(results_path),
    }
