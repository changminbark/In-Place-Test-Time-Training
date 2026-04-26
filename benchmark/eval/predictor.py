"""Predictor abstraction.

The eval harness does not depend on any particular model implementation. A
concrete predictor is plugged in by the caller (e.g. vanilla HF Gemma 3 for ICL,
or the TTT-enabled model for ttt_paper / ttt_strict).

Three modes are supported:

- "icl"        — vanilla baseline; single-forward; prompt = doc + question.
- "ttt_paper"  — paper-style TTT; single-forward; prompt = doc + question;
                 fast weights update during prefill of that prompt and reset
                 between examples. Wired exactly like ICL but the model has
                 use_ttt=True.
- "ttt_strict" — stricter two-phase TTT; ingest(document) snapshots fast
                 weights, then answer(question_only) uses the snapshot.
                 Document is NOT in the answer prompt.

Both ICL and ttt_paper use the single-call SinglePassPredictor. ttt_strict
uses the two-phase StrictTTTPredictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class PredictionResult:
    prediction: str
    latency_ms: float
    ingest_latency_ms: Optional[float] = None
    answer_latency_ms: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None


class Predictor(Protocol):
    """Common interface. `predict` takes a full example and returns one answer."""

    model_name: str
    mode: str  # "icl" | "ttt_paper" | "ttt_strict"

    def predict(self, example: dict, max_new_tokens: int = 16) -> PredictionResult:
        ...


class SinglePassPredictor:
    """One generate call. Prompt = doc + question. Used for both 'icl' and 'ttt_paper'.

    The difference between modes is purely the underlying generate_fn (vanilla
    model vs use_ttt=True model). Mode is recorded for result aggregation only.
    """

    def __init__(
        self,
        model_name: str,
        mode: str,
        generate_fn,
        prompt_template: Optional[str] = None,
    ):
        """
        generate_fn: callable(prompt: str, max_new_tokens: int)
                       -> (text: str, latency_ms: float, peak_mb: float|None)
        """
        if mode not in ("icl", "ttt_paper"):
            raise ValueError(f"SinglePassPredictor mode must be icl|ttt_paper, got {mode!r}")
        self.model_name = model_name
        self.mode = mode
        self._generate = generate_fn
        self.prompt_template = prompt_template or (
            "You are given a document and a question. Answer with just the value, "
            "no explanation.\n\n"
            "Document:\n{document}\n\n"
            "Question: {question}\nAnswer:"
        )

    def predict(self, example: dict, max_new_tokens: int = 16) -> PredictionResult:
        prompt = self.prompt_template.format(
            document=example["document"], question=example["question"]
        )
        text, latency_ms, peak_mb = self._generate(prompt, max_new_tokens)
        return PredictionResult(
            prediction=text,
            latency_ms=latency_ms,
            answer_latency_ms=latency_ms,
            peak_gpu_memory_mb=peak_mb,
        )


class StrictTTTPredictor:
    """Two-phase strict TTT. Ingest document → snapshot fast weights →
    answer question with snapshot, no document in the answer prompt.

    Requires a model implementation that exposes:
      ingest_fn(document) -> (state, latency_ms, peak_mb)
        Runs a forward over the document only. Returns an opaque state handle
        that captures the per-layer cumulative ΔW snapshots.

      answer_fn(question_prompt, state, max_new_tokens)
                  -> (text, latency_ms, peak_mb)
        Runs a separate forward/generate over the question only, with `state`
        patched into the model so TTT layers use the snapshotted ΔW instead of
        recomputing.

      reset_fn() (optional) — restore base fast weights before the next example
        (no-op if the model already does this implicitly per-call).
    """

    mode = "ttt_strict"

    def __init__(
        self,
        model_name: str,
        ingest_fn,
        answer_fn,
        reset_fn=None,
        prompt_template: Optional[str] = None,
    ):
        self.model_name = model_name
        self._ingest = ingest_fn
        self._answer = answer_fn
        self._reset = reset_fn
        self.prompt_template = prompt_template or (
            "Answer the following question with just the value, no explanation.\n\n"
            "Question: {question}\nAnswer:"
        )

    def predict(self, example: dict, max_new_tokens: int = 16) -> PredictionResult:
        if self._reset is not None:
            self._reset()
        state, ingest_ms, ingest_peak = self._ingest(example["document"])
        prompt = self.prompt_template.format(question=example["question"])
        text, answer_ms, answer_peak = self._answer(prompt, state, max_new_tokens)
        total_ms = ingest_ms + answer_ms
        peaks = [x for x in (ingest_peak, answer_peak) if x is not None]
        peak = max(peaks) if peaks else None
        return PredictionResult(
            prediction=text,
            latency_ms=total_ms,
            ingest_latency_ms=ingest_ms,
            answer_latency_ms=answer_ms,
            peak_gpu_memory_mb=peak,
        )


class EchoPredictor:
    """Dry-run predictor. Returns the ground-truth answer with zero latency.

    Useful for verifying the eval harness end-to-end before a real model is wired in.
    """

    def __init__(self, model_name: str = "echo", mode: str = "icl"):
        if mode not in ("icl", "ttt_paper", "ttt_strict"):
            raise ValueError(f"EchoPredictor mode must be icl|ttt_paper|ttt_strict, got {mode!r}")
        self.model_name = model_name
        self.mode = mode

    def predict(self, example: dict, max_new_tokens: int = 16) -> PredictionResult:
        if self.mode == "ttt_strict":
            return PredictionResult(
                prediction=example["answer"],
                latency_ms=0.0,
                ingest_latency_ms=0.0,
                answer_latency_ms=0.0,
                peak_gpu_memory_mb=None,
            )
        return PredictionResult(
            prediction=example["answer"],
            latency_ms=0.0,
            answer_latency_ms=0.0,
            peak_gpu_memory_mb=None,
        )
