"""Concrete Gemma3-TTT predictors that plug into the eval harness.

Exposes three factory functions consumed by `benchmark.scripts.evaluate` via
the `--predictor module:function` flag:

  - gemma3_icl_factory          → vanilla Gemma3 baseline (use_ttt=False)
  - gemma3_ttt_paper_factory    → TTT model, single-call eval (paper-style)
  - gemma3_ttt_strict_factory   → TTT model, two-phase ingest→answer eval

Configurable via env vars:
  - GEMMA3_BASE_MODEL_ID  (default: google/gemma-3-1b-it)
  - GEMMA3_TTT_MODEL_ID   (default: google/gemma-3-1b-it; override to a TTT
                            adapter checkpoint once trained)
  - HF_TOKEN              (or HUGGINGFACE_HUB_TOKEN; required for gated repos)

Or per-call via the benchmark config dict (`cfg["model_id"]`, `cfg["ttt_model_id"]`).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch

from .predictor import SinglePassPredictor, StrictTTTPredictor


_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")
DEFAULT_BASE_MODEL_ID = os.environ.get("GEMMA3_BASE_MODEL_ID", "google/gemma-3-1b-it")
DEFAULT_TTT_MODEL_ID = os.environ.get("GEMMA3_TTT_MODEL_ID", DEFAULT_BASE_MODEL_ID)


def _get_hf_token() -> Optional[str]:
    """Load HF token from env or project-root .env (no python-dotenv dep)."""
    for var in _TOKEN_ENV_VARS:
        v = os.environ.get(var)
        if v:
            return v
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() in _TOKEN_ENV_VARS:
                return v.strip().strip('"').strip("'")
    return None


def _device_for(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _peak_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**2


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def build_generate_subclass():
    """Subclass that injects `fast_weights` / `return_fast_weights` into the
    per-step kwargs that GenerationMixin.generate() builds via
    prepare_inputs_for_generation.

    Returned lazily so importing this module doesn't require torch/transformers
    until a factory is actually invoked.
    """
    from models.hf_gemma3.model_gemma3 import Gemma3ForCausalLMTTT

    class Gemma3ForCausalLMTTTGen(Gemma3ForCausalLMTTT):
        def prepare_inputs_for_generation(self, *args, **kwargs):
            fast_weights = kwargs.pop("fast_weights", None)
            return_fast_weights = kwargs.pop("return_fast_weights", False)
            out = super().prepare_inputs_for_generation(*args, **kwargs)
            if fast_weights is not None:
                out["fast_weights"] = fast_weights
            if return_fast_weights:
                out["return_fast_weights"] = True
            return out

    return Gemma3ForCausalLMTTTGen


def load_gemma3_ttt_model(
    repo_or_path: str,
    *,
    use_ttt: bool,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    ttt_layers=(0, 6, 12, 18, 24),
):
    """Load (model, tokenizer). Model is the generate-injection subclass."""
    from transformers import AutoTokenizer

    from models.hf_gemma3.config_gemma3 import Gemma3TTTConfig

    cls = build_generate_subclass()
    token = _get_hf_token()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cfg_kwargs = dict(use_ttt=use_ttt)
    if use_ttt:
        cfg_kwargs["ttt_layers"] = list(ttt_layers)
    cfg = Gemma3TTTConfig.from_pretrained(repo_or_path, token=token, **cfg_kwargs)

    model = cls.from_pretrained(
        repo_or_path, config=cfg, token=token, torch_dtype=torch_dtype
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(repo_or_path, token=token)
    return model, tokenizer


def make_generate_fn(model, tokenizer):
    """Returns generate_fn(prompt, max_new_tokens) -> (text, latency_ms, peak_mb).

    Used by both ICL and ttt_paper modes — they differ only in which model is loaded.
    """
    device = _device_for(model)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    @torch.inference_mode()
    def generate_fn(prompt: str, max_new_tokens: int):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        _reset_peak()
        t0 = time.perf_counter()
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        peak_mb = _peak_mb()
        new_tokens = out[0, ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, latency_ms, peak_mb

    return generate_fn


def make_strict_ttt_fns(model, tokenizer):
    """Builds (ingest_fn, answer_fn, reset_fn) for StrictTTTPredictor.

    Snapshot is moved to CPU between phases to free GPU memory; returned to the
    model device + dtype during the answer call.
    """
    device = _device_for(model)
    model_dtype = next(model.parameters()).dtype
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    @torch.inference_mode()
    def ingest_fn(document: str):
        ids = tokenizer(
            document, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        _reset_peak()
        t0 = time.perf_counter()
        out = model(input_ids=ids, return_fast_weights=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        peak_mb = _peak_mb()
        # Free the doc forward's transient memory before the answer phase by
        # parking the snapshot on CPU. Cheap because each tensor is ~d*d_ff.
        snapshot = {k: v.detach().to("cpu") for k, v in out.fast_weights.items()}
        return snapshot, latency_ms, peak_mb

    @torch.inference_mode()
    def answer_fn(prompt: str, snapshot, max_new_tokens: int):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gpu_snapshot = {
            k: v.to(device=device, dtype=model_dtype) for k, v in snapshot.items()
        }
        _reset_peak()
        t0 = time.perf_counter()
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            fast_weights=gpu_snapshot,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        peak_mb = _peak_mb()
        new_tokens = out[0, ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, latency_ms, peak_mb

    def reset_fn():
        # No persistent state on the model — each answer_fn call passes its own
        # snapshot through generate. Nothing to clear here.
        pass

    return ingest_fn, answer_fn, reset_fn


# ---------------------------------------------------------------------------
# Factory functions (for `--predictor benchmark.eval.gemma3_predictors:<name>`)
# ---------------------------------------------------------------------------

def _short_name(repo: str) -> str:
    return repo.rstrip("/").split("/")[-1]


def gemma3_icl_factory(cfg: dict):
    repo = cfg.get("model_id", DEFAULT_BASE_MODEL_ID)
    model, tok = load_gemma3_ttt_model(repo, use_ttt=False)
    return SinglePassPredictor(
        model_name=_short_name(repo),
        mode="icl",
        generate_fn=make_generate_fn(model, tok),
    )


def gemma3_ttt_paper_factory(cfg: dict):
    repo = cfg.get("ttt_model_id", DEFAULT_TTT_MODEL_ID)
    model, tok = load_gemma3_ttt_model(repo, use_ttt=True)
    return SinglePassPredictor(
        model_name=_short_name(repo) + "-ttt",
        mode="ttt_paper",
        generate_fn=make_generate_fn(model, tok),
    )


def gemma3_ttt_strict_factory(cfg: dict):
    repo = cfg.get("ttt_model_id", DEFAULT_TTT_MODEL_ID)
    model, tok = load_gemma3_ttt_model(repo, use_ttt=True)
    ingest_fn, answer_fn, reset_fn = make_strict_ttt_fns(model, tok)
    return StrictTTTPredictor(
        model_name=_short_name(repo) + "-ttt",
        ingest_fn=ingest_fn,
        answer_fn=answer_fn,
        reset_fn=reset_fn,
    )
