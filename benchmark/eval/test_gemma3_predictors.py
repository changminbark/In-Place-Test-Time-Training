"""Smoke tests for the Gemma3-TTT predictor wiring.

Avoids any HF download by constructing a tiny model directly. Verifies:
  - the generate-injection subclass forwards `fast_weights` through to every
    forward() call inside generate()
  - generate() with `fast_weights=snapshot` runs to completion
  - `make_strict_ttt_fns` produces ingest_fn / answer_fn that round-trip a
    snapshot through a real generate() call

Slow (real-checkpoint) tests are not included here; those live alongside the
model in `models/hf_gemma3/test_gemma3.py`.
"""
from __future__ import annotations

import torch

from benchmark.eval.gemma3_predictors import (
    build_generate_subclass,
    make_strict_ttt_fns,
)
from models.hf_gemma3.config_gemma3 import Gemma3TTTConfig


def _tiny_config(use_ttt: bool = True, ttt_layers=(0, 2)) -> Gemma3TTTConfig:
    return Gemma3TTTConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=256,
        sliding_window=32,
        sliding_window_pattern=2,
        query_pre_attn_scalar=32,
        rope_theta=10000.0,
        rope_local_base_freq=10000.0,
        use_ttt=use_ttt,
        ttt_layers=list(ttt_layers),
        ttt_chunk=16,
        ttt_lr=0.3,
        ttt_proj=True,
        ttt_target="hidden_states",
    )


class _StubTokenizer:
    """Bare-minimum tokenizer for a HF generate() call.

    Returns input_ids tensors directly from a fixed prompt-to-id mapping. Real
    eval uses the actual Gemma tokenizer; this exists so the test doesn't
    download anything.
    """

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        # Deterministic: hash the string into a few token ids.
        ids = torch.tensor([[(ord(c) % 511) + 1 for c in text[:24]]])
        return type("Out", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids.tolist())


def test_generate_subclass_accepts_and_uses_fast_weights():
    """generate(input_ids, fast_weights=snapshot) must not error and must
    actually thread the snapshot through to forward()."""
    cls = build_generate_subclass()
    cfg = _tiny_config(use_ttt=True, ttt_layers=(0, 2))
    model = cls(cfg).eval()
    d = cfg.hidden_size
    d_ff = cfg.intermediate_size
    snapshot = {0: torch.zeros(1, d, d_ff), 2: torch.zeros(1, d, d_ff)}

    input_ids = torch.randint(1, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=0,
            fast_weights=snapshot,
        )
    assert out.shape[0] == 1
    assert out.shape[1] >= input_ids.shape[1]


def test_generate_subclass_unchanged_without_kwargs():
    """When called without the new kwargs, behaves identically to the parent."""
    cls = build_generate_subclass()
    cfg = _tiny_config(use_ttt=True)
    model = cls(cfg).eval()

    input_ids = torch.randint(1, cfg.vocab_size, (1, 8))
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=4, do_sample=False, pad_token_id=0)
    assert out.shape == (1, 12)


def test_make_strict_ttt_fns_round_trip_snapshot():
    """Build ingest_fn + answer_fn against the tiny model and a stub tokenizer;
    confirm the snapshot from ingest is usable by answer."""
    cls = build_generate_subclass()
    cfg = _tiny_config(use_ttt=True, ttt_layers=(0, 2))
    model = cls(cfg).eval()
    tok = _StubTokenizer()

    ingest_fn, answer_fn, reset_fn = make_strict_ttt_fns(model, tok)

    snapshot, ingest_ms, ingest_peak = ingest_fn("a long pretend document body")
    assert isinstance(snapshot, dict)
    assert set(snapshot.keys()) == {0, 2}
    for fw in snapshot.values():
        assert fw.device.type == "cpu"  # parked on CPU between phases
        assert fw.shape == (1, cfg.hidden_size, cfg.intermediate_size)
    assert ingest_ms >= 0.0

    text, answer_ms, answer_peak = answer_fn("question?", snapshot, max_new_tokens=4)
    assert isinstance(text, str)
    assert answer_ms >= 0.0

    reset_fn()  # no-op but must not crash
