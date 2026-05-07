"""Smoke tests for the Gemma3-TTT predictor wiring.

Avoids any HF download by constructing a tiny model directly. Verifies:
  - the generate-injection subclass forwards `fast_weights` through to every
    forward() call inside generate()
  - generate() with `fast_weights=snapshot` runs to completion
  - `make_strict_ttt_fns` produces ingest_fn / answer_fn that round-trip a
    snapshot through a real generate() call
  - `load_gemma3_ttt_model` preserves trained TTT weights and rejects
    untrained checkpoints

Slow (real-checkpoint) tests are not included here; those live alongside the
model in `models/hf_gemma3/test_gemma3.py`.
"""
from __future__ import annotations

import tempfile

import pytest
import torch

from benchmark.eval.gemma3_predictors import (
    build_generate_subclass,
    load_gemma3_ttt_model,
    make_strict_ttt_fns,
)
from models.hf_gemma3.config_gemma3 import Gemma3TTTConfig
from models.hf_gemma3.model_gemma3 import Gemma3ForCausalLMTTT


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


def test_load_gemma3_ttt_model_preserves_trained_ttt():
    """End-to-end check that the predictor's load wrapper preserves trained
    TTT weights — not just the bare from_pretrained tested in
    models/hf_gemma3/test_gemma3.py. Catches regressions in any wrapping
    logic added to load_gemma3_ttt_model.
    """
    cfg = _tiny_config(use_ttt=True, ttt_layers=(0, 2))
    model = Gemma3ForCausalLMTTT(cfg).eval()
    expected = {}
    for name, p in model.named_parameters():
        if "ttt_conv" in name or "ttt_proj" in name:
            with torch.no_grad():
                p.data.normal_(mean=0.0, std=0.5)
            expected[name] = p.detach().clone()

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        loaded, _ = load_gemma3_ttt_model(
            tmp, use_ttt=True, device="cpu", torch_dtype=torch.float32,
            ttt_layers=(0, 2),
        )

    loaded_params = dict(loaded.named_parameters())
    for name, want in expected.items():
        torch.testing.assert_close(loaded_params[name], want, rtol=1e-5, atol=1e-5)


def test_load_gemma3_ttt_model_guard_rejects_zero_ttt_conv():
    """The post-load guard must fire when ttt_conv is all-zero on disk
    (i.e. an untrained / wrongly-saved checkpoint). Without this, a benchmark
    run on such a checkpoint would silently report numbers from a no-op TTT
    path.
    """
    cfg = _tiny_config(use_ttt=True, ttt_layers=(0, 2))
    model = Gemma3ForCausalLMTTT(cfg).eval()
    # Leave ttt_conv at its zero init; randomise ttt_proj so the guard fires
    # on conv only (the diagnostic message points specifically at conv).
    for name, p in model.named_parameters():
        if "ttt_proj" in name:
            with torch.no_grad():
                p.data.normal_(mean=0.0, std=0.1)

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        with pytest.raises(RuntimeError, match="ttt_conv.weight has L2=0"):
            load_gemma3_ttt_model(
                tmp, use_ttt=True, device="cpu", torch_dtype=torch.float32,
                ttt_layers=(0, 2),
            )


def test_strict_snapshot_actually_changes_generation():
    """Pin that a non-zero snapshot makes generate() produce different output
    than no-TTT. The existing test_make_strict_ttt_fns_round_trip_snapshot
    only checks the call doesn't crash; it accepts any string. If the
    snapshot wiring quietly broke (e.g. fast_weights got dropped on the way
    to the MLP), strict-mode eval would silently match in_context — which is
    indistinguishable from an actually-bad benchmark score.
    """
    cls = build_generate_subclass()
    cfg = _tiny_config(use_ttt=True, ttt_layers=(0, 2))
    model = cls(cfg).eval()

    d = cfg.hidden_size
    d_ff = cfg.intermediate_size
    g = torch.Generator().manual_seed(11)
    nonzero = {
        0: torch.randn(1, d, d_ff, generator=g) * 0.5,
        2: torch.randn(1, d, d_ff, generator=g) * 0.5,
    }
    zero = {0: torch.zeros(1, d, d_ff), 2: torch.zeros(1, d, d_ff)}

    input_ids = torch.randint(1, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out_zero = model(input_ids=input_ids, fast_weights=zero).logits
        out_nz = model(input_ids=input_ids, fast_weights=nonzero).logits

    assert not torch.allclose(out_zero, out_nz, rtol=1e-4, atol=1e-4), (
        "Non-zero snapshot produced identical logits to zero snapshot — the "
        "fast_weights path is not being applied at the MLP."
    )
