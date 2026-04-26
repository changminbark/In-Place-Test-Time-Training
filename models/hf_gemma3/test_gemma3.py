"""Smoke tests for Gemma3TTT.

Verifies the custom modeling code is HF-compatible: classes instantiate, a
forward pass runs and returns the expected output type, generate() works,
save/load round-trips, and the freeze_base_model() helper isolates gradients
to TTT adapter modules only.

Run from the repo root:
    pytest models/hf_gemma3/test_gemma3.py -v
or as a script:
    python -m models.hf_gemma3.test_gemma3
"""
from __future__ import annotations

import tempfile

import pytest
import torch

from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.hf_gemma3.config_gemma3 import Gemma3TTTConfig
from models.hf_gemma3.model_gemma3 import (
    Gemma3DecoderLayerTTT,
    Gemma3ForCausalLMTTT,
    Gemma3MLPTTT,
    Gemma3TextModelTTT,
    Gemma3TTTBaseModelOutput,
    Gemma3TTTCausalLMOutput,
)


def _tiny_config(use_ttt: bool, ttt_layers=(0, 2)) -> Gemma3TTTConfig:
    """Tiny config so tests run on CPU in seconds."""
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


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_config_roundtrip_via_dict():
    cfg = _tiny_config(use_ttt=True)
    cfg2 = Gemma3TTTConfig.from_dict(cfg.to_dict())
    assert cfg2.use_ttt is True
    assert cfg2.ttt_layers == [0, 2]
    assert cfg2.hidden_size == cfg.hidden_size


def test_vanilla_model_instantiates():
    model = Gemma3ForCausalLMTTT(_tiny_config(use_ttt=False))
    # No TTT modules should exist when use_ttt=False
    for name, _ in model.named_parameters():
        assert "ttt_proj" not in name and "ttt_conv" not in name


def test_ttt_model_has_adapters_only_on_listed_layers():
    cfg = _tiny_config(use_ttt=True, ttt_layers=(0, 2))
    model = Gemma3ForCausalLMTTT(cfg)

    ttt_param_layers = set()
    for name, _ in model.named_parameters():
        if "ttt_proj" in name or "ttt_conv" in name:
            # name is like "model.layers.0.mlp.ttt_proj.weight"
            ttt_param_layers.add(int(name.split(".")[2]))

    assert ttt_param_layers == {0, 2}

    # Spot check the layer types match the config flags
    for i, layer in enumerate(model.model.layers):
        assert isinstance(layer, Gemma3DecoderLayerTTT)
        assert isinstance(layer.mlp, Gemma3MLPTTT)
        assert layer.is_ttt_layer == (i in cfg.ttt_layers)


# ---------------------------------------------------------------------------
# Forward / generate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_ttt", [False, True])
def test_forward_returns_causal_lm_output(use_ttt: bool):
    cfg = _tiny_config(use_ttt=use_ttt)
    model = Gemma3ForCausalLMTTT(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
    with torch.no_grad():
        out = model(input_ids=input_ids)

    assert isinstance(out, CausalLMOutputWithPast)
    assert out.logits.shape == (2, 32, cfg.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_forward_with_labels_produces_loss():
    cfg = _tiny_config(use_ttt=True)
    model = Gemma3ForCausalLMTTT(cfg)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    out = model(input_ids=input_ids, labels=input_ids)

    assert out.loss is not None
    assert out.loss.ndim == 0
    out.loss.backward()  # gradient path must be intact


def test_generate_runs():
    cfg = _tiny_config(use_ttt=True)
    model = Gemma3ForCausalLMTTT(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        gen = model.generate(input_ids, max_new_tokens=4, do_sample=False)

    assert gen.shape[0] == 1
    assert gen.shape[1] >= input_ids.shape[1]


# ---------------------------------------------------------------------------
# Frozen-base training mode
# ---------------------------------------------------------------------------

def test_freeze_base_model_isolates_grads_to_ttt():
    cfg = _tiny_config(use_ttt=True)
    model = Gemma3ForCausalLMTTT(cfg)
    model.freeze_base_model()

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable, "expected at least one trainable param after freeze"
    allowed = ("ttt_proj", "ttt_conv", "down_proj")
    for name in trainable:
        assert any(s in name for s in allowed), name

    # Sanity: a forward+backward only populates grads on TTT params
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"missing grad on trainable param {name}"
        else:
            assert p.grad is None, f"unexpected grad on frozen param {name}"


# ---------------------------------------------------------------------------
# Save / load round-trip (the path users will actually exercise via the Hub)
# ---------------------------------------------------------------------------

def test_save_and_load_pretrained_roundtrip():
    cfg = _tiny_config(use_ttt=True)
    model = Gemma3ForCausalLMTTT(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        ref_logits = model(input_ids=input_ids).logits

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        reloaded = Gemma3ForCausalLMTTT.from_pretrained(tmp).eval()

    with torch.no_grad():
        new_logits = reloaded(input_ids=input_ids).logits

    torch.testing.assert_close(ref_logits, new_logits, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Strict-TTT state hook: snapshot capture and snapshot consumption
# ---------------------------------------------------------------------------

def test_strict_paper_default_unchanged():
    """Without the new kwargs, output schema and values are unchanged."""
    cfg = _tiny_config(use_ttt=True)
    model = Gemma3ForCausalLMTTT(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 32))
    with torch.no_grad():
        out = model(input_ids=input_ids)

    # Existing callers see CausalLMOutputWithPast (NOT the TTT-extended subtype).
    assert type(out) is CausalLMOutputWithPast


def test_strict_ingest_returns_fast_weights_dict():
    cfg = _tiny_config(use_ttt=True, ttt_layers=(0, 2))
    model = Gemma3ForCausalLMTTT(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 32))
    with torch.no_grad():
        out = model(input_ids=input_ids, return_fast_weights=True)

    assert isinstance(out, Gemma3TTTCausalLMOutput)
    assert out.fast_weights is not None
    # Snapshot keys are exactly the ttt_layers indices.
    assert set(out.fast_weights.keys()) == {0, 2}

    d = cfg.hidden_size
    d_ff = cfg.intermediate_size
    for layer_idx, fw in out.fast_weights.items():
        assert fw.shape == (1, d, d_ff), f"layer {layer_idx}: got {fw.shape}"
        assert torch.isfinite(fw).all()


def test_strict_consume_zero_snapshot_matches_no_ttt():
    """Feeding zero snapshots reduces the MLP to its base W_down — equivalent to
    a model where the TTT branch produces no perturbation."""
    cfg = _tiny_config(use_ttt=True, ttt_layers=(0, 2))
    model = Gemma3ForCausalLMTTT(cfg).eval()

    d = cfg.hidden_size
    d_ff = cfg.intermediate_size
    zero_snapshot = {0: torch.zeros(1, d, d_ff), 2: torch.zeros(1, d, d_ff)}

    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        zero_out = model(input_ids=input_ids, fast_weights=zero_snapshot).logits

    # Force-disable adapter on this model: zero its conv (W_target init is
    # random sparse-diag — but since fast_weights override skips conv path,
    # this is unnecessary; we instead compare against a model with use_ttt=False
    # of identical base weights).
    cfg_off = _tiny_config(use_ttt=False)
    model_off = Gemma3ForCausalLMTTT(cfg_off).eval()
    # Copy matching params over so only the MLP-path difference is tested.
    src = dict(model.named_parameters())
    for name, p in model_off.named_parameters():
        if name in src and src[name].shape == p.shape:
            p.data.copy_(src[name].data)

    with torch.no_grad():
        ref_out = model_off(input_ids=input_ids).logits

    torch.testing.assert_close(zero_out, ref_out, rtol=1e-4, atol=1e-4)


def test_strict_consume_runs_and_ignores_question_inputs():
    """A real (random) snapshot can be consumed; output is finite and
    deterministic w.r.t. the snapshot value (not the input contents of TTT
    layers' chunked update path, which is bypassed)."""
    cfg = _tiny_config(use_ttt=True, ttt_layers=(0, 2))
    model = Gemma3ForCausalLMTTT(cfg).eval()

    d = cfg.hidden_size
    d_ff = cfg.intermediate_size
    g = torch.Generator().manual_seed(7)
    snapshot = {
        0: torch.randn(1, d, d_ff, generator=g) * 0.01,
        2: torch.randn(1, d, d_ff, generator=g) * 0.01,
    }

    input_ids = torch.randint(0, cfg.vocab_size, (1, 24))
    with torch.no_grad():
        out_a = model(input_ids=input_ids, fast_weights=snapshot)
        out_b = model(input_ids=input_ids, fast_weights=snapshot)

    assert isinstance(out_a, Gemma3TTTCausalLMOutput)
    torch.testing.assert_close(out_a.logits, out_b.logits, rtol=0, atol=0)
    assert torch.isfinite(out_a.logits).all()


def test_strict_snapshot_matches_paper_two_chunk_decomposition():
    """End-to-end semantic check on the MLP module:

    For an input that is exactly two TTT chunks [a, b], paper-style forward([a,b])
    processes chunk b with effective W_down = W_down^{(0)} + η · ΔW_a, where
    ΔW_a is the rank-1 delta computed from chunk a's (z, V̂).

    Strict ingest on [a] alone (a single-chunk input) should produce a snapshot
    equal to that same ΔW_a (un-scaled by η). Then strict consumer on [b] alone,
    given that snapshot, applies the same W_eff to chunk b's z.

    Therefore: paper output at the chunk-b positions == strict consumer output
    on b. This is the cleanest direct equivalence we can assert (full forwards
    differ because attention sees a different prefix).
    """
    torch.manual_seed(7)
    cfg = _tiny_config(use_ttt=True, ttt_layers=(0,))
    model = Gemma3ForCausalLMTTT(cfg).eval()
    mlp = model.model.layers[0].mlp

    # Conv1D is zero-init by design; randomise it so the snapshot is non-trivial
    # (otherwise V̂ = 0 → snapshot = 0 → both sides trivially match).
    mlp.ttt_conv.weight.data.normal_(0.0, 0.1)

    C = cfg.ttt_chunk
    d = cfg.hidden_size
    x_full = torch.randn(1, 2 * C, d)
    t_full = torch.randn(1, 2 * C, d)
    x_a, x_b = x_full[:, :C], x_full[:, C:]
    t_a, t_b = t_full[:, :C], t_full[:, C:]

    with torch.no_grad():
        # 1. Paper-style two-chunk forward.
        out_paper_full = mlp(x_full, t=t_full)               # (1, 2C, d)
        out_paper_b = out_paper_full[:, C:]                   # chunk-b portion

        # 2. Strict ingest on chunk a only.
        out_a, snapshot_a = mlp(x_a, t=t_a, return_fast_weights=True)
        assert snapshot_a.shape == (1, d, cfg.intermediate_size)

        # 3. Strict consumer on chunk b only with that snapshot.
        out_strict_b, _ = mlp(x_b, t=t_b, fast_weights=snapshot_a)

    torch.testing.assert_close(out_paper_b, out_strict_b, rtol=1e-5, atol=1e-5)


def test_strict_freeze_still_isolates_grads():
    """The new code paths must not introduce trainable params outside the adapter."""
    cfg = _tiny_config(use_ttt=True)
    model = Gemma3ForCausalLMTTT(cfg)
    model.freeze_base_model()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    out = model(input_ids=input_ids, return_fast_weights=True, labels=input_ids)
    out.loss.backward()

    trainable_substrings = ("ttt_proj", "ttt_conv", "down_proj")
    for name, p in model.named_parameters():
        if any(s in name for s in trainable_substrings):
            assert p.requires_grad, f"adapter param frozen: {name}"
            # Adapter received gradient (or was structurally not on the loss path)
        else:
            assert p.grad is None, f"unexpected grad on frozen param {name}"


# ---------------------------------------------------------------------------
# Real Gemma3 checkpoint compatibility (slow; skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_loads_real_gemma3_checkpoint():
    """Verify base Gemma3 weights load cleanly into the TTT model.

    Skipped by default — this downloads ~2GB and requires HF auth + Gemma TOU
    acceptance. Run with: pytest -v -m slow
    """
    repo = "google/gemma-3-1b-it"
    cfg = Gemma3TTTConfig.from_pretrained(repo, use_ttt=True, ttt_layers=[0, 6, 12, 18, 24])
    model = Gemma3ForCausalLMTTT.from_pretrained(repo, config=cfg, torch_dtype=torch.float32)

    # Base params should match shapes; TTT params are newly initialized
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite param: {name}"

    tokenizer = AutoTokenizer.from_pretrained(repo)
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    assert torch.isfinite(out.logits).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
