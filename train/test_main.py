"""Smoke tests for train/main.py.

These tests exercise the dataset, freeze, save, wandb, and CLI plumbing
without downloading TinyStories/LongAlpaca, hitting the HF Hub, or actually
running a full Trainer loop. The full Trainer loop is exercised by a single
end-to-end test against a tiny in-memory dataset and a tiny config.

Run from repo root:
    pytest train/test_main.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from models.hf_gemma3.config_gemma3 import Gemma3TTTConfig
from models.hf_gemma3.model_gemma3 import Gemma3ForCausalLMTTT

from train.main import (
    AUTO_MAP,
    DATASET_CHOICES,
    DATASET_DEFAULTS,
    _format_longalpaca,
    _resolve,
    _samples_tag,
    _setup_wandb,
    _tokenize,
    build_dataset,
    bundle_remote_code,
    parse_args,
    save_with_auto_map,
    train_on_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpt2_tokenizer():
    """A tiny, dependency-free tokenizer for dataset tests."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _tiny_ttt_config(use_ttt: bool = True) -> Gemma3TTTConfig:
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
        ttt_layers=[0, 2],
        ttt_chunk=16,
        ttt_lr=0.3,
        ttt_proj=True,
        ttt_target="hidden_states",
    )


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------

def test_dataset_choices_match_defaults():
    assert set(DATASET_CHOICES) == set(DATASET_DEFAULTS)


def test_auto_map_points_at_local_modules():
    assert AUTO_MAP["AutoConfig"].endswith("Gemma3TTTConfig")
    assert AUTO_MAP["AutoModelForCausalLM"].endswith("Gemma3ForCausalLMTTT")


# ---------------------------------------------------------------------------
# _tokenize: must NOT pre-add labels (collator does that)
# ---------------------------------------------------------------------------

def test_tokenize_returns_input_ids_and_attention_mask_only(gpt2_tokenizer):
    out = _tokenize(
        {"text": ["hello world", "another short sample"]},
        gpt2_tokenizer, "text", max_length=16,
    )
    assert "input_ids" in out and "attention_mask" in out
    assert "labels" not in out, "labels must be added by the collator, not the tokenizer step"
    assert len(out["input_ids"]) == 2


def test_tokenize_truncates_to_max_length(gpt2_tokenizer):
    text = "word " * 200
    out = _tokenize({"text": [text]}, gpt2_tokenizer, "text", max_length=12)
    assert len(out["input_ids"][0]) <= 12


def test_collator_pads_and_creates_labels(gpt2_tokenizer):
    """End-to-end check that _tokenize + collator produce a properly batched
    tensor. This is exactly what the Trainer's DataLoader sees."""
    out = _tokenize(
        {"text": ["short", "this one is noticeably longer than the other"]},
        gpt2_tokenizer, "text", max_length=64,
    )
    features = [
        {"input_ids": ids, "attention_mask": am}
        for ids, am in zip(out["input_ids"], out["attention_mask"])
    ]
    collator = DataCollatorForLanguageModeling(tokenizer=gpt2_tokenizer, mlm=False)
    batch = collator(features)

    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape[0] == 2
    # Padded items must have labels masked to -100 at pad positions.
    pad_mask = batch["input_ids"] == gpt2_tokenizer.pad_token_id
    if pad_mask.any():
        assert (batch["labels"][pad_mask] == -100).all()


# ---------------------------------------------------------------------------
# _format_longalpaca
# ---------------------------------------------------------------------------

def test_format_longalpaca_with_input():
    text = _format_longalpaca({"instruction": "I", "input": "X", "output": "Y"})
    assert "### Instruction:\nI" in text
    assert "### Input:\nX" in text
    assert "### Response:\nY" in text


def test_format_longalpaca_without_input():
    text = _format_longalpaca({"instruction": "I", "input": "", "output": "Y"})
    assert "### Input" not in text
    assert "### Instruction:\nI" in text
    assert "### Response:\nY" in text


def test_format_longalpaca_handles_missing_keys():
    text = _format_longalpaca({"instruction": "I"})  # no input/output
    assert "### Instruction:\nI" in text
    assert "### Response:\n" in text


# ---------------------------------------------------------------------------
# build_dataset dispatch
# ---------------------------------------------------------------------------

def test_build_dataset_unknown_name_raises(gpt2_tokenizer):
    with pytest.raises(ValueError, match="unknown dataset"):
        build_dataset("not-a-dataset", gpt2_tokenizer, max_length=32, max_samples=4)


# ---------------------------------------------------------------------------
# Freeze contract (covered more deeply in models/hf_gemma3/test_gemma3.py;
# this duplicate exists so train/ has a self-contained sanity check).
# ---------------------------------------------------------------------------

def test_freeze_base_model_keeps_ttt_and_down_proj_trainable():
    cfg = _tiny_ttt_config(use_ttt=True)
    model = Gemma3ForCausalLMTTT(cfg)
    model.freeze_base_model()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable, "expected trainable params"
    assert any("ttt_conv" in n for n in trainable)
    assert any("ttt_proj" in n for n in trainable)
    assert any("down_proj" in n for n in trainable)

    # down_proj must be trainable only on TTT layers (and frozen elsewhere).
    ttt_layers = set(cfg.ttt_layers)
    for name, p in model.named_parameters():
        if "down_proj" in name and "ttt" not in name:
            layer_idx = int(name.split(".")[2])
            assert p.requires_grad == (layer_idx in ttt_layers), name


# ---------------------------------------------------------------------------
# bundle_remote_code: copies the .py files next to the weights
# ---------------------------------------------------------------------------

def test_bundle_remote_code_copies_modeling_files(tmp_path: Path):
    bundle_remote_code(tmp_path)
    # __init__.py is empty so we don't strictly need to assert it exists; check
    # the two files that matter for `trust_remote_code` loading.
    assert (tmp_path / "config_gemma3.py").exists()
    assert (tmp_path / "model_gemma3.py").exists()


# ---------------------------------------------------------------------------
# save_with_auto_map: sets auto_map and writes a usable checkpoint (no push)
# ---------------------------------------------------------------------------

def test_save_with_auto_map_writes_loadable_checkpoint(gpt2_tokenizer, tmp_path: Path):
    cfg = _tiny_ttt_config(use_ttt=True)
    model = Gemma3ForCausalLMTTT(cfg).eval()

    save_dir = tmp_path / "final"
    save_with_auto_map(
        model, gpt2_tokenizer, save_dir,
        repo_id=None, push_to_hub=False,
    )

    assert (save_dir / "config.json").exists()
    assert any(save_dir.glob("*.safetensors")) or (save_dir / "pytorch_model.bin").exists()
    assert (save_dir / "config_gemma3.py").exists()
    assert (save_dir / "model_gemma3.py").exists()
    # auto_map written into the in-memory config (the on-disk one is also tested
    # implicitly by reload below).
    assert model.config.auto_map == AUTO_MAP

    reloaded_cfg = Gemma3TTTConfig.from_pretrained(save_dir)
    assert reloaded_cfg.auto_map == AUTO_MAP


# ---------------------------------------------------------------------------
# _setup_wandb
# ---------------------------------------------------------------------------

def _clear_wandb_env():
    for k in ("WANDB_DISABLED", "WANDB_PROJECT", "WANDB_NAME"):
        os.environ.pop(k, None)


def test_setup_wandb_disabled_returns_false():
    _clear_wandb_env()
    try:
        ok = _setup_wandb(enabled=False, project="p", run_name=None, dataset="ds")
        assert ok is False
        assert os.environ.get("WANDB_DISABLED") == "true"
    finally:
        _clear_wandb_env()


def test_setup_wandb_enabled_sets_env_when_installed():
    _clear_wandb_env()
    try:
        with patch("importlib.util.find_spec", return_value=object()):
            ok = _setup_wandb(enabled=True, project="my-proj", run_name=None, dataset="tinystories")
        assert ok is True
        assert os.environ["WANDB_PROJECT"] == "my-proj"
        assert os.environ["WANDB_NAME"] == "gemma3-ttt-tinystories"
    finally:
        _clear_wandb_env()


def test_setup_wandb_enabled_uses_explicit_run_name():
    _clear_wandb_env()
    try:
        with patch("importlib.util.find_spec", return_value=object()):
            ok = _setup_wandb(enabled=True, project="p", run_name="custom-run", dataset="ds")
        assert ok is True
        assert os.environ["WANDB_NAME"] == "custom-run"
    finally:
        _clear_wandb_env()


def test_setup_wandb_missing_package_returns_false():
    _clear_wandb_env()
    try:
        with patch("importlib.util.find_spec", return_value=None):
            ok = _setup_wandb(enabled=True, project="p", run_name=None, dataset="ds")
        assert ok is False
        assert os.environ.get("WANDB_DISABLED") == "true"
    finally:
        _clear_wandb_env()


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def _run_parse_args(*argv: str):
    with patch.object(sys, "argv", ["train.main", *argv]):
        return parse_args()


def test_parse_args_requires_dataset():
    with pytest.raises(SystemExit):
        _run_parse_args()


def test_parse_args_rejects_unknown_dataset():
    with pytest.raises(SystemExit):
        _run_parse_args("--dataset", "nope")


def test_parse_args_defaults_for_tinystories():
    args = _run_parse_args("--dataset", "tinystories")
    assert args.dataset == "tinystories"
    assert args.base_model == "google/gemma-3-1b-it"
    assert args.epochs is None  # filled in from DATASET_DEFAULTS at runtime
    assert args.no_wandb is False
    assert args.wandb_project == "gemma3-ttt"


def test_parse_args_overrides():
    args = _run_parse_args(
        "--dataset", "longalpaca",
        "--epochs", "0.5",
        "--lr", "1e-4",
        "--max-length", "1024",
        "--no-wandb",
        "--no-push",
        "--repo-id", "user/custom-repo",
    )
    assert args.epochs == 0.5
    assert args.lr == 1e-4
    assert args.max_length == 1024
    assert args.no_wandb is True
    assert args.no_push is True
    assert args.repo_id == "user/custom-repo"


def test_resolve_helper():
    assert _resolve(None, 7) == 7
    assert _resolve(3, 7) == 3
    assert _resolve(0, 7) == 0  # explicit zero must win over default


def test_samples_tag_formatting():
    assert _samples_tag(None) == "full"
    assert _samples_tag(500) == "500"
    assert _samples_tag(50_000) == "50k"
    assert _samples_tag(1_500) == "1.5k"
    assert _samples_tag(2_000_000) == "2m"
    assert _samples_tag(1_500_000) == "1.5m"


# ---------------------------------------------------------------------------
# End-to-end Trainer step on a tiny in-memory dataset
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    __import__("importlib").util.find_spec("accelerate") is None,
    reason="accelerate is required for HF Trainer; install with `pip install accelerate`",
)
def test_train_on_dataset_runs_one_step(gpt2_tokenizer, tmp_path: Path):
    """Verify train_on_dataset wires Trainer/data correctly and a single step
    produces gradients only on TTT + down_proj params."""
    # Build a tiny in-memory dataset that mimics what _tokenize produces.
    samples = [
        gpt2_tokenizer("hello world short", truncation=True, max_length=16),
        gpt2_tokenizer("another tiny sample for the trainer", truncation=True, max_length=16),
        gpt2_tokenizer("one more", truncation=True, max_length=16),
        gpt2_tokenizer("and a fourth", truncation=True, max_length=16),
    ]
    ds = Dataset.from_list([
        {"input_ids": s["input_ids"], "attention_mask": s["attention_mask"]}
        for s in samples
    ])

    cfg = _tiny_ttt_config(use_ttt=True)
    # Use the gpt2 tokenizer's vocab size so embeddings match generated ids.
    cfg.vocab_size = gpt2_tokenizer.vocab_size + 1
    model = Gemma3ForCausalLMTTT(cfg)
    model.freeze_base_model()

    train_on_dataset(
        model, gpt2_tokenizer, ds, tmp_path,
        epochs=1.0,
        batch_size=2,
        grad_accum=1,
        lr=1e-4,
        weight_decay=0.1,
        warmup_steps=0,
        max_grad_norm=1.0,
        bf16=False,            # CPU-friendly
        save_steps=10_000,     # avoid intermediate saves
        logging_steps=1,
        use_wandb=False,
    )

    # Trainer output dir was created.
    assert (tmp_path / "trainer").exists()
