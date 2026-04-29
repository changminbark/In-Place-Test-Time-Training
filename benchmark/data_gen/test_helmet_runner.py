"""Unit tests for benchmark/data_gen/helmet_runner.py.

Covers:
  - _balanced_demo_pack: shot count, label balance, determinism by seed
  - _icl_examples: deterministic output, label remapping is internally
    consistent (the gold answer matches the remapped label for the
    test sample), prompt structure
  - _resolve_helmet_data_dir: env var precedence vs default
  - _rag_examples: clean FileNotFoundError when KILT data is absent
  - public generate_examples dispatch + HELMET_TASKS membership

We avoid touching the HF datasets cache here — `_load_icl_dataset` is
monkeypatched to return small synthetic in-memory datasets so tests are
hermetic and fast.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import List

import pytest

# Ensure project root is on sys.path so `benchmark.*` imports work when running
# pytest from the repo root via `uv run pytest`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.data_gen import helmet_runner as hr
from benchmark.data_gen import HELMET_TASKS, generate_examples


# ---------------------------------------------------------------------------
# Synthetic mini-dataset for ICL monkeypatching
# ---------------------------------------------------------------------------

def _fake_dataset(num_per_label: int, n_labels: int, label_names: List[str]):
    """Return list of dicts mimicking HF Dataset rows."""
    out = []
    for label_idx in range(n_labels):
        for j in range(num_per_label):
            out.append({"text": f"label{label_idx}_example{j}", "label": label_idx})
    return out


@pytest.fixture
def fake_icl_loader(monkeypatch):
    """Patch _load_icl_dataset to return a tiny deterministic dataset."""
    label_names = [f"cls_{i}" for i in range(5)]
    train = _fake_dataset(num_per_label=20, n_labels=5, label_names=label_names)
    test = _fake_dataset(num_per_label=4, n_labels=5, label_names=label_names)

    def _patched(task):
        return train, test, label_names, "text", "label"

    monkeypatch.setattr(hr, "_load_icl_dataset", _patched)
    # Also pin a shot count for the synthetic task we'll call.
    monkeypatch.setitem(
        hr._ICL_SHOTS_BY_LEN,
        "helmet_test_synthetic",
        {8192: 20, 16384: 40, 32768: 60},
    )
    # Register the synthetic task in the public dispatch sets so
    # generate_examples() routes it to the helmet runner.
    from benchmark import data_gen as dg
    monkeypatch.setattr(hr, "_ICL_TASKS", hr._ICL_TASKS | {"helmet_test_synthetic"})
    monkeypatch.setattr(hr, "HELMET_TASKS", hr.HELMET_TASKS | {"helmet_test_synthetic"})
    monkeypatch.setattr(dg, "HELMET_TASKS", dg.HELMET_TASKS | {"helmet_test_synthetic"})
    return train, test, label_names


# ---------------------------------------------------------------------------
# _balanced_demo_pack
# ---------------------------------------------------------------------------

class TestBalancedDemoPack:
    def test_shot_count_exact(self):
        data = _fake_dataset(num_per_label=10, n_labels=4, label_names=["a", "b", "c", "d"])
        picks = hr._balanced_demo_pack(data, label_field="label", shots=12, seed=1)
        assert len(picks) == 12

    def test_label_distribution_balanced(self):
        data = _fake_dataset(num_per_label=10, n_labels=4, label_names=["a", "b", "c", "d"])
        picks = hr._balanced_demo_pack(data, label_field="label", shots=12, seed=1)
        counts = Counter(p["label"] for p in picks)
        # 12 shots / 4 labels = 3 each (perfect balance for a multiple)
        assert all(counts[i] == 3 for i in range(4)), counts

    def test_label_distribution_balanced_uneven(self):
        data = _fake_dataset(num_per_label=10, n_labels=4, label_names=["a", "b", "c", "d"])
        picks = hr._balanced_demo_pack(data, label_field="label", shots=10, seed=1)
        counts = Counter(p["label"] for p in picks)
        # Each label gets at least floor(10/4)=2; difference between max/min ≤ 1
        assert max(counts.values()) - min(counts.values()) <= 1, counts

    def test_deterministic_by_seed(self):
        data = _fake_dataset(num_per_label=10, n_labels=4, label_names=["a", "b", "c", "d"])
        a = hr._balanced_demo_pack(data, label_field="label", shots=12, seed=42)
        b = hr._balanced_demo_pack(data, label_field="label", shots=12, seed=42)
        assert [(p["text"], p["label"]) for p in a] == [(p["text"], p["label"]) for p in b]

    def test_different_seeds_differ(self):
        data = _fake_dataset(num_per_label=10, n_labels=4, label_names=["a", "b", "c", "d"])
        a = hr._balanced_demo_pack(data, label_field="label", shots=12, seed=1)
        b = hr._balanced_demo_pack(data, label_field="label", shots=12, seed=2)
        assert [(p["text"], p["label"]) for p in a] != [(p["text"], p["label"]) for p in b]


# ---------------------------------------------------------------------------
# _icl_examples
# ---------------------------------------------------------------------------

class TestIclExamples:
    def test_emits_requested_count(self, fake_icl_loader):
        rows = list(hr._icl_examples(
            task="helmet_test_synthetic",
            target_tokens=8192,
            num_samples=3,
            seed=7,
        ))
        assert len(rows) == 3

    def test_schema_required_keys(self, fake_icl_loader):
        rows = list(hr._icl_examples(
            task="helmet_test_synthetic",
            target_tokens=8192,
            num_samples=2,
            seed=7,
        ))
        required = {"id", "task", "context_length_target", "document",
                    "question", "answer", "answer_aliases", "prompt", "metadata"}
        for r in rows:
            missing = required - r.keys()
            assert not missing, f"row missing keys: {missing}"
            assert r["task"] == "helmet_test_synthetic"
            assert r["context_length_target"] == 8192

    def test_answer_matches_remapped_true_label(self, fake_icl_loader):
        """For each emitted row, the gold answer must equal the per-example
        remapped index of the test sample's true label."""
        rows = list(hr._icl_examples(
            task="helmet_test_synthetic",
            target_tokens=8192,
            num_samples=5,
            seed=11,
        ))
        for r in rows:
            md = r["metadata"]
            mapped = md["label_mapping"][md["true_label_index"]]
            assert r["answer"] == str(mapped), (r["answer"], mapped)

    def test_label_mapping_is_a_permutation(self, fake_icl_loader):
        rows = list(hr._icl_examples(
            task="helmet_test_synthetic",
            target_tokens=8192,
            num_samples=4,
            seed=3,
        ))
        for r in rows:
            mapping = r["metadata"]["label_mapping"]
            n = r["metadata"]["n_labels"]
            assert sorted(mapping) == list(range(n))

    def test_prompt_ends_with_label_cue(self, fake_icl_loader):
        rows = list(hr._icl_examples(
            task="helmet_test_synthetic",
            target_tokens=8192,
            num_samples=1,
            seed=3,
        ))
        assert rows[0]["prompt"].rstrip().endswith("label:")

    def test_prompt_contains_test_query(self, fake_icl_loader):
        rows = list(hr._icl_examples(
            task="helmet_test_synthetic",
            target_tokens=8192,
            num_samples=1,
            seed=3,
        ))
        # The test sample text should appear in the prompt (it's the {question})
        # The first 4 chars are deterministic — we can't know which label
        # was sampled, so just check that *some* "labelN_example" string is there.
        assert "label" in rows[0]["prompt"] and "_example" in rows[0]["prompt"]

    def test_deterministic_by_seed(self, fake_icl_loader):
        a = list(hr._icl_examples(task="helmet_test_synthetic", target_tokens=8192, num_samples=3, seed=99))
        b = list(hr._icl_examples(task="helmet_test_synthetic", target_tokens=8192, num_samples=3, seed=99))
        assert [r["prompt"] for r in a] == [r["prompt"] for r in b]
        assert [r["answer"] for r in a] == [r["answer"] for r in b]

    def test_unsupported_length_raises(self, fake_icl_loader):
        with pytest.raises(KeyError, match="not configured for"):
            list(hr._icl_examples(
                task="helmet_test_synthetic",
                target_tokens=1024,  # not in the configured 8k/16k/32k
                num_samples=1,
                seed=1,
            ))

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError, match="has no shot-count mapping"):
            list(hr._icl_examples(
                task="helmet_does_not_exist",
                target_tokens=8192,
                num_samples=1,
                seed=1,
            ))


# ---------------------------------------------------------------------------
# _resolve_helmet_data_dir
# ---------------------------------------------------------------------------

class TestHelmetDataDir:
    def test_default_under_third_party(self, monkeypatch):
        monkeypatch.delenv("HELMET_DATA_DIR", raising=False)
        p = hr._resolve_helmet_data_dir()
        # Should resolve to <repo>/third_party/HELMET/data regardless of cwd.
        assert p.parts[-3:] == ("third_party", "HELMET", "data"), p

    def test_env_override(self, monkeypatch, tmp_path):
        custom = tmp_path / "elsewhere" / "kilt"
        monkeypatch.setenv("HELMET_DATA_DIR", str(custom))
        assert hr._resolve_helmet_data_dir() == custom


# ---------------------------------------------------------------------------
# _rag_examples
# ---------------------------------------------------------------------------

class TestRagExamples:
    def test_missing_data_raises_actionable_error(self, monkeypatch, tmp_path):
        # Point HELMET_DATA_DIR at an empty directory so the file is missing.
        monkeypatch.setenv("HELMET_DATA_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError, match=r"download_data\.sh"):
            list(hr._rag_examples(
                task="helmet_nq",
                target_tokens=8192,
                num_samples=1,
                seed=0,
            ))

    def test_unsupported_length_raises(self):
        with pytest.raises(KeyError, match="not configured for"):
            list(hr._rag_examples(
                task="helmet_nq",
                target_tokens=1024,
                num_samples=1,
                seed=0,
            ))

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError, match="no RAG file template"):
            list(hr._rag_examples(
                task="helmet_does_not_exist",
                target_tokens=8192,
                num_samples=1,
                seed=0,
            ))


# ---------------------------------------------------------------------------
# Public dispatch in benchmark.data_gen.__init__
# ---------------------------------------------------------------------------

class TestPublicDispatch:
    def test_helmet_tasks_membership(self):
        # All ICL families should be present
        assert "helmet_trec_coarse" in HELMET_TASKS
        assert "helmet_banking77" in HELMET_TASKS
        # All RAG families should be present
        assert "helmet_nq" in HELMET_TASKS
        assert "helmet_hotpotqa" in HELMET_TASKS

    def test_dispatch_routes_helmet_to_runner(self, fake_icl_loader):
        rows = list(generate_examples(
            task="helmet_test_synthetic",
            target_tokens=8192,
            num_samples=2,
            tokenizer_model_id="ignored",
            seed=5,
        ))
        assert len(rows) == 2
        assert all(r["task"] == "helmet_test_synthetic" for r in rows)

    def test_dispatch_unknown_helmet_task_raises(self):
        # A name with helmet_ prefix that doesn't match any registered task
        # should land in the helmet runner and raise from there.
        with pytest.raises(KeyError):
            list(generate_examples(
                task="helmet_unregistered_xyz",
                target_tokens=8192,
                num_samples=1,
                tokenizer_model_id="ignored",
                seed=0,
            ))
