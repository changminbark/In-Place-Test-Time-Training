"""Generate HELMET-style benchmark rows in our schema.

This is a parallel runner to `ruler_runner.py`. It supports two HELMET
categories — ICL (many-shot classification) and RAG (long-context QA over
retrieved documents).

ICL is self-contained: the underlying HuggingFace datasets (CogComp/trec,
PolyAI/banking77, etc.) are pulled at runtime; no external data is needed.
The N-shot demonstration pack and the test query are assembled here, with
a balanced label sampler that mirrors HELMET's `balance_labels`.

RAG requires a one-time download of HELMET's pre-processed KILT JSONLs
(see `benchmark/README.md` for the command). The path is resolved via the
HELMET_DATA_DIR env var, defaulting to `third_party/HELMET/data`.

Output schema matches `_to_our_schema` in ruler_runner.py: every row carries
`id`, `task`, `context_length_target`, `document`, `question`, `answer`,
`answer_aliases`, `metadata`, plus an extra `prompt` field that contains the
fully-rendered HELMET prompt (the predictor uses this verbatim instead of
its default doc+question template).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Task → context-length → shot-count mapping for ICL.
#
# Lifted from HELMET's `configs/icl_short.yaml`. These shot counts are
# pre-tuned so that the rendered prompt at the target context length sits
# just under the cap when tokenized with Llama-3 (HELMET's reference). With
# Gemma-3's tokenizer the actual count drifts ~5-10% — close enough for the
# bucket label.
# ---------------------------------------------------------------------------

_ICL_SHOTS_BY_LEN: Dict[str, Dict[int, int]] = {
    "helmet_trec_coarse": {8192: 400, 16384: 800, 32768: 1600},
    "helmet_trec_fine":   {8192: 400, 16384: 800, 32768: 1600},
    "helmet_banking77":   {8192: 360, 16384: 720, 32768: 1450},
    "helmet_clinic150":   {8192: 440, 16384: 880, 32768: 1750},
    "helmet_nlu":         {8192: 510, 16384: 1020, 32768: 2040},
}


_ICL_USER_TEMPLATE = (
    'Use the provided mapping from the text to label to assign a label to '
    'the text. Only output "label: {{label}}" and nothing else. \n\n'
    "{context}\n\n{question}"
)
_ICL_SYSTEM_TEMPLATE = "label:"
_ICL_PROMPT_TEMPLATE = _ICL_USER_TEMPLATE + "\n" + _ICL_SYSTEM_TEMPLATE
_ICL_ITEM_TEMPLATE = "{text}\nlabel: {label}"


def _load_icl_dataset(task: str):
    """Returns (train_dataset, test_dataset, id2label, text_field, label_field).

    Uses the parquet revision of each HF dataset. Newer datasets versions
    (>=4.0) reject legacy loading scripts; the auto-converted parquet branch
    (`refs/convert/parquet`) is the supported path and preserves ClassLabel
    metadata, which we need for `.names`.
    """
    from datasets import load_dataset

    PARQUET = {"revision": "refs/convert/parquet"}

    if task == "helmet_trec_coarse":
        ds = load_dataset("CogComp/trec", **PARQUET)
        return ds["train"], ds["test"], ds["train"].features["coarse_label"].names, "text", "coarse_label"
    if task == "helmet_trec_fine":
        ds = load_dataset("CogComp/trec", **PARQUET)
        return ds["train"], ds["test"], ds["train"].features["fine_label"].names, "text", "fine_label"
    if task == "helmet_banking77":
        ds = load_dataset("PolyAI/banking77", **PARQUET)
        return ds["train"], ds["test"], ds["train"].features["label"].names, "text", "label"
    if task == "helmet_clinic150":
        ds = load_dataset("clinc/clinc_oos", **PARQUET)
        return ds["train"], ds["validation"], ds["train"].features["intent"].names, "text", "intent"
    if task == "helmet_nlu":
        full = load_dataset("xingkunliuxtracta/nlu_evaluation_data", **PARQUET)["train"]
        split = full.train_test_split(test_size=0.1, seed=42)
        return split["train"], split["test"], full.features["label"].names, "text", "label"
    raise KeyError(f"unknown HELMET ICL task {task!r}")


def _balanced_demo_pack(train, label_field: str, shots: int, seed: int) -> List[dict]:
    """Pick `shots` demonstrations with a balanced label distribution."""
    rng = random.Random(seed)
    by_label: Dict[int, List[dict]] = {}
    for ex in train:
        by_label.setdefault(ex[label_field], []).append(ex)
    n_labels = len(by_label)
    rounds = math.ceil(shots / n_labels)
    rounds_lists: List[List[dict]] = [[] for _ in range(rounds)]
    for label_examples in by_label.values():
        # Sample `rounds` indices without replacement (with replacement only if
        # rounds > available, which would be rare for these datasets).
        if rounds <= len(label_examples):
            indices = rng.sample(range(len(label_examples)), rounds)
        else:
            indices = []
            while len(indices) < rounds:
                indices.extend(rng.sample(range(len(label_examples)),
                                          min(rounds - len(indices), len(label_examples))))
        for i, idx in enumerate(indices):
            rounds_lists[i].append(label_examples[idx])
    for r in rounds_lists:
        rng.shuffle(r)
    flat = [item for r in rounds_lists for item in r]
    return flat[:shots]


def _icl_examples(
    task: str,
    target_tokens: int,
    num_samples: int,
    seed: int,
) -> Iterator[dict]:
    """Yield ICL rows in our schema."""
    if task not in _ICL_SHOTS_BY_LEN:
        raise KeyError(f"{task} has no shot-count mapping in helmet_runner")
    if target_tokens not in _ICL_SHOTS_BY_LEN[task]:
        raise KeyError(
            f"{task} not configured for {target_tokens} tokens; supported: "
            f"{sorted(_ICL_SHOTS_BY_LEN[task].keys())}"
        )
    shots = _ICL_SHOTS_BY_LEN[task][target_tokens]

    train, test, id2label, text_field, label_field = _load_icl_dataset(task)

    # Balance the test split too — sample `num_samples` with even label spread,
    # mirroring HELMET's behavior in load_icl.
    test_pool = list(test)
    test_picks = _balanced_demo_pack(test_pool, label_field, num_samples, seed)

    n_labels = len(id2label)

    for idx, sample in enumerate(test_picks):
        # HELMET uses a per-sample seed derived from the test text — we do the
        # same so that the demo pack is deterministic and example-specific.
        local_seed = (int(hashlib.sha256(sample[text_field].encode("utf-8")).hexdigest(), 16) + seed) % 2**31
        demos = _balanced_demo_pack(train, label_field, shots, local_seed)

        # HELMET shuffles the integer label mapping per example to defeat
        # base-rate shortcuts. We do the same.
        label_mapping = list(range(n_labels))
        random.Random(local_seed).shuffle(label_mapping)

        context = "\n\n".join(
            _ICL_ITEM_TEMPLATE.format(
                text=d[text_field],
                label=str(label_mapping[int(d[label_field])]),
            )
            for d in demos
        )
        question = sample[text_field]
        answer = str(label_mapping[int(sample[label_field])])

        prompt = _ICL_PROMPT_TEMPLATE.format(context=context, question=question)

        yield {
            "id": f"{task}_{target_tokens}_{idx:04d}",
            "task": task,
            "context_length_target": target_tokens,
            "document": context,
            "question": question + "\n" + _ICL_SYSTEM_TEMPLATE,
            "answer": answer,
            "answer_aliases": [answer],
            "prompt": prompt,
            "metadata": {
                "task_variant": task,
                "shots": shots,
                "n_labels": n_labels,
                "label_mapping": label_mapping,
                "true_label_index": int(sample[label_field]),
                "true_label_name": id2label[int(sample[label_field])],
            },
        }


# ---------------------------------------------------------------------------
# RAG
#
# HELMET's RAG section consumes pre-processed KILT JSONLs that bundle a
# question with a set of retrieved Wikipedia passages (`ctxs`). Each ctx has
# `title`, `text`, and `has_answer`. The number of retrieved passages (`k`)
# controls the prompt length; HELMET's `configs/rag_short.yaml` maps:
#   k50  ≈  8k tokens
#   k105 ≈ 16k
#   k220 ≈ 32k
#   k440 ≈ 64k  (skipped — out of Gemma-3-1b's 32k window)
#
# These JSONLs ship in HELMET's `data.tar.gz` from HF dataset
# `princeton-nlp/HELMET`. Users run scripts/download_data.sh once to populate
# `third_party/HELMET/data/kilt/`. We resolve the path via HELMET_DATA_DIR
# (default: `third_party/HELMET/data`).
# ---------------------------------------------------------------------------

_RAG_K_BY_LEN: Dict[int, int] = {8192: 50, 16384: 105, 32768: 220}

# (task, k) -> filename inside data/kilt/
_RAG_FILE_TEMPLATES: Dict[str, str] = {
    "helmet_nq":         "kilt/nq-dev-multikilt_1000_k{k}_dep6.jsonl",
    "helmet_hotpotqa":   "kilt/hotpotqa-dev-multikilt_1000_k{k}_dep3.jsonl",
    "helmet_triviaqa":   "kilt/triviaqa-dev-multikilt_1000_k{k}_dep6.jsonl",
    "helmet_popqa":      "kilt/popqa_test_1000_k{k}_dep6.jsonl",
}

# Few-shot demo pack (k=3 retrieved docs each). HELMET defaults to 2 shots.
_RAG_DEMO_TEMPLATES: Dict[str, str] = {
    "helmet_nq":         "kilt/nq-train-multikilt_1000_k3_dep6.jsonl",
    "helmet_hotpotqa":   "kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl",
    "helmet_triviaqa":   "kilt/triviaqa-train-multikilt_1000_k3_dep6.jsonl",
    "helmet_popqa":      "kilt/popqa_test_1000_k3_dep6.jsonl",
}

_RAG_USER_TEMPLATE = (
    "Use the given documents to write a concise and short answer to the "
    "question. Write your answer in the following format:\nAnswer: [answer]\n\n"
    "{demos}{context}\n\nQuestion: {question}"
)
_RAG_SYSTEM_TEMPLATE = "Answer:"
_RAG_PROMPT_TEMPLATE = _RAG_USER_TEMPLATE + "\n" + _RAG_SYSTEM_TEMPLATE
_RAG_PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
_RAG_DEMO_TEMPLATE = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"


def _resolve_helmet_data_dir() -> Path:
    p = os.environ.get("HELMET_DATA_DIR")
    if p:
        return Path(p)
    return Path(__file__).resolve().parents[2] / "third_party" / "HELMET" / "data"


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _format_passages(ctxs: List[dict]) -> str:
    return "\n\n".join(_RAG_PASSAGE_TEMPLATE.format(**c) for c in ctxs)


def _rag_examples(
    task: str,
    target_tokens: int,
    num_samples: int,
    seed: int,
    shots: int = 2,
) -> Iterator[dict]:
    if task not in _RAG_FILE_TEMPLATES:
        raise KeyError(f"{task} has no RAG file template")
    if target_tokens not in _RAG_K_BY_LEN:
        raise KeyError(
            f"{task} not configured for {target_tokens} tokens; supported: "
            f"{sorted(_RAG_K_BY_LEN.keys())}"
        )

    k = _RAG_K_BY_LEN[target_tokens]
    data_dir = _resolve_helmet_data_dir()
    test_path = data_dir / _RAG_FILE_TEMPLATES[task].format(k=k)
    demo_path = data_dir / _RAG_DEMO_TEMPLATES[task]
    if not test_path.exists():
        raise FileNotFoundError(
            f"HELMET RAG data missing: {test_path}\n"
            "Run `bash third_party/HELMET/scripts/download_data.sh` once "
            "(downloads ~4 GB to that directory) and set HELMET_DATA_DIR if "
            "you put it elsewhere."
        )
    if not demo_path.exists():
        raise FileNotFoundError(f"HELMET RAG demo file missing: {demo_path}")

    test_rows = _read_jsonl(test_path)
    demo_rows = _read_jsonl(demo_path)

    rng = random.Random(seed)
    if num_samples and len(test_rows) > num_samples:
        # HELMET picks by unique id/question; we just sample.
        rng.shuffle(test_rows)
        test_rows = test_rows[:num_samples]

    for idx, sample in enumerate(test_rows):
        # Per-sample seed for stable few-shot demo selection (mirrors HELMET).
        local_seed = (
            int(hashlib.sha256(str(sample.get("id") or sample.get("question", ""))
                               .encode("utf-8")).hexdigest(), 16) + seed
        ) % 2**31
        local_rng = random.Random(local_seed)

        if shots > 0:
            picks = local_rng.sample(demo_rows, min(shots, len(demo_rows)))
            demo_text = "\n\n".join(
                _RAG_DEMO_TEMPLATE.format(
                    documents=_format_passages(d.get("ctxs", [])),
                    question=d["question"],
                    answer=(d["answers"][0] if isinstance(d.get("answers"), list) else d.get("answer", "")),
                )
                for d in picks
            ) + "\n\n"
        else:
            demo_text = ""

        passages = _format_passages(sample.get("ctxs", []))
        prompt = _RAG_PROMPT_TEMPLATE.format(
            demos=demo_text, context=passages, question=sample["question"]
        )
        answers = sample.get("answers") or [sample.get("answer", "")]
        if isinstance(answers, str):
            answers = [answers]

        yield {
            "id": f"{task}_{target_tokens}_{idx:04d}",
            "task": task,
            "context_length_target": target_tokens,
            "document": passages,
            "question": "Question: " + sample["question"] + "\n" + _RAG_SYSTEM_TEMPLATE,
            "answer": answers[0],
            "answer_aliases": answers,
            "prompt": prompt,
            "metadata": {
                "task_variant": task,
                "k": k,
                "shots": shots,
                "kilt_id": sample.get("id"),
            },
        }


# ---------------------------------------------------------------------------
# Public entrypoint (matches ruler_runner.generate_examples signature)
# ---------------------------------------------------------------------------

_ICL_TASKS = set(_ICL_SHOTS_BY_LEN.keys())
_RAG_TASKS = {"helmet_nq", "helmet_hotpotqa", "helmet_triviaqa", "helmet_popqa"}
HELMET_TASKS = _ICL_TASKS | _RAG_TASKS


def generate_examples(
    task: str,
    target_tokens: int,
    num_samples: int,
    tokenizer_model_id: str,  # accepted for signature parity; ignored here
    seed: int,
) -> Iterator[dict]:
    if task in _ICL_TASKS:
        yield from _icl_examples(task, target_tokens, num_samples, seed)
        return
    if task in _RAG_TASKS:
        yield from _rag_examples(task, target_tokens, num_samples, seed)
        return
    raise KeyError(f"unknown HELMET task {task!r}; known: {sorted(HELMET_TASKS)}")
