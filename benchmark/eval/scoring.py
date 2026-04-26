"""Normalized exact-match scoring."""

from __future__ import annotations

from typing import Dict

from ..data_gen import normalize_answer


def score_example(
    example: Dict,
    prediction: str,
    lowercase: bool = True,
    strip_whitespace: bool = True,
    strip_punctuation: bool = True,
) -> bool:
    """Returns True if `prediction` normalized-equals the answer or any alias.

    The prediction is matched against aliases via substring containment after
    normalization; this tolerates models that say "48291." or "The code is 48291"
    while still failing models that emit the wrong value.
    """
    norm_pred = normalize_answer(
        prediction,
        lowercase=lowercase,
        strip_whitespace=strip_whitespace,
        strip_punctuation=strip_punctuation,
    )
    candidates = [example["answer"]] + list(example.get("answer_aliases", []))
    for cand in candidates:
        norm_cand = normalize_answer(
            cand,
            lowercase=lowercase,
            strip_whitespace=strip_whitespace,
            strip_punctuation=strip_punctuation,
        )
        if not norm_cand:
            continue
        if norm_cand == norm_pred or norm_cand in norm_pred:
            return True
    return False
