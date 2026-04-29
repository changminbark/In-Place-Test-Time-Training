"""Normalized exact-match + token-F1 scoring.

`score_example` (substring-EM) is the primary correctness signal used by the
benchmark runner — it's the headline metric in HELMET and matches what RULER
expects.

`f1_metrics` adds HELMET's token-F1 (and exact-match, substring-EM) for
HELMET-comparable reporting. Available for callers but not used by the runner
by default.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Tuple

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


# ---------------------------------------------------------------------------
# HELMET-style metrics (vendored from third_party/HELMET/utils.py to avoid a
# runtime dep on HELMET's package). Used for richer per-row metrics on RAG /
# ICL tasks where headline numbers are reported as substring-EM / F1.
# ---------------------------------------------------------------------------

_PUNCT = set(string.punctuation)


def _helmet_normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in _PUNCT)
    return " ".join(s.split())


def _f1(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    p = _helmet_normalize(prediction)
    g = _helmet_normalize(ground_truth)
    if p in {"yes", "no", "noanswer"} and p != g:
        return 0.0, 0.0, 0.0
    if g in {"yes", "no", "noanswer"} and p != g:
        return 0.0, 0.0, 0.0
    p_toks = p.split()
    g_toks = g.split()
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0, 0.0, 0.0
    precision = n_same / max(1, len(p_toks))
    recall = n_same / max(1, len(g_toks))
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def _max_over(metric_fn, prediction: str, ground_truths: Iterable) -> float:
    """ground_truths can be str | List[str] | List[List[str]]."""
    if isinstance(ground_truths, str):
        gts: List[str] = [ground_truths]
    else:
        gts = []
        for g in ground_truths:
            if isinstance(g, list):
                gts.extend(g)
            else:
                gts.append(g)
    if not gts:
        return 0.0
    return max(metric_fn(prediction, g) for g in gts)


def f1_metrics(prediction: str, ground_truths) -> Dict[str, float]:
    """Returns HELMET-style {exact_match, substring_em, f1} as floats in [0, 1]."""
    em = _max_over(lambda p, g: float(_helmet_normalize(p) == _helmet_normalize(g)),
                   prediction, ground_truths)
    sub_em = _max_over(lambda p, g: float(_helmet_normalize(g) in _helmet_normalize(p)),
                       prediction, ground_truths)
    f1 = _max_over(lambda p, g: _f1(p, g)[0], prediction, ground_truths)
    return {"exact_match": em, "substring_em": sub_em, "f1": f1}
