"""Shared primitives for benchmark data generation.

Includes a TokenBudgeter that targets a token length using the Gemma tokenizer,
filler-sentence sampling, needle placement helpers, and answer normalization.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass
from typing import Callable, List, Optional


_ENTITY_POOLS = {
    "vault": [f"Vault {name}" for name in [
        "Indigo", "Crimson", "Maroon", "Azure", "Jade", "Onyx", "Amber",
        "Cerulean", "Scarlet", "Violet", "Umber", "Saffron", "Cobalt",
    ]],
    "archive": [f"Archive {name}" for name in [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta",
        "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu",
    ]],
    "counter": [f"Counter {c}" for c in string.ascii_uppercase[:12]],
    "variable": [f"var_{w}" for w in [
        "temp", "load", "score", "count", "level", "depth", "phase",
        "pulse", "gain", "tick", "flux", "index",
    ]],
    "person": [
        "Lena", "Oscar", "Mira", "Dev", "Priya", "Kai", "Noor",
        "Ines", "Raj", "Tomas", "Yui", "Anwar",
    ],
}

_FILLER_TEMPLATES = [
    "The report from {archive} was filed on day {n}.",
    "{person} submitted a note about {vault}.",
    "A reading of {n} was recorded near {archive}.",
    "Maintenance on {vault} is scheduled for week {n}.",
    "Inventory for {archive} reached {n} units.",
    "{person} logged a visit to {vault} at hour {n}.",
    "Sensor {n} was calibrated in {archive}.",
    "The room adjacent to {vault} had temperature {n}.",
    "{person} reviewed the materials stored in {archive}.",
    "Checkpoint {n} passed inspection near {vault}.",
]


@dataclass
class TokenBudgeter:
    """Fills a document to a target token length using the given tokenizer."""

    tokenizer: "object"
    target_tokens: int
    tolerance: int = 64

    def _token_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def fill(
        self,
        sentence_sampler: Callable[[], str],
        required_inserts: Optional[List[tuple]] = None,
    ) -> str:
        """Build a document by sampling filler sentences until the target token budget is met.

        `required_inserts` is an optional list of (normalized_position_in_[0,1], sentence) tuples
        marking where answer-bearing sentences must land. Positions are respected approximately,
        within the filler sequence.
        """
        required_inserts = list(required_inserts or [])
        required_inserts.sort(key=lambda p: p[0])

        # Generate filler up to budget, then splice in required sentences at target positions.
        filler: List[str] = []
        current_tokens = 0
        # Reserve a small budget for the inserts themselves.
        reserved = sum(self._token_len(s) + 1 for _, s in required_inserts)
        target_filler_tokens = max(0, self.target_tokens - reserved)

        while current_tokens < target_filler_tokens - self.tolerance:
            s = sentence_sampler()
            filler.append(s)
            current_tokens += self._token_len(s) + 1  # +1 approx for join space

        # Splice inserts at their target fractional positions in the filler list.
        for frac, sentence in required_inserts:
            idx = max(0, min(len(filler), int(round(frac * len(filler)))))
            filler.insert(idx, sentence)

        doc = " ".join(filler)
        # Trim or pad to target ±tolerance via sentence-level adjustment.
        doc = self._trim_to_budget(doc, sentence_sampler)
        return doc

    def _trim_to_budget(self, doc: str, sentence_sampler: Callable[[], str]) -> str:
        ids = self.tokenizer.encode(doc, add_special_tokens=False)
        if len(ids) > self.target_tokens + self.tolerance:
            # Hard truncate by tokens, then decode back.
            ids = ids[: self.target_tokens]
            doc = self.tokenizer.decode(ids, skip_special_tokens=True)
        elif len(ids) < self.target_tokens - self.tolerance:
            # Append more filler until inside tolerance.
            while len(ids) < self.target_tokens - self.tolerance:
                doc = doc + " " + sentence_sampler()
                ids = self.tokenizer.encode(doc, add_special_tokens=False)
        return doc


def random_number(rng: random.Random, lo: int = 1000, hi: int = 99999) -> int:
    return rng.randint(lo, hi)


def random_entity(rng: random.Random, pool: str) -> str:
    return rng.choice(_ENTITY_POOLS[pool])


def random_person(rng: random.Random) -> str:
    return rng.choice(_ENTITY_POOLS["person"])


def make_filler_sampler(rng: random.Random) -> Callable[[], str]:
    def _sample() -> str:
        template = rng.choice(_FILLER_TEMPLATES)
        return template.format(
            archive=random_entity(rng, "archive"),
            vault=random_entity(rng, "vault"),
            person=random_person(rng),
            n=random_number(rng, 1, 999),
        )
    return _sample


def place_needle(position: str, rng: random.Random) -> float:
    """Map a symbolic position to a fractional placement in [0, 1]."""
    if position == "early":
        return rng.uniform(0.05, 0.2)
    if position == "middle":
        return rng.uniform(0.4, 0.6)
    if position == "late":
        return rng.uniform(0.8, 0.95)
    raise ValueError(f"unknown position: {position}")


_PUNCT_RE = re.compile(r"^[\s" + re.escape(string.punctuation) + r"]+|[\s" + re.escape(string.punctuation) + r"]+$")


def normalize_answer(
    text: str,
    lowercase: bool = True,
    strip_whitespace: bool = True,
    strip_punctuation: bool = True,
) -> str:
    out = text
    if lowercase:
        out = out.lower()
    if strip_whitespace:
        out = out.strip()
    if strip_punctuation:
        out = _PUNCT_RE.sub("", out)
    return out
