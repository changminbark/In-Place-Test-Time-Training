"""single_needle: one fact in a long distractor document."""

from __future__ import annotations

import random
from typing import Dict

from .common import (
    TokenBudgeter,
    make_filler_sampler,
    place_needle,
    random_entity,
    random_number,
)


def gen_single_needle(
    tokenizer,
    target_tokens: int,
    position: str,
    seed: int,
    example_idx: int,
) -> Dict:
    rng = random.Random(f"{seed}:single_needle:{target_tokens}:{position}:{example_idx}")
    vault = random_entity(rng, "vault")
    code = random_number(rng, 10000, 99999)

    needle = f"The access code for {vault} is {code}."
    question = f"What is the access code for {vault}?"
    answer = str(code)

    sampler = make_filler_sampler(rng)
    frac = place_needle(position, rng)
    budgeter = TokenBudgeter(tokenizer=tokenizer, target_tokens=target_tokens)
    document = budgeter.fill(sampler, required_inserts=[(frac, needle)])

    return {
        "id": f"single_needle_{target_tokens}_{example_idx:04d}",
        "task": "single_needle",
        "context_length_target": target_tokens,
        "document": document,
        "question": question,
        "answer": answer,
        "answer_aliases": [answer],
        "metadata": {
            "needle_position": position,
            "seed": example_idx,
            "bridge_entity": vault,
        },
    }
