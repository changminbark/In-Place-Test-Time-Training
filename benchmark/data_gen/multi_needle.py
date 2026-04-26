"""multi_needle: two linked facts, answer requires composing them."""

from __future__ import annotations

import random
from typing import Dict

from .common import (
    TokenBudgeter,
    make_filler_sampler,
    place_needle,
    random_entity,
    random_number,
    random_person,
)


def gen_multi_needle(
    tokenizer,
    target_tokens: int,
    position_1: str,
    position_2: str,
    seed: int,
    example_idx: int,
) -> Dict:
    rng = random.Random(
        f"{seed}:multi_needle:{target_tokens}:{position_1}:{position_2}:{example_idx}"
    )
    archive = random_entity(rng, "archive")
    person = random_person(rng)
    extension = random_number(rng, 1000, 9999)

    needle_1 = f"{archive} is managed by {person}."
    needle_2 = f"{person}'s extension is {extension}."
    question = f"What is the extension for the manager of {archive}?"
    answer = str(extension)

    sampler = make_filler_sampler(rng)
    frac_1 = place_needle(position_1, rng)
    frac_2 = place_needle(position_2, rng)
    budgeter = TokenBudgeter(tokenizer=tokenizer, target_tokens=target_tokens)
    document = budgeter.fill(
        sampler, required_inserts=[(frac_1, needle_1), (frac_2, needle_2)]
    )

    return {
        "id": f"multi_needle_{target_tokens}_{example_idx:04d}",
        "task": "multi_needle",
        "context_length_target": target_tokens,
        "document": document,
        "question": question,
        "answer": answer,
        "answer_aliases": [answer],
        "metadata": {
            "needle_position_1": position_1,
            "needle_position_2": position_2,
            "bridge_entity": person,
            "seed": example_idx,
        },
    }
