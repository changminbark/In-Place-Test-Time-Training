"""variable_tracking: many state updates, return the final value of a target."""

from __future__ import annotations

import random
from typing import Dict

from .common import TokenBudgeter, make_filler_sampler, random_entity, random_number


def gen_variable_tracking(
    tokenizer,
    target_tokens: int,
    seed: int,
    example_idx: int,
    num_variables: int = 4,
    min_updates_per_var: int = 3,
) -> Dict:
    rng = random.Random(f"{seed}:variable_tracking:{target_tokens}:{example_idx}")

    counters = [random_entity(rng, "counter") for _ in range(num_variables)]
    # Ensure uniqueness (Counter A/B/C/... pool has enough entries)
    counters = list(dict.fromkeys(counters))
    while len(counters) < num_variables:
        counters.append(random_entity(rng, "counter"))
        counters = list(dict.fromkeys(counters))

    target = rng.choice(counters)

    # We'll generate a stream of updates interleaved with filler. Build update
    # sentences and track ground-truth final values.
    state = {c: None for c in counters}
    updates = []
    update_count = {c: 0 for c in counters}

    # Seed each counter with min_updates_per_var updates, then keep adding until
    # the token budget is mostly filled; finalize by updating target last.
    def make_update_sentence(counter: str, value: int) -> str:
        return f"{counter} is now {value}."

    # Generate initial updates to meet minimum counts
    for c in counters:
        for _ in range(min_updates_per_var):
            v = random_number(rng, 1, 500)
            updates.append(make_update_sentence(c, v))
            state[c] = v
            update_count[c] += 1

    # Interleave: we'll add more updates as filler demand grows, but we also want
    # plain filler sentences in between so the task is actually hard.
    sampler = make_filler_sampler(rng)

    # Use a TokenBudgeter variant that alternates filler with additional updates.
    def interleaved_sampler() -> str:
        # 30% chance: add another update to a random counter (not target on final slot)
        if rng.random() < 0.3:
            c = rng.choice(counters)
            v = random_number(rng, 1, 500)
            state[c] = v
            update_count[c] += 1
            return make_update_sentence(c, v)
        return sampler()

    budgeter = TokenBudgeter(tokenizer=tokenizer, target_tokens=target_tokens)

    # We first emit the seeded updates as "required inserts" spread across the doc,
    # then fill the rest via the interleaved sampler.
    inserts = []
    n = len(updates)
    for i, sentence in enumerate(updates):
        frac = (i + 0.5) / n
        inserts.append((frac, sentence))

    document = budgeter.fill(interleaved_sampler, required_inserts=inserts)

    # Ensure the target's final update appears AFTER all other updates we've laid down,
    # by appending one authoritative final update near the end.
    final_value = random_number(rng, 1, 500)
    final_sentence = f"{target} is now {final_value}."
    document = document + " " + final_sentence
    state[target] = final_value
    update_count[target] += 1

    question = f"What is the final value of {target}?"
    answer = str(final_value)

    return {
        "id": f"variable_tracking_{target_tokens}_{example_idx:04d}",
        "task": "variable_tracking",
        "context_length_target": target_tokens,
        "document": document,
        "question": question,
        "answer": answer,
        "answer_aliases": [answer],
        "metadata": {
            "target_variable": target,
            "num_updates": update_count[target],
            "num_variables": num_variables,
            "seed": example_idx,
        },
    }
