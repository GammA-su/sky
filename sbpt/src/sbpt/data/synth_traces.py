"""Synthetic trace generators."""

from __future__ import annotations

import random
from typing import Iterable

from sbpt.data.schemas import SynthExample


STATE_PARSE = 0
STATE_COMPUTE = 1
STATE_RENDER = 2


def generate_addition_example(rng: random.Random, min_digits: int, max_digits: int) -> SynthExample:
    a_digits = rng.randint(min_digits, max_digits)
    b_digits = rng.randint(min_digits, max_digits)
    a = rng.randint(10 ** (a_digits - 1), 10 ** a_digits - 1)
    b = rng.randint(10 ** (b_digits - 1), 10 ** b_digits - 1)
    prompt = f"Add {a} and {b}. Answer:"
    completion = f" {a + b}"
    state_ids = [STATE_PARSE, STATE_COMPUTE, STATE_RENDER]
    return SynthExample(prompt=prompt, completion=completion, state_ids=state_ids)


def generate_addition_samples(
    n: int,
    seed: int = 0,
    min_digits: int = 2,
    max_digits: int = 3,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for _ in range(n):
        ex = generate_addition_example(rng, min_digits=min_digits, max_digits=max_digits)
        rows.append(
            {
                "prompt": ex.prompt,
                "completion": ex.completion,
                "state_ids": ex.state_ids,
            }
        )
    return rows


def iter_synth(
    n: int,
    seed: int = 0,
    min_digits: int = 2,
    max_digits: int = 3,
) -> Iterable[dict[str, object]]:
    rng = random.Random(seed)
    for _ in range(n):
        ex = generate_addition_example(rng, min_digits=min_digits, max_digits=max_digits)
        yield {
            "prompt": ex.prompt,
            "completion": ex.completion,
            "state_ids": ex.state_ids,
        }
