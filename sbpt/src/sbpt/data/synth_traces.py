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


def _digits_for(value: int, total_digits: int) -> list[int]:
    digits = [int(ch) for ch in reversed(str(value))]
    if len(digits) < total_digits:
        digits.extend([0] * (total_digits - len(digits)))
    return digits


def _encode_carry_state(pos_index: int, carry: int) -> int:
    return pos_index * 2 + carry


def generate_add_carry_trace_example(
    rng: random.Random,
    min_digits: int,
    max_digits: int,
) -> dict[str, object]:
    a_digits = rng.randint(min_digits, max_digits)
    b_digits = rng.randint(min_digits, max_digits)
    a = rng.randint(10 ** (a_digits - 1), 10 ** a_digits - 1)
    b = rng.randint(10 ** (b_digits - 1), 10 ** b_digits - 1)
    prompt = f"Add {a} and {b}. Reply with only the final sum."
    sum_str = str(a + b)
    completion = f" {sum_str}"

    total_digits = len(sum_str)
    a_rev = _digits_for(a, total_digits)
    b_rev = _digits_for(b, total_digits)

    carry = 0
    state_ids: list[int] = []
    for pos in range(total_digits):
        state_ids.append(_encode_carry_state(pos, carry))
        digit_sum = a_rev[pos] + b_rev[pos] + carry
        carry = 1 if digit_sum >= 10 else 0

    digit_offset = len(completion) - len(sum_str)
    state_spans: list[tuple[int, int, int]] = []
    if digit_offset > 0:
        state_spans.append((0, digit_offset, state_ids[-1]))
    for idx in range(len(sum_str)):
        pos_index = total_digits - 1 - idx
        state_value = state_ids[pos_index]
        start_char = digit_offset + idx
        state_spans.append((start_char, start_char + 1, state_value))

    return {
        "prompt": prompt,
        "completion": completion,
        "state_ids": state_ids,
        "state_spans": state_spans,
        "task_type": "add_carry_trace",
    }


def generate_add_carry_trace_samples(
    n: int,
    seed: int = 0,
    min_digits: int = 6,
    max_digits: int = 9,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for _ in range(n):
        rows.append(generate_add_carry_trace_example(rng, min_digits=min_digits, max_digits=max_digits))
    return rows


def iter_add_carry_trace(
    n: int,
    seed: int = 0,
    min_digits: int = 6,
    max_digits: int = 9,
) -> Iterable[dict[str, object]]:
    rng = random.Random(seed)
    for _ in range(n):
        yield generate_add_carry_trace_example(rng, min_digits=min_digits, max_digits=max_digits)
