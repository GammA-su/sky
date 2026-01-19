"""Generate bridge dataset with final-sum targets and stepwise supervision."""

from __future__ import annotations

import argparse
import json
import os
import random

from sbpt.utils.runtime import configure_runtime, setup_logging


def _digits_for(value: int, total_digits: int) -> list[int]:
    digits = [int(ch) for ch in reversed(str(value))]
    if len(digits) < total_digits:
        digits.extend([0] * (total_digits - len(digits)))
    return digits


def _encode_carry_state(pos_index: int, carry: int) -> int:
    return pos_index * 2 + carry


def make_bridge_example(
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
    stepwise_completion = " ".join(sum_str)

    total_digits = len(sum_str)
    a_rev = _digits_for(a, total_digits)
    b_rev = _digits_for(b, total_digits)

    carry = 0
    state_ids: list[int] = []
    for pos in range(total_digits):
        state_ids.append(_encode_carry_state(pos, carry))
        digit_sum = a_rev[pos] + b_rev[pos] + carry
        carry = 1 if digit_sum >= 10 else 0

    state_spans: list[tuple[int, int, int]] = []
    start_char = 0
    for idx in range(total_digits):
        pos_index = total_digits - 1 - idx
        state_value = state_ids[pos_index]
        state_spans.append((start_char, start_char + 1, state_value))
        start_char += 2

    return {
        "prompt": prompt,
        "completion": completion,
        "stepwise_completion": stepwise_completion,
        "state_ids": state_ids,
        "state_spans": state_spans,
        "task_type": "add_carry_bridge",
    }


def main() -> None:
    logger = setup_logging("sbpt.make_add_carry_bridge")
    configure_runtime(logger=logger)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_out = os.path.join(repo_root, "out", "train_add_carry_bridge.jsonl")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--n", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-digits", "--min_digits", dest="min_digits", type=int, default=6)
    parser.add_argument("--max-digits", "--max_digits", dest="max_digits", type=int, default=9)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    buffer: list[str] = []
    buffer_limit = 4096
    dumps = json.dumps
    wrote = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            row = make_bridge_example(rng, min_digits=args.min_digits, max_digits=args.max_digits)
            buffer.append(dumps(row, ensure_ascii=False, separators=(",", ":")))
            wrote += 1
            if len(buffer) >= buffer_limit:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()
        if buffer:
            f.write("\n".join(buffer) + "\n")
    logger.info("wrote_rows=%s path=%s", wrote, args.out)


if __name__ == "__main__":
    main()
