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


def _build_state_spans(a: int, b: int, sum_str: str) -> tuple[list[int], list[tuple[int, int, int]]]:
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
    return state_ids, state_spans


def make_pair(rng: random.Random, equiv_id: int, min_digits: int, max_digits: int) -> tuple[dict, dict]:
    a_digits = rng.randint(min_digits, max_digits)
    b_digits = rng.randint(min_digits, max_digits)
    a = rng.randint(10 ** (a_digits - 1), 10 ** a_digits - 1)
    b = rng.randint(10 ** (b_digits - 1), 10 ** b_digits - 1)
    sum_str = str(a + b)
    completion = " ".join(sum_str)
    state_ids, state_spans = _build_state_spans(a, b, sum_str)

    p1 = f"Add {a} and {b}. Reply with digits separated by spaces."
    distract = rng.choice(
        [
            "Ignore extra whitespace in the prompt.",
            "Note: format must be digits separated by spaces.",
            "Extra text is noise.",
        ]
    )
    p2 = f"{distract}\nCompute {a} + {b} and output digits separated by spaces only."

    base = {
        "task_type": "add_carry_stepwise",
        "state_ids": state_ids,
        "state_spans": state_spans,
    }
    r1 = {"prompt": p1, "completion": completion, "equiv_id": equiv_id, **base}
    r2 = {"prompt": p2, "completion": completion, "equiv_id": equiv_id, **base}
    return r1, r2


def main() -> None:
    logger = setup_logging("sbpt.eval_robust_pairs_stepwise")
    configure_runtime(logger=logger)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_out = os.path.join(repo_root, "out", "eval_robust_pairs_stepwise.jsonl")

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-digits", "--min_digits", dest="min_digits", type=int, default=6)
    ap.add_argument("--max-digits", "--max_digits", dest="max_digits", type=int, default=9)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    buffer = []
    buffer_limit = 4096
    dumps = json.dumps
    wrote = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.pairs):
            r1, r2 = make_pair(rng, equiv_id=i, min_digits=args.min_digits, max_digits=args.max_digits)
            buffer.append(dumps(r1, ensure_ascii=False, separators=(",", ":")))
            buffer.append(dumps(r2, ensure_ascii=False, separators=(",", ":")))
            wrote += 2
            if len(buffer) >= buffer_limit:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()
        if buffer:
            f.write("\n".join(buffer) + "\n")
    logger.info("wrote_rows=%s path=%s", wrote, args.out)


if __name__ == "__main__":
    main()
