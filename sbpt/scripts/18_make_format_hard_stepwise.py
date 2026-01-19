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


def make_pair(rng: random.Random, task_id: int, min_digits: int, max_digits: int) -> list[dict]:
    a_digits = rng.randint(min_digits, max_digits)
    b_digits = rng.randint(min_digits, max_digits)
    a = rng.randint(10 ** (a_digits - 1), 10 ** a_digits - 1)
    b = rng.randint(10 ** (b_digits - 1), 10 ** b_digits - 1)
    sum_str = str(a + b)

    temptation = rng.choice(
        [
            "Include a short explanation after the digits.",
            "Add extra whitespace.",
            "Prefix the answer with 'sum:'.",
        ]
    )
    prompt = f"Add {a} and {b}. Reply with digits separated by spaces. {temptation}"
    good = " ".join(sum_str)
    state_ids, state_spans = _build_state_spans(a, b, sum_str)

    corrupt_type = rng.choice(["double_space", "newline", "prefix", "suffix", "compact"])
    if corrupt_type == "double_space":
        bad = good.replace(" ", "  ", 1)
    elif corrupt_type == "newline":
        bad = good.replace(" ", "\n", 1)
    elif corrupt_type == "prefix":
        bad = f"sum: {good}"
    elif corrupt_type == "suffix":
        bad = f"{good} thanks"
    else:
        bad = good.replace(" ", "")

    base = {
        "task_type": "add_carry_stepwise",
        "state_ids": state_ids,
        "state_spans": state_spans,
    }
    return [
        {
            **base,
            "prompt": prompt,
            "completion": good,
            "verify_label": 1,
            "verify_type": "stepwise_format",
            "task_id": task_id,
        },
        {
            **base,
            "prompt": prompt,
            "completion": bad,
            "verify_label": 0,
            "verify_type": "stepwise_format",
            "task_id": task_id,
        },
    ]


def main() -> None:
    logger = setup_logging("sbpt.eval_format_hard_stepwise")
    configure_runtime(logger=logger)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_out = os.path.join(repo_root, "out", "eval_format_hard_stepwise.jsonl")

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--pairs", "--n", dest="pairs", type=int, default=200)
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
            rows = make_pair(rng, task_id=i, min_digits=args.min_digits, max_digits=args.max_digits)
            for row in rows:
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
