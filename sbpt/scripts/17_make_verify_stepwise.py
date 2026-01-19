import argparse
import json
import os
import random

from sbpt.utils.runtime import configure_runtime, setup_logging


def make_pair(rng: random.Random, min_digits: int, max_digits: int) -> list[dict]:
    a_digits = rng.randint(min_digits, max_digits)
    b_digits = rng.randint(min_digits, max_digits)
    a = rng.randint(10 ** (a_digits - 1), 10 ** a_digits - 1)
    b = rng.randint(10 ** (b_digits - 1), 10 ** b_digits - 1)
    sum_str = str(a + b)
    prompt = f"Add {a} and {b}. Reply with digits separated by spaces."
    good = " ".join(sum_str)

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

    return [
        {
            "prompt": prompt,
            "completion": good,
            "verify_label": 1,
            "verify_type": "stepwise_format",
            "task_type": "add_carry_stepwise",
        },
        {
            "prompt": prompt,
            "completion": bad,
            "verify_label": 0,
            "verify_type": "stepwise_format",
            "task_type": "add_carry_stepwise",
        },
    ]


def main() -> None:
    logger = setup_logging("sbpt.make_verify_stepwise")
    configure_runtime(logger=logger)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_out = os.path.join(repo_root, "out", "sbpt_verify_stepwise.jsonl")

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-digits", "--min_digits", dest="min_digits", type=int, default=2)
    ap.add_argument("--max-digits", "--max_digits", dest="max_digits", type=int, default=5)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    buffer = []
    buffer_limit = 4096
    dumps = json.dumps
    wrote = 0
    log_every = 20000
    with open(args.out, "w", encoding="utf-8") as f:
        while wrote < args.n:
            rows = make_pair(rng, min_digits=args.min_digits, max_digits=args.max_digits)
            for row in rows:
                if wrote >= args.n:
                    break
                buffer.append(dumps(row, ensure_ascii=False, separators=(",", ":")))
                wrote += 1
            if wrote % log_every == 0:
                logger.info("wrote_rows=%s", wrote)
            if len(buffer) >= buffer_limit:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()
        if buffer:
            f.write("\n".join(buffer) + "\n")
    logger.info("done wrote_rows=%s path=%s", wrote, args.out)


if __name__ == "__main__":
    main()
