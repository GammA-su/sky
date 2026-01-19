import argparse
import json
import os
import random

from sbpt.utils.runtime import configure_runtime, setup_logging


def make_pair(rng: random.Random, equiv_id: int, min_digits: int, max_digits: int) -> tuple[dict, dict]:
    a_digits = rng.randint(min_digits, max_digits)
    b_digits = rng.randint(min_digits, max_digits)
    a = rng.randint(10 ** (a_digits - 1), 10 ** a_digits - 1)
    b = rng.randint(10 ** (b_digits - 1), 10 ** b_digits - 1)
    sum_str = str(a + b)
    completion = " ".join(sum_str)

    p1 = f"Add {a} and {b}. Reply with digits separated by spaces."
    distract = rng.choice(
        [
            "Ignore extra whitespace in the prompt.",
            "Note: format must be digits separated by spaces.",
            "Extra text is noise.",
        ]
    )
    p2 = f"{distract}\nCompute {a} + {b} and output digits separated by spaces only."

    r1 = {"prompt": p1, "completion": completion, "equiv_id": equiv_id}
    r2 = {"prompt": p2, "completion": completion, "equiv_id": equiv_id}
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
