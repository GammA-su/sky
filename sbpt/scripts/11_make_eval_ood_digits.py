import argparse
import json
import os
import random

from sbpt.utils.runtime import configure_runtime, setup_logging


def main() -> None:
    logger = setup_logging("sbpt.eval_ood_digits")
    configure_runtime(logger=logger)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_out = os.path.join(repo_root, "out", "eval_ood_digits.jsonl")

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-digits", type=int, default=6)
    ap.add_argument("--max-digits", type=int, default=9)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    buffer = []
    buffer_limit = 4096
    dumps = json.dumps
    wrote = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            a_digits = rng.randint(args.min_digits, args.max_digits)
            b_digits = rng.randint(args.min_digits, args.max_digits)
            a = rng.randint(10 ** (a_digits - 1), 10 ** a_digits - 1)
            b = rng.randint(10 ** (b_digits - 1), 10 ** b_digits - 1)
            prompt = f"Add {a} and {b}. Answer:"
            completion = f" {a + b}"
            row = {"prompt": prompt, "completion": completion, "state_ids": [0, 1, 2]}
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
