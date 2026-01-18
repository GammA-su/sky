"""Generate stepwise digit addition dataset with carry states."""

from __future__ import annotations

import argparse
import json
import os

from sbpt.data.synth_traces import iter_add_carry_stepwise
from sbpt.utils.runtime import configure_runtime, setup_logging


def main() -> None:
    logger = setup_logging("sbpt.make_add_carry_stepwise")
    configure_runtime(logger=logger)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_out = os.path.join(repo_root, "out", "train_add_carry_stepwise.jsonl")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--n", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-digits", "--min_digits", dest="min_digits", type=int, default=6)
    parser.add_argument("--max-digits", "--max_digits", dest="max_digits", type=int, default=9)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in iter_add_carry_stepwise(
            n=args.n,
            seed=args.seed,
            min_digits=args.min_digits,
            max_digits=args.max_digits,
        ):
            f.write(json.dumps(row) + "\n")
            count += 1
    logger.info("wrote_rows=%s path=%s", count, args.out)


if __name__ == "__main__":
    main()
