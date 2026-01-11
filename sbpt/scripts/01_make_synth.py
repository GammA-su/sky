"""Generate synthetic traces dataset."""

from __future__ import annotations

import argparse
import json

from sbpt.data.synth_traces import iter_synth
from sbpt.utils.runtime import configure_runtime, setup_logging


def main() -> None:
    logger = setup_logging("sbpt.make_synth")
    configure_runtime(logger=logger)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-digits", type=int, default=2)
    parser.add_argument("--max-digits", type=int, default=3)
    args = parser.parse_args()

    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in iter_synth(
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
