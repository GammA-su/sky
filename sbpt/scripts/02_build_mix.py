"""Build a mixed dataset from JSONL sources."""

from __future__ import annotations

import argparse
import os

from sbpt.utils.runtime import configure_runtime, setup_logging


def main() -> None:
    logger = setup_logging("sbpt.build_mix")
    configure_runtime(logger=logger)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_out = os.path.join(repo_root, "out", "sbpt_mix.jsonl")
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default=default_out)
    args = parser.parse_args()

    count = 0
    with open(args.out, "w", encoding="utf-8") as out_fp:
        for path in args.inputs:
            with open(path, "r", encoding="utf-8") as in_fp:
                for line in in_fp:
                    line = line.strip()
                    if not line:
                        continue
                    out_fp.write(line + "\n")
                    count += 1
    logger.info("wrote_rows=%s path=%s", count, args.out)


if __name__ == "__main__":
    main()
