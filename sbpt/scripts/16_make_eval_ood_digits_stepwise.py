import argparse
import json
import os

from sbpt.data.synth_traces import iter_add_carry_stepwise
from sbpt.utils.runtime import configure_runtime, setup_logging


def main() -> None:
    logger = setup_logging("sbpt.eval_ood_digits_stepwise")
    configure_runtime(logger=logger)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_out = os.path.join(repo_root, "out", "eval_ood_digits_stepwise.jsonl")

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-digits", "--min_digits", dest="min_digits", type=int, default=6)
    ap.add_argument("--max-digits", "--max_digits", dest="max_digits", type=int, default=9)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    buffer = []
    buffer_limit = 4096
    dumps = json.dumps
    wrote = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in iter_add_carry_stepwise(
            n=args.n,
            seed=args.seed,
            min_digits=args.min_digits,
            max_digits=args.max_digits,
        ):
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
