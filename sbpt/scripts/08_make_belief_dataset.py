import argparse
import json
import os

from sbpt.data.enumerators import enumerate_hypotheses
from sbpt.utils.runtime import configure_runtime, setup_logging


def main():
    ap = argparse.ArgumentParser()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_out = os.path.join(repo_root, "out", "sbpt_belief.jsonl")
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--max_h", type=int, default=32)
    args = ap.parse_args()

    log = setup_logging("sbpt.make_belief")
    configure_runtime(logger=log)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    wrote = 0
    kept = 0
    log_every = 20000
    buffer = []
    buffer_limit = 4096
    dumps = json.dumps
    with open(args.in_path, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            ex = json.loads(line)
            # Enumerate hypotheses deterministically from the example
            hyps, mask = enumerate_hypotheses(ex, max_h=args.max_h)
            if not hyps or not any(mask):
                wrote += 1
                continue
            ex["hypotheses"] = hyps
            ex["correct_mask"] = mask
            buffer.append(dumps(ex, ensure_ascii=False, separators=(",", ":")))
            wrote += 1
            kept += 1
            if kept % log_every == 0:
                log.info("kept_rows=%s processed_rows=%s", kept, wrote)
            if len(buffer) >= buffer_limit:
                fout.write("\n".join(buffer) + "\n")
                buffer.clear()
        if buffer:
            fout.write("\n".join(buffer) + "\n")
    log.info("done processed_rows=%s kept_rows=%s out=%s", wrote, kept, args.out)


if __name__ == "__main__":
    main()
