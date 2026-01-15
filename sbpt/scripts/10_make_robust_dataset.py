import argparse
import json
import os
import random

from sbpt.utils.runtime import configure_runtime, setup_logging

def make_pair(rng: random.Random, equiv_id: int):
    a = rng.randint(10, 999)
    b = rng.randint(10, 999)
    s = a + b

    # Two paraphrase/distractor variants with the SAME expected completion
    p1 = f"Compute the sum: {a} + {b}. Give only the number."
    distract = rng.choice([
        "Ignore any irrelevant text.",
        "Note: this line is noise and should not matter.",
        "Some people like cats; this sentence is unrelated.",
    ])
    p2 = f"{distract} What is {a} plus {b}? Output only the number."

    completion = str(s)

    r1 = {"prompt": p1, "completion": completion, "equiv_id": equiv_id, "robust_type": "paraphrase"}
    r2 = {"prompt": p2, "completion": completion, "equiv_id": equiv_id, "robust_type": "distractor"}
    return r1, r2

def main():
    ap = argparse.ArgumentParser()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_out = os.path.join(repo_root, "out", "sbpt_robust.jsonl")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--pairs", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    log = setup_logging("sbpt.make_robust")
    configure_runtime(logger=log)
    rng = random.Random(args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wrote = 0
    with open(args.out, "w", encoding="utf-8") as f:
        buffer = []
        buffer_limit = 4096
        dumps = json.dumps
        log_every = 20000
        for i in range(args.pairs):
            r1, r2 = make_pair(rng, equiv_id=i)
            buffer.append(dumps(r1, ensure_ascii=False, separators=(",", ":")))
            buffer.append(dumps(r2, ensure_ascii=False, separators=(",", ":")))
            wrote += 2
            if wrote % log_every == 0:
                log.info("wrote_rows=%s pairs_done=%s", wrote, i + 1)
            if len(buffer) >= buffer_limit:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()
        if buffer:
            f.write("\n".join(buffer) + "\n")
    log.info("done wrote_rows=%s path=%s", wrote, args.out)

if __name__ == "__main__":
    main()
