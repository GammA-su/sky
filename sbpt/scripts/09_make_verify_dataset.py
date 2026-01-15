import argparse
import json
import os
import random

from sbpt.utils.runtime import configure_runtime, setup_logging

def make_task(rng: random.Random):
    # Simple deterministic JSON constraint task
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    s = a + b
    prompt = (
        "Return STRICT JSON only with keys: "
        '{"a":int,"b":int,"sum":int}. '
        f"Use a={a} and b={b}."
    )
    good = json.dumps({"a": a, "b": b, "sum": s}, ensure_ascii=False, separators=(",", ":"))
    # Corruptions: missing brace / wrong key / wrong value type
    corrupt_type = rng.choice(["missing_brace", "wrong_key", "string_sum"])
    if corrupt_type == "missing_brace":
        bad = good[:-1]  # drop closing brace
    elif corrupt_type == "wrong_key":
        bad = json.dumps({"aa": a, "b": b, "sum": s}, ensure_ascii=False, separators=(",", ":"))
    else:
        bad = json.dumps({"a": a, "b": b, "sum": str(s)}, ensure_ascii=False, separators=(",", ":"))

    # We store verifier labels for the completion itself
    return [
        {"prompt": prompt, "completion": good, "verify_label": 1, "verify_type": "json_schema"},
        {"prompt": prompt, "completion": bad, "verify_label": 0, "verify_type": "json_schema"},
    ]

def main():
    ap = argparse.ArgumentParser()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_out = os.path.join(repo_root, "out", "sbpt_verify.jsonl")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    log = setup_logging("sbpt.make_verify")
    configure_runtime(logger=log)
    rng = random.Random(args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wrote = 0
    with open(args.out, "w", encoding="utf-8") as f:
        buffer = []
        buffer_limit = 4096
        dumps = json.dumps
        log_every = 20000
        while wrote < args.n:
            rows = make_task(rng)
            for r in rows:
                if wrote >= args.n:
                    break
                buffer.append(dumps(r, ensure_ascii=False, separators=(",", ":")))
                wrote += 1
            if wrote % log_every == 0:
                log.info("wrote_rows=%s", wrote)
            if len(buffer) >= buffer_limit:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()
        if buffer:
            f.write("\n".join(buffer) + "\n")
    log.info("done wrote_rows=%s path=%s", wrote, args.out)

if __name__ == "__main__":
    main()
