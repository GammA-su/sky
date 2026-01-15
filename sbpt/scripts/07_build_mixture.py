import argparse
import os
import random

from sbpt.utils.runtime import configure_runtime, setup_logging

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line

def main():
    ap = argparse.ArgumentParser()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_out = os.path.join(repo_root, "out", "sbpt_mixture.jsonl")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--src", action="append", required=True,
                    help="Repeated: name=PATH:weight  (e.g. synth=out/sbpt_synth.jsonl:40)")
    args = ap.parse_args()

    log = setup_logging("sbpt.mix")
    configure_runtime(logger=log)
    rnd = random.Random(args.seed)

    sources = []
    for item in args.src:
        name, rest = item.split("=", 1)
        path, w = rest.rsplit(":", 1)
        w = int(w)
        sources.append((name, path, w))
    log.info("sources=%s", " ".join([f"{n}:{w}" for n, _, w in sources]))

    paths = [p for (_, p, _) in sources]
    iters = [(n, iter_jsonl(p), w) for (n, p, w) in sources]
    bag = []
    for i, (_, _, w) in enumerate(iters):
        bag.extend([i] * w)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wrote = 0
    log_every = 20000
    with open(args.out, "w", encoding="utf-8") as out:
        buffer = []
        buffer_limit = 4096
        randrange = rnd.randrange
        bag_len = len(bag)
        while wrote < args.n:
            i = bag[randrange(bag_len)]
            n, it, _ = iters[i]
            try:
                row = next(it)
            except StopIteration:
                # restart exhausted iterator (deterministic behavior preserved by fixed seed + same file content)
                iters[i] = (n, iter_jsonl(paths[i]), iters[i][2])
                row = next(iters[i][1])
            buffer.append(row)
            wrote += 1
            if wrote % log_every == 0:
                log.info("wrote_rows=%s", wrote)
            if len(buffer) >= buffer_limit:
                out.write("\n".join(buffer) + "\n")
                buffer.clear()
        if buffer:
            out.write("\n".join(buffer) + "\n")
    log.info("done wrote_rows=%s path=%s", wrote, args.out)

if __name__ == "__main__":
    main()
