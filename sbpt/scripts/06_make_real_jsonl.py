import argparse
import json
import os

from datasets import load_dataset

from sbpt.utils.runtime import configure_runtime, setup_logging

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_out = os.path.join(repo_root, "out", "sbpt_real.jsonl")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset name, e.g. HuggingFaceFW/fineweb-edu")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--shuffle_buffer", type=int, default=20000)
    args = ap.parse_args()

    log = setup_logging("sbpt.make_real")
    configure_runtime(logger=log)
    log.info(
        "dataset=%s split=%s text_field=%s n=%s shuffle_buffer=%s",
        args.dataset,
        args.split,
        args.text_field,
        args.n,
        args.shuffle_buffer,
    )

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    if args.shuffle_buffer and args.shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wrote = 0
    with open(args.out, "w", encoding="utf-8") as f:
        buffer = []
        buffer_limit = 4096
        dumps = json.dumps
        for ex in ds:
            if wrote >= args.n:
                break
            txt = ex.get(args.text_field, None)
            if not txt or not isinstance(txt, str):
                continue
            txt = txt.strip()
            if len(txt) < 64:
                continue
            # Use current loader format: prompt+completion
            row = {"prompt": "", "completion": txt}
            buffer.append(dumps(row, ensure_ascii=False, separators=(",", ":")))
            wrote += 1
            if wrote % 10000 == 0:
                log.info("wrote_rows=%s", wrote)
            if len(buffer) >= buffer_limit:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()
        if buffer:
            f.write("\n".join(buffer) + "\n")
    log.info("done wrote_rows=%s path=%s", wrote, args.out)

if __name__ == "__main__":
    main()
