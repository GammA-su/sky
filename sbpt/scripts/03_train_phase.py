"""Train a SBPT phase."""

from __future__ import annotations

import argparse
import os

from sbpt.train.loop import train
from sbpt.utils.io import load_yaml
from sbpt.utils.runtime import configure_runtime, setup_logging


def main() -> None:
    logger = setup_logging("sbpt.train_phase")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--init", default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    runtime = configure_runtime(prefer_gpu=not args.cpu, logger=logger)
    use_gpu = runtime["use_gpu"]
    logger.info("train_use_gpu=%s", use_gpu)
    cfg = load_yaml(args.config)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_ckpt = os.path.join(repo_root, "out", "sbpt_ckpt.pt")
    cfg.setdefault("train", {})
    cfg["train"].setdefault("save_path", default_ckpt)
    if args.out:
        cfg["train"]["save_path"] = args.out
    if args.init:
        cfg["train"]["init_ckpt"] = args.init
    train(cfg, data_path=args.data, steps_override=args.steps, cpu=not use_gpu)


if __name__ == "__main__":
    main()
