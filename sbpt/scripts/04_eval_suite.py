"""Run evaluation suite."""

from __future__ import annotations

import argparse

from sbpt.eval.report import save_report
from sbpt.eval.suite import run_eval
from sbpt.utils.runtime import configure_runtime, setup_logging


def main() -> None:
    logger = setup_logging("sbpt.eval_suite")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    runtime = configure_runtime(prefer_gpu=not args.cpu, logger=logger)
    use_gpu = runtime["use_gpu"]
    logger.info("eval_use_gpu=%s", use_gpu)
    metrics = run_eval(args.ckpt, cfg_path=args.config, cpu=not use_gpu)
    save_report(metrics, path=args.report)
    logger.info("metrics=%s", metrics)


if __name__ == "__main__":
    main()
