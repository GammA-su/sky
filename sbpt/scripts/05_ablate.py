"""Ablation runner stub."""

from __future__ import annotations

import argparse

from sbpt.utils.runtime import configure_runtime, setup_logging


def main() -> None:
    logger = setup_logging("sbpt.ablate")
    configure_runtime(logger=logger)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    logger.info("ablation_stub=true config=%s", args.config)


if __name__ == "__main__":
    main()
