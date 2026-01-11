"""Distributed utility stubs."""

from __future__ import annotations

import os


def is_distributed() -> bool:
    return "RANK" in os.environ or "WORLD_SIZE" in os.environ


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0
