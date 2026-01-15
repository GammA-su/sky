"""Evaluation reporting."""

from __future__ import annotations

from typing import Any

import os

from sbpt.utils.io import save_json


def save_report(metrics: dict[str, Any], path: str | None = None) -> None:
    if path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        path = os.path.join(repo_root, "out", "sbpt_report.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_json(path, metrics)
