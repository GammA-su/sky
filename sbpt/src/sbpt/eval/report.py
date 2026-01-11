"""Evaluation reporting."""

from __future__ import annotations

from typing import Any

from sbpt.utils.io import save_json


def save_report(metrics: dict[str, Any], path: str = "/tmp/sbpt_report.json") -> None:
    save_json(path, metrics)
