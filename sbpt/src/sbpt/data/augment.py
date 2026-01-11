"""Data augmentation stubs for robustness phase."""

from __future__ import annotations

from typing import Iterable


def add_distractor(example: dict, distractor: str = "") -> dict:
    prompt = str(example.get("prompt", ""))
    if distractor:
        prompt = f"{prompt} {distractor}"
    out = dict(example)
    out["prompt"] = prompt
    return out


def paraphrase_identity(example: dict) -> dict:
    return dict(example)


def augment_batch(rows: Iterable[dict]) -> list[dict]:
    return [paraphrase_identity(row) for row in rows]
