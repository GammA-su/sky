"""Cheap hypothesis enumerators."""

from __future__ import annotations

import re
from typing import List, Tuple


def _extract_numbers(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"\d+", text)]


def enumerate_hypotheses(example: dict, max_h: int = 8) -> Tuple[List[str], List[bool]]:
    prompt = str(example.get("prompt", ""))
    numbers = _extract_numbers(prompt)
    if len(numbers) < 2:
        return [], []
    a, b = numbers[0], numbers[1]
    target = a + b
    candidates = [
        (f"{a}+{b}", a + b),
        (f"{a}-{b}", a - b),
        (f"{b}-{a}", b - a),
        (f"{a}*{b}", a * b),
        (f"{a}", a),
        (f"{b}", b),
    ]
    hypotheses: List[str] = []
    correct: List[bool] = []
    for expr, value in candidates:
        if len(hypotheses) >= max_h:
            break
        hypotheses.append(expr)
        correct.append(value == target)
    return hypotheses, correct
