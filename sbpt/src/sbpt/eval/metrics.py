"""Evaluation metrics."""

from __future__ import annotations

import json
import re
from typing import Iterable

import torch
import torch.nn.functional as F


def exact_match(preds: Iterable[str], targets: Iterable[str]) -> float:
    pairs = list(zip(preds, targets))
    if not pairs:
        return 0.0
    correct = sum(int(p.strip() == t.strip()) for p, t in pairs)
    return correct / len(pairs)


def exact_match_stepwise(preds: Iterable[str], targets: Iterable[str]) -> tuple[float, float]:
    pairs = list(zip(preds, targets))
    if not pairs:
        return 0.0, 0.0
    strict = 0
    normalized = 0
    for pred, target in pairs:
        pred_str = pred.strip()
        target_str = target.strip()
        if pred_str == target_str:
            strict += 1
        pred_norm = pred_str.replace(" ", "")
        target_norm = target_str.replace(" ", "")
        if pred_norm == target_norm:
            normalized += 1
    total = len(pairs)
    return strict / total, normalized / total


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    correct = (preds == labels) & mask
    return float(correct.sum().item() / mask.sum().item())


def constraint_satisfaction(texts: Iterable[str], pattern: str) -> float:
    regex = re.compile(pattern)
    texts_list = list(texts)
    if not texts_list:
        return 0.0
    ok = sum(1 for t in texts_list if regex.search(t))
    return ok / len(texts_list)


def format_strict_pass(pred: str) -> bool:
    text = pred.strip()
    if not text:
        return False
    if text[0] not in "{[" or text[-1] not in "}]":
        return False
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def stepwise_format_pass(pred: str) -> bool:
    text = pred.strip()
    if not text:
        return False
    return re.fullmatch(r"\d(?: \d)*", text) is not None


def equiv_consistency(outputs: list[str], equiv_ids: list[int]) -> float:
    if not outputs:
        return 0.0
    groups: dict[int, set[str]] = {}
    for output, equiv_id in zip(outputs, equiv_ids):
        if equiv_id < 0:
            continue
        groups.setdefault(equiv_id, set()).add(output.strip())
    if not groups:
        return 0.0
    consistent = sum(1 for vals in groups.values() if len(vals) == 1)
    return consistent / len(groups)


def expected_calibration_error(
    confidences: Iterable[float],
    correct: Iterable[bool],
    n_bins: int = 10,
) -> float:
    conf_list = list(confidences)
    corr_list = list(correct)
    if not conf_list:
        return 0.0
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    conf = torch.tensor(conf_list)
    corr = torch.tensor(corr_list, dtype=torch.float32)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (conf >= bins[i]) & (conf < bins[i + 1])
        if in_bin.sum() == 0:
            continue
        acc = corr[in_bin].mean().item()
        avg_conf = conf[in_bin].mean().item()
        ece += abs(avg_conf - acc) * (in_bin.float().mean().item())
    return float(ece)


def brier_score(confidences: Iterable[float], correct: Iterable[bool]) -> float:
    conf = torch.tensor(list(confidences))
    corr = torch.tensor(list(correct), dtype=torch.float32)
    if conf.numel() == 0:
        return 0.0
    return float(((conf - corr) ** 2).mean().item())


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    vocab = logits.shape[-1]
    labels_flat = labels.view(-1)
    logp_flat = log_probs.view(-1, vocab)
    mask = labels_flat != -100
    if mask.sum() == 0:
        return torch.zeros(logits.shape[0])
    selected = logp_flat[torch.arange(labels_flat.shape[0]), labels_flat]
    selected = selected * mask
    seq_logp = selected.view(labels.shape).sum(dim=-1)
    return seq_logp
