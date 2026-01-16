"""Checkpoint helpers."""

from __future__ import annotations

from typing import Any, Tuple

import logging
import torch


def save_ckpt(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: dict[str, Any], step: int) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
    }
    torch.save(state, path)


def _filter_state_dict(
    model_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, tensor in model_state.items():
        target = target_state.get(key)
        if target is None:
            skipped.append(key)
            continue
        if target.shape != tensor.shape:
            skipped.append(key)
            continue
        filtered[key] = tensor
    return filtered, skipped


def load_ckpt(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> Tuple[dict[str, Any], int]:
    state = torch.load(path, map_location="cpu")
    logger = logging.getLogger("sbpt.ckpt")
    if strict:
        model.load_state_dict(state["model"])
    else:
        target_state = model.state_dict()
        filtered, skipped = _filter_state_dict(state["model"], target_state)
        if skipped:
            logger.warning("ckpt_skipped_keys=%s", ",".join(skipped))
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            logger.warning("ckpt_missing_keys=%s", ",".join(missing))
        if unexpected:
            logger.warning("ckpt_unexpected_keys=%s", ",".join(unexpected))
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    config = state.get("config", {})
    step = int(state.get("step", 0))
    return config, step
