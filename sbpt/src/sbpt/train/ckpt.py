"""Checkpoint helpers."""

from __future__ import annotations

from typing import Any, Tuple

import torch


def save_ckpt(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: dict[str, Any], step: int) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
    }
    torch.save(state, path)


def load_ckpt(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[dict[str, Any], int]:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    config = state.get("config", {})
    step = int(state.get("step", 0))
    return config, step
