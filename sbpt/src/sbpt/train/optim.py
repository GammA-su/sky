"""Optimizer and scheduler."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> AdamW:
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: AdamW, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        return max(0.0, 1.0 - (step / total_steps))
    return LambdaLR(optimizer, lr_lambda)
