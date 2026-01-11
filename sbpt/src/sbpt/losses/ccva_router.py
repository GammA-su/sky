"""Counterfactual value routing loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ccva_router_loss(
    delta_hat: torch.Tensor,
    delta: torch.Tensor,
    continue_prob: torch.Tensor,
    target_continue: float = 0.5,
) -> dict[str, torch.Tensor]:
    if delta.numel() == 1:
        delta = delta.expand_as(delta_hat)
    mse = F.mse_loss(delta_hat, delta)
    budget = (continue_prob.mean() - target_continue) ** 2
    loss = mse + budget
    return {"loss_router": loss, "loss_mse": mse, "loss_budget": budget}
