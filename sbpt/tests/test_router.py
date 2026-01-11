import torch

from sbpt.losses.ccva_router import ccva_router_loss


def test_router_loss_finite() -> None:
    delta_hat = torch.tensor([0.1, 0.2])
    delta = torch.tensor([0.2, 0.1])
    continue_prob = torch.tensor([0.6, 0.4])
    losses = ccva_router_loss(delta_hat, delta, continue_prob, target_continue=0.5)
    assert torch.isfinite(losses["loss_router"])
    assert losses["loss_budget"] >= 0
