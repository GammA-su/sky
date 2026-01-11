import torch

from sbpt.losses.belief_collapse import belief_collapse_loss
from sbpt.losses.lm import compute_lm_loss
from sbpt.losses.state_transition import compute_state_losses


def test_lm_loss_shape() -> None:
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[1, 2, -100, 3], [4, 5, 6, -100]])
    loss = compute_lm_loss(logits, labels)
    assert torch.isfinite(loss)


def test_state_transition_loss() -> None:
    state_logits = torch.randn(2, 3, 4)
    state_targets = torch.tensor([[0, 1, 2], [1, 2, 3]])
    adj = torch.eye(4, dtype=torch.bool)
    losses = compute_state_losses(state_logits, state_targets, adj)
    assert torch.isfinite(losses["loss_state"])
    assert torch.isfinite(losses["loss_trans"])


def test_belief_collapse_loss_decreases() -> None:
    logits_bad = torch.zeros(1, 4)
    logits_good = torch.tensor([[0.0, 0.0, 5.0, 0.0]])
    correct_mask = torch.tensor([[0, 0, 1, 0]], dtype=torch.bool)
    loss_bad = belief_collapse_loss(logits_bad, correct_mask, progress=1.0)
    loss_good = belief_collapse_loss(logits_good, correct_mask, progress=1.0)
    assert loss_good < loss_bad
