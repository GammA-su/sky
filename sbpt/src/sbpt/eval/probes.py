"""SBPT-specific diagnostic probes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def state_accuracy(state_logits: torch.Tensor, state_targets: torch.Tensor) -> float:
    if state_logits.numel() == 0:
        return 0.0
    preds = state_logits.argmax(dim=-1)
    correct = (preds == state_targets).float()
    return float(correct.mean().item())


def belief_entropy(belief_logits: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if belief_logits.numel() == 0:
        return 0.0
    log_probs = F.log_softmax(belief_logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    if mask is not None:
        mask_f = mask.to(belief_logits.device).float()
        entropy = (entropy * (mask_f.sum(dim=-1) > 0).float())
    return float(entropy.mean().item())


def router_continue_prob(router_out: dict) -> float:
    if not router_out or "continue_prob" not in router_out:
        return 0.0
    return float(router_out["continue_prob"].mean().item())
