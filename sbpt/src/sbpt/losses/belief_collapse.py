"""Belief collapse loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def belief_collapse_loss(
    belief_logits: torch.Tensor,
    correct_mask: torch.Tensor,
    progress: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if belief_logits.numel() == 0:
        return torch.tensor(0.0, device=belief_logits.device)
    progress = float(max(0.0, min(1.0, progress)))
    log_probs = F.log_softmax(belief_logits, dim=-1)
    probs = log_probs.exp()

    if mask is None:
        mask = torch.ones_like(belief_logits, dtype=torch.bool)

    mask_f = mask.float()
    uniform = mask_f / mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)

    correct = correct_mask & mask
    correct_f = correct.float()
    correct_norm = correct_f / correct_f.sum(dim=-1, keepdim=True).clamp_min(1.0)

    target = (1.0 - progress) * uniform + progress * correct_norm
    ce = -(target * log_probs).sum(dim=-1)

    entropy = -(probs * log_probs).sum(dim=-1)
    loss = ce.mean() + progress * entropy.mean()
    return loss
