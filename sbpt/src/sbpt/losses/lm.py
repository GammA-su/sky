"""Language modeling loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.tensor(0.0, device=labels.device)
    vocab = logits.shape[-1]
    loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), ignore_index=-100)
    return loss
