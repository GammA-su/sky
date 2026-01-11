"""State transition supervision loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_state_losses(
    state_logits: torch.Tensor,
    state_targets: torch.Tensor,
    adjacency: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    state_logits: (B, A, Z)
    state_targets: (B, A)
    adjacency: (Z, Z) bool where True indicates allowed transition
    """
    batch, anchors, n_states = state_logits.shape
    loss_state = F.cross_entropy(
        state_logits.view(batch * anchors, n_states),
        state_targets.view(-1),
    )
    loss_trans = torch.tensor(0.0, device=state_logits.device)
    if adjacency is not None:
        probs = torch.softmax(state_logits, dim=-1)
        invalid = (~adjacency.to(torch.bool)).float()
        for i in range(anchors - 1):
            p_i = probs[:, i, :]
            p_j = probs[:, i + 1, :]
            invalid_mass = torch.einsum("bi,ij,bj->b", p_i, invalid, p_j)
            loss_trans = loss_trans + invalid_mass.mean()
        loss_trans = loss_trans / max(anchors - 1, 1)
    return {"loss_state": loss_state, "loss_trans": loss_trans}
