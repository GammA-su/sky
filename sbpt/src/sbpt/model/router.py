"""Router for adaptive compute."""

from __future__ import annotations

import torch
from torch import nn


class Router(nn.Module):
    def __init__(self, d_model: int, hidden: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        delta_hat = self.mlp(features).squeeze(-1)
        continue_prob = torch.sigmoid(delta_hat)
        return {"delta_hat": delta_hat, "continue_prob": continue_prob}
