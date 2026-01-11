"""Auxiliary heads for SBPT."""

from __future__ import annotations

import hashlib
from typing import List

import torch
from torch import nn


def _hash_to_id(text: str, vocab: int) -> int:
    digest = hashlib.md5(text.encode("utf-8", errors="replace")).digest()
    value = int.from_bytes(digest[:4], "little")
    return value % vocab


class StateHead(nn.Module):
    def __init__(self, d_model: int, n_states: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, n_states)

    def forward(self, state_slots: torch.Tensor) -> torch.Tensor:
        pooled = state_slots.mean(dim=1)
        return self.proj(pooled)


class BeliefHead(nn.Module):
    def __init__(self, d_model: int, vocab: int = 1024) -> None:
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self, pooled_hidden: torch.Tensor, hypotheses: List[List[str]]
    ) -> torch.Tensor:
        device = pooled_hidden.device
        max_h = max((len(h) for h in hypotheses), default=0)
        if max_h == 0:
            return torch.empty((pooled_hidden.shape[0], 0), device=device)
        logits = torch.full((pooled_hidden.shape[0], max_h), -1e9, device=device)
        proj_hidden = self.proj(pooled_hidden)
        for i, h_list in enumerate(hypotheses):
            if not h_list:
                continue
            ids = torch.tensor([_hash_to_id(h, self.vocab) for h in h_list], device=device)
            h_emb = self.emb(ids)
            scores = (proj_hidden[i].unsqueeze(0) * h_emb).sum(dim=-1)
            logits[i, : len(h_list)] = scores
        return logits


class VerifierHeads(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.constraint = nn.Linear(d_model, 1)
        self.consistency = nn.Linear(d_model, 1)
        self.calibration = nn.Linear(d_model, 1)

    def forward(self, pooled_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "constraint_logit": self.constraint(pooled_hidden).squeeze(-1),
            "consistency_logit": self.consistency(pooled_hidden).squeeze(-1),
            "calibration_logit": self.calibration(pooled_hidden).squeeze(-1),
        }
