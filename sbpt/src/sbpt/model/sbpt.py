"""SBPT model definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn

from sbpt.model.exits import ExitHead, pool_hidden
from sbpt.model.heads import BeliefHead, StateHead, VerifierHeads
from sbpt.model.router import Router


def _build_attn_mask(token_len: int, slot_len: int, device: torch.device) -> torch.Tensor:
    total = token_len + slot_len
    mask = torch.zeros((total, total), device=device, dtype=torch.bool)
    if token_len > 0:
        token_mask = torch.triu(torch.ones((token_len, token_len), device=device, dtype=torch.bool), diagonal=1)
        mask[:token_len, :token_len] = token_mask
    return mask


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


@dataclass
class SBPTConfig:
    vocab_size: int = 259
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    n_state_slots: int = 8
    n_state_ids: int = 3
    exit_layers: List[int] | None = None
    use_router: bool = True
    router_hidden: int = 128
    belief_vocab: int = 1024

    def resolved_exit_layers(self) -> List[int]:
        if self.exit_layers is not None:
            return self.exit_layers
        mid = max(1, self.n_layers // 2)
        return [mid, self.n_layers]


class SBPTModel(nn.Module):
    def __init__(self, config: SBPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len + config.n_state_slots, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.state_slots = nn.Parameter(torch.randn(config.n_state_slots, config.d_model) * 0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout) for _ in range(config.n_layers)]
        )

        exit_layers = config.resolved_exit_layers()
        self.exit_layers = exit_layers
        self.exit_heads = nn.ModuleDict()
        for idx, layer_id in enumerate(exit_layers):
            self.exit_heads[f"d{layer_id}"] = ExitHead(config.d_model, config.vocab_size)

        self.router = Router(config.d_model, hidden=config.router_hidden) if config.use_router else None
        self.state_head = StateHead(config.d_model, config.n_state_ids)
        self.belief_head = BeliefHead(config.d_model, vocab=config.belief_vocab)
        self.verifier_heads = VerifierHeads(config.d_model)

        anchor1 = max(1, config.n_layers // 3)
        anchor2 = max(anchor1 + 1, (2 * config.n_layers) // 3)
        anchor2 = min(anchor2, config.n_layers)
        self.state_anchors = [anchor1, anchor2, config.n_layers]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        device = input_ids.device
        batch_size, token_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, token_len), device=device, dtype=torch.bool)

        token_emb = self.token_emb(input_ids)
        slots = self.state_slots.unsqueeze(0).expand(batch_size, -1, -1)
        hidden = torch.cat([token_emb, slots], dim=1)
        total_len = hidden.shape[1]

        pos_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.drop(hidden + self.pos_emb(pos_ids))

        attn_mask = _build_attn_mask(token_len, self.config.n_state_slots, device=device)
        slot_mask = torch.ones((batch_size, self.config.n_state_slots), device=device, dtype=torch.bool)
        full_mask = torch.cat([attention_mask, slot_mask], dim=1)
        key_padding_mask = ~full_mask

        exits: Dict[str, torch.Tensor] = {}
        router_out: Dict[str, torch.Tensor] | None = None
        anchor_states: Dict[int, torch.Tensor] = {}
        pooled_token = None

        for layer_idx, block in enumerate(self.blocks, start=1):
            hidden = block(hidden, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            if layer_idx in self.state_anchors:
                anchor_states[layer_idx] = hidden[:, token_len:, :]
            if layer_idx in self.exit_layers:
                head = self.exit_heads[f"d{layer_idx}"]
                token_hidden = hidden[:, :token_len, :]
                exits[f"d{layer_idx}"] = head(token_hidden)
                if pooled_token is None:
                    pooled_token = pool_hidden(token_hidden, attention_mask)

        if pooled_token is None:
            pooled_token = pool_hidden(hidden[:, :token_len, :], attention_mask)

        if self.router is not None:
            router_out = self.router(pooled_token)

        slot_hidden_final = hidden[:, token_len:, :]
        aux: Dict[str, torch.Tensor] = {}
        if anchor_states:
            ordered_logits = [self.state_head(anchor_states[a]) for a in self.state_anchors if a in anchor_states]
            aux["state_logits"] = torch.stack(ordered_logits, dim=1)
        aux.update(self.verifier_heads(pooled_token))

        if metadata is not None and "hypotheses" in metadata:
            hypotheses = metadata.get("hypotheses", [])
            belief_logits = self.belief_head(pooled_token, hypotheses)
            aux["belief_logits"] = belief_logits

        return {
            "logits": exits[f"d{self.exit_layers[-1]}"] if exits else torch.empty(0),
            "state_slots": slot_hidden_final,
            "exits": exits,
            "router": router_out or {},
            "aux": aux,
        }
