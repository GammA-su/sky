"""Training loop for SBPT."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch

from sbpt.data.loaders import DataConfig, build_dataloader
from sbpt.data.schemas import ByteTokenizer
from sbpt.losses.belief_collapse import belief_collapse_loss
from sbpt.losses.ccva_router import ccva_router_loss
from sbpt.losses.lm import compute_lm_loss
from sbpt.losses.state_transition import compute_state_losses
from sbpt.model.sbpt import SBPTConfig, SBPTModel
from sbpt.train.ckpt import save_ckpt
from sbpt.train.logging import Logger
from sbpt.train.optim import build_optimizer, build_scheduler
from sbpt.train.phases import get_phase_defaults
from sbpt.utils.seed import set_seed


def _cycle(loader: Iterable[dict]) -> Iterable[dict]:
    while True:
        for batch in loader:
            yield batch


def _build_adjacency(n_states: int) -> torch.Tensor:
    adj = torch.zeros((n_states, n_states), dtype=torch.bool)
    for i in range(n_states):
        adj[i, i] = True
    for i in range(n_states - 1):
        adj[i, i + 1] = True
    return adj


def train(
    cfg: Dict[str, Any],
    data_path: Optional[str] = None,
    steps_override: Optional[int] = None,
    cpu: bool = False,
) -> Dict[str, float]:
    train_cfg = cfg.get("train", {})
    phase = str(train_cfg.get("phase", "phase0_lm"))
    weights = cfg.get("loss_weights", {})
    if not weights:
        weights = get_phase_defaults(phase)

    seed = int(train_cfg.get("seed", 123))
    set_seed(seed)

    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    tokenizer = ByteTokenizer()

    data_dict = cfg.get("data", {})
    data_cfg = DataConfig(**data_dict)
    if data_path:
        data_cfg.type = "jsonl"
        data_cfg.path = data_path

    batch_size = int(train_cfg.get("batch_size", 8))
    max_len = cfg.get("eval", {}).get("max_len")
    loader = build_dataloader(data_cfg, tokenizer, batch_size=batch_size, max_len=max_len)

    model_cfg_path = cfg.get("model")
    if isinstance(model_cfg_path, str):
        from sbpt.utils.io import load_yaml

        model_dict = load_yaml(model_cfg_path)["model"]
    else:
        model_dict = cfg.get("model", {})
    model_cfg = SBPTConfig(**model_dict)
    model = SBPTModel(model_cfg).to(device)

    optimizer = build_optimizer(model, lr=float(train_cfg.get("lr", 1e-3)), weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    steps = int(steps_override or train_cfg.get("steps", 200))
    scheduler = build_scheduler(optimizer, total_steps=steps)

    logger = Logger(train_cfg.get("log_path"))
    iterator = _cycle(loader)
    adjacency = _build_adjacency(model_cfg.n_state_ids).to(device)

    model.train()
    last_loss = 0.0
    for step in range(1, steps + 1):
        batch = next(iterator)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        state_ids = batch["state_ids"].to(device)
        metadata = {"hypotheses": batch.get("hypotheses", [])}

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, metadata=metadata)
        logits = outputs["logits"]
        aux = outputs.get("aux", {})
        exits = outputs.get("exits", {})
        router_out = outputs.get("router", {})

        loss_total = torch.tensor(0.0, device=device)
        loss_items: Dict[str, float] = {}

        if weights.get("lm", 0.0) > 0:
            lm_loss = compute_lm_loss(logits, labels)
            loss_total = loss_total + float(weights["lm"]) * lm_loss
            loss_items["lm"] = float(lm_loss.detach().cpu())

        if weights.get("state", 0.0) > 0 and "state_logits" in aux:
            state_losses = compute_state_losses(aux["state_logits"], state_ids, adjacency)
            loss_total = loss_total + float(weights.get("state", 1.0)) * state_losses["loss_state"]
            loss_total = loss_total + float(weights.get("transition", 0.0)) * state_losses["loss_trans"]
            loss_items["state"] = float(state_losses["loss_state"].detach().cpu())
            loss_items["trans"] = float(state_losses["loss_trans"].detach().cpu())

        if weights.get("belief", 0.0) > 0 and "belief_logits" in aux:
            correct_mask = batch.get("correct_mask")
            hypothesis_mask = batch.get("hypothesis_mask")
            if correct_mask is not None:
                belief_loss = belief_collapse_loss(
                    aux["belief_logits"],
                    correct_mask.to(device),
                    progress=step / max(steps, 1),
                    mask=hypothesis_mask.to(device) if hypothesis_mask is not None else None,
                )
                loss_total = loss_total + float(weights["belief"]) * belief_loss
                loss_items["belief"] = float(belief_loss.detach().cpu())

        if weights.get("router", 0.0) > 0 and router_out:
            exit_keys = sorted(exits.keys(), key=lambda k: int(k[1:]))
            if len(exit_keys) >= 2:
                shallow = exits[exit_keys[0]]
                deep = exits[exit_keys[-1]]
                l_shallow = compute_lm_loss(shallow, labels)
                l_deep = compute_lm_loss(deep, labels)
                delta = (l_shallow - l_deep).detach()
                router_losses = ccva_router_loss(
                    router_out["delta_hat"],
                    delta,
                    router_out["continue_prob"],
                    target_continue=0.5,
                )
                loss_total = loss_total + router_losses["loss_router"] * float(weights["router"])
                loss_items["router"] = float(router_losses["loss_router"].detach().cpu())

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        last_loss = float(loss_total.detach().cpu())
        if step % int(train_cfg.get("log_every", 10)) == 0:
            logger.log({"step": step, "loss": last_loss, **loss_items})

    save_path = str(train_cfg.get("save_path", "/tmp/sbpt_ckpt.pt"))
    save_ckpt(save_path, model, optimizer, cfg, steps)
    logger.close()
    return {"loss": last_loss}
