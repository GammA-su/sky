"""Evaluation suite runner."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from sbpt.data.loaders import DataConfig, build_dataloader
from sbpt.data.schemas import ByteTokenizer
from sbpt.eval.metrics import exact_match, token_accuracy, expected_calibration_error
from sbpt.eval.probes import belief_entropy, router_continue_prob, state_accuracy
from sbpt.model.sbpt import SBPTConfig, SBPTModel
from sbpt.utils.io import load_yaml


def _decode_sequences(tokenizer: ByteTokenizer, ids: torch.Tensor) -> List[str]:
    texts = []
    for row in ids.tolist():
        texts.append(tokenizer.decode([i for i in row if i != -100]))
    return texts


def run_eval(ckpt_path: str, cfg_path: Optional[str] = None, cpu: bool = False) -> Dict[str, float]:
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    tokenizer = ByteTokenizer()

    if cfg_path:
        cfg = load_yaml(cfg_path)
    else:
        cfg = {"eval": {"n": 200, "seed": 321, "min_digits": 2, "max_digits": 3}}

    eval_cfg = cfg.get("eval", {})
    data_cfg = DataConfig(
        type="synth",
        n=int(eval_cfg.get("n", 200)),
        seed=int(eval_cfg.get("seed", 321)),
        min_digits=int(eval_cfg.get("min_digits", 2)),
        max_digits=int(eval_cfg.get("max_digits", 3)),
    )

    loader = build_dataloader(data_cfg, tokenizer, batch_size=8, shuffle=False)

    state = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg = state.get("config", {})
    model_cfg_source = ckpt_cfg.get("model")
    if isinstance(model_cfg_source, str):
        model_dict = load_yaml(model_cfg_source).get("model", {})
    elif isinstance(model_cfg_source, dict):
        model_dict = model_cfg_source
    else:
        model_dict = {}
    model_dict.setdefault("vocab_size", tokenizer.vocab_size)
    model_cfg = SBPTConfig(**model_dict)
    model = SBPTModel(model_cfg)
    model.load_state_dict(state["model"])
    model = model.to(device)
    model.eval()

    all_pred_text: List[str] = []
    all_target_text: List[str] = []
    all_conf: List[float] = []
    all_correct: List[bool] = []
    token_accs: List[float] = []
    state_accs: List[float] = []
    belief_ents: List[float] = []
    router_probs: List[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = {"hypotheses": batch.get("hypotheses", [])}
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, metadata=metadata)
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1)
            pred_text = _decode_sequences(tokenizer, preds.masked_fill(labels == -100, -100).cpu())
            target_text = _decode_sequences(tokenizer, labels.cpu())

            all_pred_text.extend(pred_text)
            all_target_text.extend(target_text)

            token_accs.append(token_accuracy(logits.cpu(), labels.cpu()))

            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values
            mask = labels != -100
            conf = (max_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
            correct = (preds == labels) | (labels == -100)
            correct = (correct & mask).float().sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
            all_conf.extend(conf.cpu().tolist())
            all_correct.extend((correct.cpu() > 0.99).tolist())

            if "state_logits" in outputs.get("aux", {}):
                state_accs.append(state_accuracy(outputs["aux"]["state_logits"], batch["state_ids"].to(device)))
            if "belief_logits" in outputs.get("aux", {}):
                belief_ents.append(belief_entropy(outputs["aux"]["belief_logits"], batch.get("hypothesis_mask")))
            router_probs.append(router_continue_prob(outputs.get("router", {})))

    metrics = {
        "exact_match": exact_match(all_pred_text, all_target_text),
        "token_accuracy": float(sum(token_accs) / max(len(token_accs), 1)),
        "ece": expected_calibration_error(all_conf, all_correct),
        "state_accuracy": float(sum(state_accs) / max(len(state_accs), 1)),
        "belief_entropy": float(sum(belief_ents) / max(len(belief_ents), 1)),
        "router_continue": float(sum(router_probs) / max(len(router_probs), 1)),
    }
    return metrics
