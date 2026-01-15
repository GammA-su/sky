"""Evaluation suite runner."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from sbpt.data.loaders import DataConfig, build_dataloader
from sbpt.data.schemas import ByteTokenizer
from sbpt.eval.metrics import (
    equiv_consistency,
    exact_match,
    expected_calibration_error,
    format_strict_pass,
    token_accuracy,
)
from sbpt.eval.probes import belief_entropy, router_continue_prob, state_accuracy
from sbpt.model.sbpt import SBPTConfig, SBPTModel
from sbpt.utils.io import load_yaml


def _decode_sequences(tokenizer: ByteTokenizer, ids: torch.Tensor) -> List[str]:
    texts = []
    for row in ids.tolist():
        texts.append(tokenizer.decode([i for i in row if i != -100]))
    return texts


def run_eval(ckpt_path: str, cfg_path: Optional[str] = None, cpu: bool = False) -> Dict[str, float | dict]:
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    tokenizer = ByteTokenizer()

    if cfg_path:
        cfg = load_yaml(cfg_path)
    else:
        cfg = {"eval": {"n": 200, "seed": 321, "min_digits": 2, "max_digits": 3}}

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

    eval_cfg = cfg.get("eval", {})
    datasets_cfg = eval_cfg.get("datasets")
    if datasets_cfg is None:
        datasets_cfg = {
            "in_dist": {
                "type": "synth",
                "n": int(eval_cfg.get("n", 200)),
                "seed": int(eval_cfg.get("seed", 321)),
                "min_digits": int(eval_cfg.get("min_digits", 2)),
                "max_digits": int(eval_cfg.get("max_digits", 3)),
            }
        }

    results: Dict[str, Dict[str, float]] = {}
    for name, ds_cfg in datasets_cfg.items():
        ds_type = ds_cfg.get("type", "synth")
        if ds_type == "jsonl":
            data_cfg = DataConfig(type="jsonl", path=str(ds_cfg.get("path", "")))
        else:
            data_cfg = DataConfig(
                type="synth",
                n=int(ds_cfg.get("n", 200)),
                seed=int(ds_cfg.get("seed", 321)),
                min_digits=int(ds_cfg.get("min_digits", 2)),
                max_digits=int(ds_cfg.get("max_digits", 3)),
            )

        loader = build_dataloader(
            data_cfg,
            tokenizer,
            batch_size=8,
            shuffle=False,
            max_len=model_cfg.max_seq_len,
        )

        all_pred_text: List[str] = []
        all_target_text: List[str] = []
        all_conf: List[float] = []
        all_correct: List[bool] = []
        token_accs: List[float] = []
        state_accs: List[float] = []
        belief_ents: List[float] = []
        router_probs: List[float] = []
        equiv_outputs: List[str] = []
        equiv_ids: List[int] = []
        verify_total = 0
        verify_correct = 0
        format_pass_total = 0

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

                equiv_batch = batch.get("equiv_id")
                if equiv_batch is not None:
                    for idx, equiv_id in enumerate(equiv_batch.tolist()):
                        if equiv_id >= 0:
                            equiv_outputs.append(pred_text[idx])
                            equiv_ids.append(int(equiv_id))

                verify_batch = batch.get("verify_label")
                if verify_batch is not None:
                    for idx, label in enumerate(verify_batch.tolist()):
                        if label < 0:
                            continue
                        pred_pass = format_strict_pass(pred_text[idx])
                        verify_total += 1
                        verify_correct += int(pred_pass == bool(label))
                        format_pass_total += int(pred_pass)

        metrics = {
            "exact_match": exact_match(all_pred_text, all_target_text),
            "token_accuracy": float(sum(token_accs) / max(len(token_accs), 1)),
            "ece": expected_calibration_error(all_conf, all_correct),
            "state_accuracy": float(sum(state_accs) / max(len(state_accs), 1)),
            "belief_entropy": float(sum(belief_ents) / max(len(belief_ents), 1)),
            "router_continue": float(sum(router_probs) / max(len(router_probs), 1)),
        }
        if equiv_ids:
            metrics["equiv_consistency"] = equiv_consistency(equiv_outputs, equiv_ids)
        if verify_total > 0:
            metrics["verify_accuracy"] = verify_correct / verify_total
            metrics["format_pass_rate"] = format_pass_total / verify_total

        results[name] = metrics

    summary: Dict[str, float] = {}
    for key in [
        "exact_match",
        "token_accuracy",
        "ece",
        "state_accuracy",
        "belief_entropy",
        "router_continue",
        "equiv_consistency",
        "verify_accuracy",
        "format_pass_rate",
    ]:
        vals = [m[key] for m in results.values() if key in m]
        if vals:
            summary[key] = float(sum(vals) / len(vals))

    return {"datasets": results, "summary": summary}
