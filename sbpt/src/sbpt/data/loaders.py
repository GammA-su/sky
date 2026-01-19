"""Dataset loading and collation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from sbpt.data.enumerators import enumerate_hypotheses
from sbpt.data.schemas import ByteTokenizer
from sbpt.data.synth_traces import generate_addition_samples
from sbpt.utils.io import load_jsonl
from sbpt.utils.seed import seed_worker


_STATE_ANCHORS = 3


@dataclass
class DataConfig:
    type: str = "synth"
    n: int = 500
    seed: int = 0
    min_digits: int = 2
    max_digits: int = 3
    path: Optional[str] = None


class RowsDataset(Dataset):
    def __init__(self, rows: List[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def _maybe_add_hypotheses(row: dict, max_h: int = 8) -> dict:
    if "hypotheses" in row and "correct_mask" in row:
        return row
    hypotheses, correct = enumerate_hypotheses(row, max_h=max_h)
    if hypotheses:
        row = dict(row)
        row["hypotheses"] = hypotheses
        row["correct_mask"] = correct
    return row


def _build_rows_from_config(cfg: DataConfig) -> list[dict]:
    if cfg.type == "synth":
        rows = generate_addition_samples(
            n=cfg.n,
            seed=cfg.seed,
            min_digits=cfg.min_digits,
            max_digits=cfg.max_digits,
        )
        return rows
    if cfg.type == "jsonl" and cfg.path:
        return load_jsonl(cfg.path)
    return []


def _select_state_anchors(state_ids: Sequence[int], anchors: int = _STATE_ANCHORS) -> List[int]:
    if anchors <= 0:
        return []
    if not state_ids:
        return [0] * anchors
    if len(state_ids) == anchors:
        return list(state_ids)
    if len(state_ids) < anchors:
        pad = state_ids[-1]
        return list(state_ids) + [pad] * (anchors - len(state_ids))
    if anchors == 1:
        return [state_ids[-1]]
    last_idx = len(state_ids) - 1
    indices = [(i * last_idx) // (anchors - 1) for i in range(anchors)]
    return [state_ids[i] for i in indices]


def expand_state_spans(row: dict, completion: str, tokenizer: ByteTokenizer) -> Tuple[Optional[List[int]], int]:
    spans = row.get("state_spans")
    if not spans or not completion:
        return None, 0
    normalized_spans: list[Tuple[int, int, int]] = []
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 3:
            continue
        try:
            start = int(span[0])
            end = int(span[1])
            state = int(span[2])
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start or start >= len(completion):
            continue
        end = min(end, len(completion))
        normalized_spans.append((start, end, state))
    if not normalized_spans:
        return None, 0
    normalized_spans.sort(key=lambda s: s[0])
    token_states: List[int] = []
    supervised = 0
    for start, end, state in normalized_spans:
        chunk = completion[start:end]
        if not chunk:
            continue
        tokens = tokenizer.encode(chunk, add_bos=False, add_eos=False)
        if not tokens:
            continue
        token_states.extend([state] * len(tokens))
        supervised += len(tokens)
    if not token_states:
        return None, 0
    return token_states, supervised


def collate_batch(
    batch: List[dict],
    tokenizer: ByteTokenizer,
    max_len: Optional[int] = None,
) -> dict:
    input_ids_list: List[List[int]] = []
    labels_list: List[List[int]] = []
    attention_list: List[List[int]] = []
    state_ids_list: List[List[int]] = []
    stepwise_input_ids_list: List[List[int]] = []
    stepwise_labels_list: List[List[int]] = []
    stepwise_attention_list: List[List[int]] = []
    has_stepwise_completion = False
    state_spans_list: List[list] = []

    hypotheses_list: List[List[str]] = []
    correct_list: List[List[bool]] = []
    equiv_ids: List[int] = []
    verify_labels: List[int] = []
    verify_types: List[str] = []
    task_types: List[str] = []

    state_supervised_tokens = 0
    num_carry_examples = 0

    for row in batch:
        row = _maybe_add_hypotheses(row)
        prompt = str(row.get("prompt", ""))
        completion = str(row.get("completion", ""))
        prompt_ids = tokenizer.encode(prompt, add_bos=False, add_eos=False)
        completion_ids = tokenizer.encode(completion, add_bos=False, add_eos=False)
        input_ids = [tokenizer.bos_id] + prompt_ids + completion_ids + [tokenizer.eos_id]

        if max_len is not None and len(input_ids) > max_len:
            input_ids = input_ids[:max_len]

        labels = [-100] * len(input_ids)
        prompt_len = 1 + len(prompt_ids)
        start = max(prompt_len - 1, 0)
        for i in range(start, len(input_ids) - 1):
            labels[i] = input_ids[i + 1]

        attention = [1] * len(input_ids)
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_list.append(attention)

        stepwise_text = row.get("stepwise_completion", None)
        stepwise_completion = str(stepwise_text) if stepwise_text is not None else ""
        if stepwise_completion:
            has_stepwise_completion = True
        stepwise_ids = tokenizer.encode(stepwise_completion, add_bos=False, add_eos=False)
        stepwise_input_ids = [tokenizer.bos_id] + prompt_ids + stepwise_ids + [tokenizer.eos_id]
        if max_len is not None and len(stepwise_input_ids) > max_len:
            stepwise_input_ids = stepwise_input_ids[:max_len]
        stepwise_labels = [-100] * len(stepwise_input_ids)
        if stepwise_completion:
            prompt_len = 1 + len(prompt_ids)
            start = max(prompt_len - 1, 0)
            for i in range(start, len(stepwise_input_ids) - 1):
                stepwise_labels[i] = stepwise_input_ids[i + 1]
        stepwise_attention = [1] * len(stepwise_input_ids)
        stepwise_input_ids_list.append(stepwise_input_ids)
        stepwise_labels_list.append(stepwise_labels)
        stepwise_attention_list.append(stepwise_attention)

        state_completion = completion
        if stepwise_completion:
            state_completion = stepwise_completion
        state_tokens, supervised = expand_state_spans(row, state_completion, tokenizer)
        if state_tokens is not None:
            state_supervised_tokens += supervised
            if supervised > 0:
                num_carry_examples += 1
            final_state_ids = state_tokens
        else:
            final_state_ids = list(row.get("state_ids", [0, 1, 2]))

        state_ids = _select_state_anchors(final_state_ids)
        state_ids_list.append(state_ids)

        hypotheses = list(row.get("hypotheses", []))
        correct = list(row.get("correct_mask", []))
        hypotheses_list.append(hypotheses)
        correct_list.append(correct)

        equiv_id = row.get("equiv_id", None)
        equiv_ids.append(int(equiv_id) if equiv_id is not None else -1)
        verify_label = row.get("verify_label", None)
        verify_labels.append(int(verify_label) if verify_label is not None else -1)
        verify_type = row.get("verify_type", None)
        verify_types.append(str(verify_type) if verify_type is not None else "")

        task_type = row.get("task_type", None)
        task_types.append(str(task_type) if task_type is not None else "")
        state_spans_list.append(list(row.get("state_spans", [])) if "state_spans" in row else [])

    max_len_batch = max(len(ids) for ids in input_ids_list)
    padded_inputs = []
    padded_labels = []
    padded_attention = []
    for ids, labels, attn in zip(input_ids_list, labels_list, attention_list):
        pad = max_len_batch - len(ids)
        padded_inputs.append(ids + [tokenizer.pad_id] * pad)
        padded_labels.append(labels + [-100] * pad)
        padded_attention.append(attn + [0] * pad)

    input_ids_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    labels_tensor = torch.tensor(padded_labels, dtype=torch.long)
    attention_tensor = torch.tensor(padded_attention, dtype=torch.bool)
    state_ids_tensor = torch.tensor(state_ids_list, dtype=torch.long)
    equiv_ids_tensor = torch.tensor(equiv_ids, dtype=torch.long)
    verify_labels_tensor = torch.tensor(verify_labels, dtype=torch.long)

    if has_stepwise_completion:
        max_step_len = max(len(ids) for ids in stepwise_input_ids_list)
        padded_stepwise_inputs = []
        padded_stepwise_labels = []
        padded_stepwise_attention = []
        for ids, labels, attn in zip(stepwise_input_ids_list, stepwise_labels_list, stepwise_attention_list):
            pad = max_step_len - len(ids)
            padded_stepwise_inputs.append(ids + [tokenizer.pad_id] * pad)
            padded_stepwise_labels.append(labels + [-100] * pad)
            padded_stepwise_attention.append(attn + [0] * pad)
        stepwise_input_ids_tensor = torch.tensor(padded_stepwise_inputs, dtype=torch.long)
        stepwise_labels_tensor = torch.tensor(padded_stepwise_labels, dtype=torch.long)
        stepwise_attention_tensor = torch.tensor(padded_stepwise_attention, dtype=torch.bool)
    else:
        stepwise_input_ids_tensor = None
        stepwise_labels_tensor = None
        stepwise_attention_tensor = None

    max_h = max((len(h) for h in hypotheses_list), default=0)
    if max_h > 0:
        correct_mask = torch.zeros((len(batch), max_h), dtype=torch.bool)
        hypothesis_mask = torch.zeros((len(batch), max_h), dtype=torch.bool)
        for i, (h_list, c_list) in enumerate(zip(hypotheses_list, correct_list)):
            if not h_list:
                continue
            hypothesis_mask[i, : len(h_list)] = True
            correct_mask[i, : len(c_list)] = torch.tensor(c_list, dtype=torch.bool)
    else:
        correct_mask = None
        hypothesis_mask = None

    batch_out = {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "attention_mask": attention_tensor,
        "state_ids": state_ids_tensor,
        "hypotheses": hypotheses_list,
        "correct_mask": correct_mask,
        "hypothesis_mask": hypothesis_mask,
        "equiv_id": equiv_ids_tensor,
        "verify_label": verify_labels_tensor,
        "verify_type": verify_types,
        "task_type": task_types,
        "state_supervised_tokens": state_supervised_tokens,
        "num_carry_examples": num_carry_examples,
        "has_stepwise_completion": has_stepwise_completion,
        "state_spans": state_spans_list,
    }
    if has_stepwise_completion:
        batch_out.update(
            {
                "stepwise_input_ids": stepwise_input_ids_tensor,
                "stepwise_labels": stepwise_labels_tensor,
                "stepwise_attention_mask": stepwise_attention_tensor,
            }
        )
    return batch_out


def build_dataloader(
    cfg: DataConfig,
    tokenizer: ByteTokenizer,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    max_len: Optional[int] = None,
) -> DataLoader:
    rows = _build_rows_from_config(cfg)
    dataset = RowsDataset(rows)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_batch(b, tokenizer, max_len=max_len),
        worker_init_fn=lambda worker_id: seed_worker(worker_id, cfg.seed),
        generator=generator,
    )
