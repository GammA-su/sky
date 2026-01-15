"""Dataset loading and collation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from sbpt.data.enumerators import enumerate_hypotheses
from sbpt.data.schemas import ByteTokenizer
from sbpt.data.synth_traces import generate_addition_samples
from sbpt.utils.io import load_jsonl
from sbpt.utils.seed import seed_worker


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


def collate_batch(
    batch: List[dict],
    tokenizer: ByteTokenizer,
    max_len: Optional[int] = None,
) -> dict:
    input_ids_list: List[List[int]] = []
    labels_list: List[List[int]] = []
    attention_list: List[List[int]] = []
    state_ids_list: List[List[int]] = []

    hypotheses_list: List[List[str]] = []
    correct_list: List[List[bool]] = []
    equiv_ids: List[int] = []
    verify_labels: List[int] = []

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

        state_ids = row.get("state_ids", [0, 1, 2])
        state_ids_list.append(list(state_ids))

        hypotheses = list(row.get("hypotheses", []))
        correct = list(row.get("correct_mask", []))
        hypotheses_list.append(hypotheses)
        correct_list.append(correct)

        equiv_id = row.get("equiv_id", None)
        equiv_ids.append(int(equiv_id) if equiv_id is not None else -1)
        verify_label = row.get("verify_label", None)
        verify_labels.append(int(verify_label) if verify_label is not None else -1)

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

    return {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "attention_mask": attention_tensor,
        "state_ids": state_ids_tensor,
        "hypotheses": hypotheses_list,
        "correct_mask": correct_mask,
        "hypothesis_mask": hypothesis_mask,
        "equiv_id": equiv_ids_tensor,
        "verify_label": verify_labels_tensor,
    }


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
