import subprocess
import sys
from pathlib import Path

import torch

from sbpt.data.loaders import DataConfig, build_dataloader
from sbpt.data.schemas import ByteTokenizer
from sbpt.eval.probes import state_accuracy


def test_stepwise_format_state_targets_present(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sbpt_format_hard_stepwise.jsonl"

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "18_make_format_hard_stepwise.py"),
        "--out",
        str(out_path),
        "--pairs",
        "5",
        "--seed",
        "3",
        "--min-digits",
        "2",
        "--max-digits",
        "3",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    tokenizer = ByteTokenizer()
    cfg = DataConfig(type="jsonl", path=str(out_path))
    loader = build_dataloader(cfg, tokenizer, batch_size=5, shuffle=False, max_len=128)
    batch = next(iter(loader))

    assert "state_spans" in batch
    assert "state_ids" in batch
    assert any(batch["state_spans"])
    assert batch["state_ids"].numel() > 0
    assert torch.unique(batch["state_ids"]).numel() > 1

    logits = torch.zeros(
        batch["state_ids"].shape[0],
        batch["state_ids"].shape[1],
        int(batch["state_ids"].max().item()) + 1,
    )
    acc = state_accuracy(logits, batch["state_ids"])
    assert isinstance(acc, float)
