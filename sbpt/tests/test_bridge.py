import json
import subprocess
import sys
from pathlib import Path

from sbpt.data.schemas import ByteTokenizer
from sbpt.train.loop import train


def test_bridge_script_outputs_fields(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "bridge.jsonl"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "20_make_train_add_carry_bridge.py"),
        "--out",
        str(out_path),
        "--n",
        "10",
        "--seed",
        "7",
        "--min-digits",
        "2",
        "--max-digits",
        "3",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 10
    for row in rows:
        assert row["task_type"] == "add_carry_bridge"
        assert "completion" in row
        assert "stepwise_completion" in row
        assert "state_spans" in row
        assert row["completion"].startswith(" ")


def test_bridge_stepwise_aux_loss_runs(tmp_path: Path) -> None:
    row = {
        "prompt": "Add 12 and 3. Reply with only the final sum.",
        "completion": " 15",
        "stepwise_completion": "1 5",
        "task_type": "add_carry_bridge",
    }
    data_path = tmp_path / "bridge_data.jsonl"
    data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    tokenizer = ByteTokenizer()
    cfg = {
        "model": {
            "vocab_size": tokenizer.vocab_size,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 64,
            "max_seq_len": 64,
            "dropout": 0.0,
            "n_state_slots": 2,
            "n_state_ids": 4,
            "use_router": False,
        },
        "data": {"type": "jsonl", "path": str(data_path)},
        "train": {
            "phase": "phase1_state",
            "steps": 1,
            "batch_size": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "seed": 1,
            "log_every": 1,
            "save_path": str(tmp_path / "ckpt.pt"),
        },
        "loss_weights": {"lm": 1.0, "state": 0.0, "trans": 0.0, "bridge_stepwise_lm": 0.3},
    }

    result = train(cfg, steps_override=1, cpu=True)
    assert "loss" in result
