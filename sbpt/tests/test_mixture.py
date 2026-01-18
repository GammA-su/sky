import json
import subprocess
import sys
from pathlib import Path


def test_mixture_includes_stepwise(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    step_path = tmp_path / "stepwise.jsonl"
    out_path = tmp_path / "mix.jsonl"
    row = {
        "prompt": "Add 12 and 3. Reply with digits separated by spaces.",
        "completion": "1 5",
        "task_type": "add_carry_stepwise",
        "state_spans": [[0, 1, 0], [2, 3, 1]],
    }
    step_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "07_build_mixture.py"),
        "--out",
        str(out_path),
        "--n",
        "1",
        "--seed",
        "123",
        "--src",
        f"carry_step={step_path}:1",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    parsed = json.loads(lines[0])
    assert parsed.get("task_type") == "add_carry_stepwise"
