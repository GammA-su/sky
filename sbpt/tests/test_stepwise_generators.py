import json
import subprocess
import sys
from pathlib import Path


def _run_script(script_name: str, args: list[str], cwd: Path) -> None:
    cmd = [sys.executable, str(cwd / "scripts" / script_name)] + args
    subprocess.run(cmd, cwd=cwd, check=True)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_verify_stepwise_deterministic(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out1 = tmp_path / "verify_stepwise_1.jsonl"
    out2 = tmp_path / "verify_stepwise_2.jsonl"

    args = ["--out", str(out1), "--n", "4", "--seed", "7", "--min-digits", "2", "--max-digits", "3"]
    _run_script("17_make_verify_stepwise.py", args, repo_root)
    _run_script("17_make_verify_stepwise.py", ["--out", str(out2), "--n", "4", "--seed", "7", "--min-digits", "2", "--max-digits", "3"], repo_root)

    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
    rows = _load_jsonl(out1)
    assert rows
    for row in rows:
        assert row["verify_type"] == "stepwise_format"
        assert "prompt" in row and "completion" in row
        assert "verify_label" in row


def test_format_hard_stepwise_deterministic(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out1 = tmp_path / "format_stepwise_1.jsonl"
    out2 = tmp_path / "format_stepwise_2.jsonl"

    args = ["--out", str(out1), "--pairs", "2", "--seed", "11", "--min-digits", "2", "--max-digits", "3"]
    _run_script("18_make_format_hard_stepwise.py", args, repo_root)
    _run_script("18_make_format_hard_stepwise.py", ["--out", str(out2), "--pairs", "2", "--seed", "11", "--min-digits", "2", "--max-digits", "3"], repo_root)

    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
    rows = _load_jsonl(out1)
    assert rows
    for row in rows:
        assert row["verify_type"] == "stepwise_format"
        assert "task_id" in row
        assert "prompt" in row and "completion" in row


def test_robust_pairs_stepwise_deterministic(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out1 = tmp_path / "robust_stepwise_1.jsonl"
    out2 = tmp_path / "robust_stepwise_2.jsonl"

    args = ["--out", str(out1), "--pairs", "2", "--seed", "13", "--min-digits", "2", "--max-digits", "3"]
    _run_script("19_make_robust_pairs_stepwise.py", args, repo_root)
    _run_script("19_make_robust_pairs_stepwise.py", ["--out", str(out2), "--pairs", "2", "--seed", "13", "--min-digits", "2", "--max-digits", "3"], repo_root)

    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
    rows = _load_jsonl(out1)
    assert rows
    for row in rows:
        assert "equiv_id" in row
        assert "prompt" in row and "completion" in row
