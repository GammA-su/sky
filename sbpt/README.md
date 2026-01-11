# SBPT — Stateful Belief-Program Transformer

Tool-free, bounded-inference research model with:
- latent state slots
- state transition supervision
- hypothesis belief collapse
- adaptive early-exit routing (counterfactual value)
- lightweight internal verification + calibration

## Quickstart

```bash
uv venv && uv sync
uv run pytest -q
uv run python scripts/01_make_synth.py --out data.jsonl --n 200
uv run python scripts/03_train_phase.py --config configs/train/phase1_state.yaml --data data.jsonl --steps 200 --cpu
uv run python scripts/04_eval_suite.py --ckpt /tmp/sbpt_ckpt.pt
```

See configs/ and scripts/ for end-to-end runs.
