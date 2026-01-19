"""Phase definitions for training."""

from __future__ import annotations

PHASE_DEFAULTS = {
    "phase0_lm": {"lm": 1.0, "bridge_stepwise_lm": 0.3},
    "phase1_state": {"lm": 1.0, "state": 1.0, "transition": 0.1, "bridge_stepwise_lm": 0.3},
    "phase2_belief": {"lm": 1.0, "belief": 0.5, "bridge_stepwise_lm": 0.3},
    "phase3_verify": {"lm": 1.0, "verify": 0.1, "calib": 0.1, "bridge_stepwise_lm": 0.3},
    "phase4_robust": {"lm": 1.0, "robust": 0.1, "bridge_stepwise_lm": 0.3},
}


def get_phase_defaults(phase: str) -> dict[str, float]:
    return dict(PHASE_DEFAULTS.get(phase, {}))
