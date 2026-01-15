"""Seed helpers for determinism."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int, base_seed: Optional[int] = None) -> None:
    seed = (base_seed or 0) + worker_id
    set_seed(seed)
