"""Global seeding."""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed random, numpy and torch (cpu, cuda, mps)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
