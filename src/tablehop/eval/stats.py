"""Bootstrap statistics (spec 04)."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi): 95% percentile bootstrap CI of the mean. values: shape (N,)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap(a, b, n: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap of mean(a-b) over questions. p = min(1, 2*min(P(d<=0), P(d>=0)))."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = min(1.0, 2 * min(float((means <= 0).mean()), float((means >= 0).mean())))
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p": p}


def seed_summary(xs) -> tuple[float, float]:
    """(mean, sample std with ddof=1; 0.0 for a single seed)."""
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def gap(id_acc: float, ood_acc: float) -> tuple[float, float]:
    """(absolute, relative) ID->OOD gap; relative is 0.0 if id_acc == 0."""
    d = id_acc - ood_acc
    return d, (d / id_acc if id_acc else 0.0)
