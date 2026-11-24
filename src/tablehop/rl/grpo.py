"""GRPO baselines B2a/B2b: G samples per question, group-normalised trajectory return, same clipped
objective as PPO, no value head. Return = RewardFn.terminal(...) only (hop terms are not used)."""
from __future__ import annotations

import numpy as np

from tablehop.rl.rollout import Trajectory


def group_advantages(returns: np.ndarray, group: int, eps: float = 1e-6) -> np.ndarray:
    """returns (n_questions*group,) ordered by group -> (R - mean_g)/(std_g + eps). Equal rewards -> 0."""
    r = np.asarray(returns, np.float64).reshape(-1, group)
    return ((r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + eps)).reshape(-1)


def flatten_grpo(trajs: list[Trajectory], group: int) -> tuple[list[dict], np.ndarray]:
    """Every action of a trajectory gets that trajectory's group advantage."""
    adv = group_advantages(np.array([t.terminal for t in trajs]), group)
    steps, a = [], []
    for t, x in zip(trajs, adv):
        steps += t.steps
        a += [x] * len(t.steps)
    return steps, np.asarray(a)


def train(cfg: dict, seed: int):
    from tablehop.rl.runner import run

    return run(cfg, seed, "grpo")
