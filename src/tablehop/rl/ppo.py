"""PPO for Hop-PPO (H1, A1-A4): GAE, clipped objective, per-update logging (specs/02 "PPO")."""
from __future__ import annotations

import numpy as np
import torch

from tablehop.models.policy import Policy, collate, run_policy
from tablehop.rl.rollout import Trajectory


def gae(rewards: list[float], values: list[float], gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE for one finished episode (V(s_T)=0). Returns (advantages, returns=adv+values)."""
    n = len(rewards)
    adv = np.zeros(n, np.float64)
    last = 0.0
    for t in reversed(range(n)):
        nv = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * nv - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + np.asarray(values)


def flatten_ppo(trajs: list[Trajectory], gamma: float, lam: float) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Steps, advantages and returns for a batch; terminal reward is added to each trajectory's last step."""
    steps, advs, rets = [], [], []
    for t in trajs:
        r = [s["reward"] for s in t.steps]
        r[-1] += t.terminal
        a, g = gae(r, [s["value"] for s in t.steps], gamma, lam)
        steps += t.steps
        advs.append(a)
        rets.append(g)
    return steps, np.concatenate(advs), np.concatenate(rets)


def clipped_loss(policy: Policy, batch: dict, old_logp, adv, ret, clip_eps: float, value_coef: float,
                 ent_coef: float):
    """PPO clipped objective on one minibatch -> (loss, stats). value_coef=0 -> no value term (GRPO)."""
    logits, value = run_policy(policy, batch)
    logp_all = torch.log_softmax(logits, -1)
    logp = logp_all.gather(1, batch["action"][:, None]).squeeze(1)
    ratio = torch.exp(logp - old_logp)
    pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()
    vloss = ((value - ret) ** 2).mean() if ret is not None else torch.zeros(())
    ent = -(logp_all.exp() * logp_all).sum(-1).mean()
    loss = pg + value_coef * vloss - ent_coef * ent
    stats = {"ratio_mean": ratio.mean().item(), "approx_kl": (old_logp - logp).mean().item(),
             "clip_frac": ((ratio - 1).abs() > clip_eps).float().mean().item(),
             "value_loss": vloss.item(), "entropy": ent.item(), "pg_loss": pg.item()}
    return loss, stats


def update(policy: Policy, opt, steps: list[dict], adv: np.ndarray, ret: np.ndarray | None, pcfg: dict,
           ent_coef: float, use_value: bool, rng: np.random.Generator) -> dict:
    """`epochs` passes of shuffled minibatches over all steps. Advantages are normalised per batch."""
    adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv - adv.mean()
    policy.train()
    acc: dict[str, list[float]] = {}
    n, mb = len(steps), int(pcfg["minibatch"])
    for _ in range(int(pcfg["epochs"])):
        perm = rng.permutation(n)
        for s in range(0, n, mb):
            idx = perm[s:s + mb]
            batch = collate([steps[i] for i in idx])
            batch["action"] = torch.tensor([steps[i]["action"] for i in idx])
            old = torch.tensor([steps[i]["logp"] for i in idx], dtype=torch.float32)
            a = torch.tensor(adv[idx], dtype=torch.float32)
            r = torch.tensor(ret[idx], dtype=torch.float32) if use_value else None
            loss, st = clipped_loss(policy, batch, old, a, r, float(pcfg["clip_eps"]),
                                    float(pcfg["value_coef"]) if use_value else 0.0, ent_coef)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(pcfg["grad_clip"]))
            opt.step()
            for k, v in st.items():
                acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    var = returns.var()
    return float("nan") if var < 1e-12 else float(1 - (returns - values).var() / var)


def train(cfg: dict, seed: int):
    from tablehop.rl.runner import run

    return run(cfg, seed, "ppo")
