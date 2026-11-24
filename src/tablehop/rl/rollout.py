"""Batched rollouts over many envs (shared by PPO and GRPO). Answer head called once per batch of episodes.

Step dict: q, trace, cands, rel, mask (numpy), action, logp, value, reward. Reward of a step = hop reward;
the terminal reward stays in `Trajectory.terminal`; learners add it to the last step's reward.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from tablehop.data.schema import Example
from tablehop.env.render import render_evidence
from tablehop.env.table_env import TableEnv
from tablehop.models.policy import Policy, collate, run_policy
from tablehop.rl.reward import RewardFn


@dataclass
class Trajectory:
    ex: Example
    steps: list[dict] = field(default_factory=list)
    infos: list = field(default_factory=list)  # HopInfo of selected entities (no STOP, no illegal)
    pred: str = ""
    terminal: float = 0.0
    comps: dict = field(default_factory=dict)
    stopped: bool = False  # ended by an explicit STOP action

    @property
    def entities(self) -> list:
        return [h.entity for h in self.infos]

    def hop_return(self) -> float:
        return float(sum(s["reward"] for s in self.steps))

    def total_return(self) -> float:
        return self.hop_return() + self.terminal


class EnvPool:
    """LRU of per-example base envs (candidate generation + embeddings are reused across samples)."""

    def __init__(self, cache, cfg: dict, size: int = 512):
        self.cache, self.cfg, self.size = cache, cfg, size
        self._d: OrderedDict[str, TableEnv] = OrderedDict()

    def make(self, ex: Example) -> TableEnv:
        base = self._d.get(ex.id)
        if base is None:
            base = TableEnv(ex, self.cache, self.cfg)
            self._d[ex.id] = base
            if len(self._d) > self.size:
                self._d.popitem(last=False)
        else:
            self._d.move_to_end(ex.id)
        env = copy.copy(base)
        env.reset()
        return env


def rollout(policy: Policy, exs: list[Example], pool: EnvPool, head, rfn: RewardFn,
            greedy: bool = False, gen: torch.Generator | None = None) -> list[Trajectory]:
    """One trajectory per entry of `exs` (repeat an example to sample a group)."""
    envs = [pool.make(e) for e in exs]
    trajs = [Trajectory(e) for e in exs]
    states = [env._state() for env in envs]
    active = list(range(len(exs)))
    policy.eval()
    while active:
        steps = [{"q": states[i].q_emb, "trace": states[i].trace_embs, "cands": states[i].cand_embs,
                  "rel": states[i].rel_ids, "mask": states[i].mask, "feats": states[i].cand_feats} for i in active]
        with torch.no_grad():
            logits, value = run_policy(policy, collate(steps))
            logp_all = torch.log_softmax(logits, -1)
            act = logits.argmax(-1) if greedy else torch.multinomial(logp_all.exp(), 1, generator=gen).squeeze(-1)
        nxt = []
        for k, i in enumerate(active):
            a = int(act[k])
            st = steps[k]
            st.update(action=a, logp=float(logp_all[k, a]), value=float(value[k]), reward=0.0)
            new_state, info, done = envs[i].step(a)
            if info.legal and info.entity is not None:
                trajs[i].infos.append(info)
            st["reward"] = rfn.hop(info, exs[i])
            trajs[i].steps.append(st)
            states[i] = new_state
            if done:
                trajs[i].stopped = info.stop and info.legal
            else:
                nxt.append(i)
        active = nxt
    _finish(trajs, head, rfn)
    return trajs


def _finish(trajs: list[Trajectory], head, rfn: RewardFn) -> None:
    items = [(t.ex, [h.entity for h in t.infos]) for t in trajs]
    preds = head.answer([(ex.id, ex.question, render_evidence(ents, ex, getattr(head, "evidence_mode", "entity"))) for ex, ents in items])
    w = rfn.w
    for t, p in zip(trajs, preds):
        t.pred = p
        t.terminal = rfn.terminal(p, t.ex, t.infos)
        gold_ok = t.ex.gold_path_quality != "none" and bool(t.ex.gold_path)
        schema = -w["w_schema"] if (t.infos and not t.infos[-1].schema_ok) else 0.0
        t.comps = {
            "hop": sum(w["w_hop"] * float(h.on_gold and gold_ok) for h in t.infos),
            "red": -w["w_red"] * sum(h.redundant for h in t.infos),
            "schema": schema, "step": -w["w_step"] * len(t.infos),
            "terminal": t.terminal - schema,
        }
