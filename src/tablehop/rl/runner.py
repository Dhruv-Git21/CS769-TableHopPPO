"""Shared training harness for PPO (H1, A1-A4) and GRPO (B2a, B2b): setup, loop, eval, checkpoint/resume.

Budget unit = trajectories (cfg rl.budget_trajectories). PPO: rl.rollout_batch questions x 1 sample per
update; GRPO: grpo.questions_per_update x grpo.group_size. Policy/value live on CPU; the frozen answer head
uses cfg device. Synthetic configs (`answer_head.type: stub_synthetic`) use a hashed bag-of-words encoder and a
rule-based head: that is the LABELLED sanity setting only, never a benchmark.
"""
from __future__ import annotations

import json
import random
import re
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from tablehop.data.load import load_split
from tablehop.eval.metrics import aggregate, path_validity
from tablehop.models.encoder import EmbeddingCache
from tablehop.models.policy import Policy
from tablehop.rl.reward import RewardFn
from tablehop.rl.rollout import EnvPool, Trajectory, rollout
from tablehop.utils.logging import CSVLogger, get_logger
from tablehop.utils.runmeta import RunMeta, new_run_dir
from tablehop.utils.seeding import set_seed

log = get_logger("rl")
_W = re.compile(r"\w+")


def hashed_encoder(dim: int):
    """Deterministic bag-of-words hashing encoder (synthetic sanity only)."""
    def enc(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), dim), np.float32)
        for i, t in enumerate(texts):
            for w in _W.findall(t.lower()):
                out[i, zlib.crc32(w.encode()) % dim] += 1.0
        return out
    return enc


class StubSyntheticHead:
    """Rule-based answer head for synthetic data: answer = last evidence value (year for passages)."""
    sha256 = "stub_synthetic"

    def answer(self, batch) -> list[str]:
        outs = []
        for _, _, ev in batch:
            if not ev:
                outs.append("")
                continue
            last = ev[-1]
            m = re.search(r"founded in (\d{4})", last)
            if " : " in last:
                outs.append(m.group(1) if m else "")
            else:
                outs.append(last.rsplit(" | ", 1)[-1].strip())
        return outs


def build_cache_head(cfg: dict):
    ah = cfg.get("answer_head", {})
    if ah.get("type") == "stub_synthetic":
        cache = EmbeddingCache(cache_dir=None, encoder=hashed_encoder(int(cfg["rl"]["stub_dim"])))
        return cache, StubSyntheticHead()
    from tablehop.models.answer_head import AnswerHead
    from tablehop.utils.device import get_device

    dev = get_device(cfg.get("device", "auto"))
    cache = EmbeddingCache(cfg["env"]["encoder"], cfg["paths"]["cache_dir"], device=dev,
                           batch_size=int(cfg["env"]["embed_batch"]))
    return cache, AnswerHead.load(ah["path"], dev)


def make_policy(cfg: dict, d_emb: int) -> Policy:
    return Policy(d_emb, int(cfg["rl"]["hidden"]))


def evaluate_policy(policy, exs, pool, head, rfn, batch: int = 64) -> list[dict]:
    """Greedy rollouts -> per-example records (aggregate() keys + hop rewards / reward components)."""
    recs: list[dict] = []
    for s in range(0, len(exs), batch):
        for t in rollout(policy, exs[s:s + batch], pool, head, rfn, greedy=True):
            ents = t.entities
            recs.append({
                "id": t.ex.id, "question": t.ex.question, "pred": t.pred, "gold": t.ex.answer,
                "path": ents, "gold_path": t.ex.gold_path if t.ex.gold_path_quality != "none" else None,
                "trace_info": [{"entity": h.entity, "legal": h.legal, "redundant": h.redundant}
                               for h in t.infos],
                "hop_rewards": [s_["reward"] for s_ in t.steps], "terminal": t.terminal,
                "comps": t.comps, "stopped": t.stopped})
    return recs


def batch_stats(trajs: list[Trajectory]) -> dict:
    """Reward-hacking signals and reward components over a training batch."""
    n = len(trajs)
    comps = {k: float(np.mean([t.comps[k] for t in trajs])) for k in ("hop", "red", "schema", "step", "terminal")}
    val = []
    for t in trajs:
        gp = t.ex.gold_path
        if gp and t.ex.gold_path_quality != "none" and t.infos:
            val.append(path_validity([{"entity": h.entity, "legal": h.legal, "redundant": h.redundant}
                                      for h in t.infos], gp)[0])
    return {
        "mean_return": float(np.mean([t.total_return() for t in trajs])),
        "mean_terminal": float(np.mean([t.terminal for t in trajs])),
        "train_f1": float(np.mean([t.comps["terminal"] for t in trajs])),
        "mean_hops": float(np.mean([len(t.infos) for t in trajs])),
        "hop_validity": float(np.mean(val)) if val else float("nan"),
        "stop_at_t1_rate": float(np.mean([len(t.infos) <= 1 for t in trajs])),
        "unique_entity_ratio": float(np.mean([len(set(t.entities)) / max(1, len(t.entities)) for t in trajs])),
        **{f"r_{k}": v for k, v in comps.items()}, "n": n}


class Sampler:
    """Epoch-shuffled question stream (resumable)."""

    def __init__(self, n: int, seed: int):
        self.n, self.rng, self.order, self.pos = n, random.Random(seed), [], 0

    def next(self, k: int) -> list[int]:
        out = []
        while len(out) < k:
            if self.pos >= len(self.order):
                self.order, self.pos = list(range(self.n)), 0
                self.rng.shuffle(self.order)
            out.append(self.order[self.pos])
            self.pos += 1
        return out


def run(cfg: dict, seed: int, algo: str) -> Path:
    from tablehop.rl import grpo, ppo

    set_seed(seed)
    rcfg = cfg["rl"]
    rfn = RewardFn(cfg)
    cache, head = build_cache_head(cfg)
    train_ex = [e for e in load_split(cfg["data"], "train") if e.answer]
    dev = load_split(cfg["data"], "dev")[: int(rcfg["eval_subset"])]
    pool = EnvPool(cache, cfg, int(rcfg["env_pool"]))
    d_emb = int(cache.example_embs(train_ex[0]).q.shape[0])
    policy = make_policy(cfg, d_emb)
    opt = torch.optim.Adam(policy.parameters(), lr=float(cfg["ppo"]["lr_head"]))
    pcfg = {**cfg["ppo"], "minibatch": rcfg["minibatch"]}
    if algo == "ppo":
        per_update, nq, group = int(cfg["ppo"]["rollout_batch"]), int(cfg["ppo"]["rollout_batch"]), 1
    else:
        group, nq = int(cfg["grpo"]["group_size"]), int(cfg["grpo"]["questions_per_update"])
        per_update = nq * group
    n_upd = int(rcfg["budget_trajectories"]) // per_update

    start, seen, rows = 0, 0, []
    sampler = Sampler(len(train_ex), seed)
    gen = torch.Generator().manual_seed(seed)
    nprng = np.random.default_rng(seed)
    if cfg.get("resume"):
        run_dir = Path(cfg["resume"])
        rm = RunMeta(run_dir, json.loads((run_dir / "meta.json").read_text()))
        ck = torch.load(run_dir / "checkpoint.pt", weights_only=False)
        policy.load_state_dict(ck["policy"])
        opt.load_state_dict(ck["opt"])
        start, seen, rows = ck["update"], ck["seen"], ck["rows"]
        sampler.order, sampler.pos, sampler.rng = ck["order"], ck["pos"], random.Random(seed + start)
        gen.set_state(ck["gen"])
        nprng.bit_generator.state = ck["nprng"]
        log.info("resumed %s at update %d", run_dir.name, start)
    else:
        rm = new_run_dir(cfg["method"], cfg["data"].get("dataset", "hybridqa"), seed, cfg,
                         cfg.get("paths", {}).get("results_dir", "results"))
        rm.meta["answer_head_sha256"] = getattr(head, "sha256", "unknown")
        rm._write()
        run_dir = rm.run_dir
    logger = CSVLogger(run_dir)
    logger.rows = rows
    t0 = time.time() - (rows[-1]["elapsed_s"] if rows else 0.0)

    def save(u: int) -> None:
        torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(), "update": u, "seen": seen,
                    "rows": logger.rows, "order": sampler.order, "pos": sampler.pos, "gen": gen.get_state(),
                    "nprng": nprng.bit_generator.state, "d_emb": d_emb, "hidden": int(rcfg["hidden"])},
                   run_dir / "checkpoint.pt")

    for u in range(start, n_upd):
        qs = [train_ex[i] for i in sampler.next(nq)]
        exs = [e for e in qs for _ in range(group)]
        trajs = rollout(policy, exs, pool, head, rfn, greedy=False, gen=gen)
        seen += len(trajs)
        ent = float(cfg["ppo"]["entropy_coef"]) * (1 - u / n_upd)
        if algo == "ppo":
            steps, adv, ret = ppo.flatten_ppo(trajs, float(cfg["ppo"]["gamma"]), float(cfg["ppo"]["gae_lambda"]))
            st = ppo.update(policy, opt, steps, adv, ret, pcfg, ent, True, nprng)
            st["explained_var"] = ppo.explained_variance(np.array([s["value"] for s in steps]), ret)
        else:
            steps, adv = grpo.flatten_grpo(trajs, group)
            st = ppo.update(policy, opt, steps, adv, None, pcfg, ent, False, nprng)
        row = {"update": u + 1, "trajectories_seen": seen, **batch_stats(trajs), **st, "entropy_coef": ent,
               "elapsed_s": time.time() - t0}
        if (u + 1) % int(rcfg["eval_every"]) == 0 or u + 1 == n_upd:
            m = aggregate(evaluate_policy(policy, dev, pool, head, rfn))
            row.update(dev_em=m["em"], dev_f1=m["f1"], dev_validity_strict=m["path_validity_strict"],
                       dev_mean_hops=m["mean_hops"])
            log.info("upd %d/%d seen %d ret %.3f dev EM %.3f F1 %.3f valid %.3f hops %.2f (%.0fs)", u + 1, n_upd,
                     seen, row["mean_return"], m["em"], m["f1"], m["path_validity_strict"], m["mean_hops"],
                     time.time() - t0)
            logger.log(row)
            save(u + 1)
        else:
            logger.log(row)
    rm.meta.update(train_time_s=time.time() - t0, trajectories_seen=seen, n_updates=n_upd)
    rm.finish("done")
    return run_dir
