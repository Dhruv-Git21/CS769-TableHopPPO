"""B3: soft-region baseline (specs/07). MLP scorer over hop-0 candidates + their 1-hop neighbours.

Trained by marginal likelihood of candidates whose text contains a gold answer string (answer signal
only, no path labels). Inference: top-`top_n` (3) candidates by score -> frozen answer head.
Candidate features: [q*c, c, q, cos, bm25, hop0_score, is_hop0, relation one-hot(4), q-overlap, header-overlap].
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tablehop.data.load import load_split
from tablehop.data.schema import EntityRef, Example
from tablehop.env.candidates import REL, CandidateGen
from tablehop.eval.metrics import normalize_answer
from tablehop.models.encoder import EmbeddingCache
from tablehop.utils.device import get_device
from tablehop.utils.logging import CSVLogger, get_logger
from tablehop.utils.runmeta import new_run_dir
from tablehop.utils.seeding import set_seed

log = get_logger("b3")
N_REL = 4  # start, same_row, same_col, link


def candidate_pool(ex: Example, embs, env_cfg: dict) -> tuple[list[EntityRef], np.ndarray]:
    """Hop-0 candidates plus every 1-hop neighbour of them. Returns (refs, feats (N, 3D+3+N_REL+2))."""
    gen = CandidateGen(ex, embs, env_cfg)
    h0 = [r for r in gen.hop0().refs if r is not None]
    rel: dict[EntityRef, int] = {r: REL["start"] for r in h0}
    for r in h0:
        nxt = gen.next(r)
        for ref, rid in zip(nxt.refs, nxt.rels):
            if ref is not None and ref not in rel:
                rel[ref] = int(rid)
    refs = list(rel)
    c = gen.embed(refs)  # (N, D)
    q = np.broadcast_to(embs.q, c.shape)
    scal = np.zeros((len(refs), 3 + N_REL), dtype=np.float32)
    for i, r in enumerate(refs):
        if r.kind == "cell":
            scal[i, 1], scal[i, 2] = gen.bm25[r.row, r.col], gen.hop0_score[r.row, r.col]
        scal[i, 0] = float(c[i] @ embs.q)
        scal[i, 3 + rel[r]] = 1.0
    scal[:, 3 + REL["start"]] = [float(r in set(h0)) for r in refs]  # start slot doubles as is_hop0
    lex = gen.features(refs)[:, 2:]  # token-overlap features shared with the RL policy (spec change R2)
    return refs, np.concatenate([q * c, c, q, scal, lex], axis=1).astype(np.float32)


def ref_text(ex: Example, r: EntityRef) -> str:
    if r.kind == "passage":
        return ex.passages.get(r.passage_id or "", "")
    return ex.table.rows[r.row][r.col].text


def positives(ex: Example, refs: list[EntityRef]) -> np.ndarray:
    """(N,) bool: candidate text contains a normalised gold answer string (token-boundary match)."""
    golds = [f" {g} " for g in (normalize_answer(a) for a in ex.answer) if g]
    out = np.zeros(len(refs), dtype=bool)
    for i, r in enumerate(refs):
        t = f" {normalize_answer(ref_text(ex, r))} "
        out[i] = any(g in t for g in golds)
    return out


class Scorer(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, d_in) -> (N,)
        return self.net(x).squeeze(-1)


def marginal_nll(scores: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """-log sum_{i in pos} softmax(scores)_i. scores (N,), pos (N,) bool with >=1 True."""
    return -(torch.logsumexp(scores.masked_fill(~pos, float("-inf")), 0) - torch.logsumexp(scores, 0))


def featurize(exs: list[Example], cache: EmbeddingCache, env_cfg: dict) -> list[tuple]:
    out = []
    for e in exs:
        refs, x = candidate_pool(e, cache.example_embs(e), env_cfg)
        out.append((refs, x, positives(e, refs)))
    return out


def select(scorer: Scorer, refs: list[EntityRef], x: np.ndarray, top_n: int) -> list[EntityRef]:
    with torch.no_grad():
        s = scorer(torch.from_numpy(x)).numpy()
    return [refs[i] for i in np.argsort(-s, kind="stable")[:top_n]]


def _recall_top_n(scorer: Scorer, feats: list[tuple], top_n: int) -> float:
    """Fraction of examples having a positive candidate whose top-n contains one (no answer head)."""
    hit = tot = 0
    scorer.eval()
    with torch.no_grad():
        for refs, x, pos in feats:
            if not pos.any():
                continue
            tot += 1
            top = np.argsort(-scorer(torch.from_numpy(x)).numpy(), kind="stable")[:top_n]
            hit += bool(pos[top].any())
    scorer.train()
    return hit / max(tot, 1)


def make_cache(cfg: dict, device: str) -> EmbeddingCache:
    return EmbeddingCache(cfg["env"]["encoder"], cfg["paths"]["cache_dir"], device,
                          batch_size=int(cfg["env"]["embed_batch"]))


def train(cfg: dict, seed: int) -> Path:
    """Train the scorer; saves checkpoint.pt (state dict + dims). Returns the run dir."""
    scfg = dict(cfg["soft"])
    set_seed(seed)
    rng = random.Random(seed)
    rm = new_run_dir("b3_soft", cfg["data"].get("dataset", "hybridqa"), seed, cfg,
                     cfg.get("paths", {}).get("results_dir", "results"))
    cache = make_cache(cfg, get_device(cfg.get("device", "auto")))
    t0 = time.time()
    tr = [e for e in load_split(cfg["data"], "train") if e.answer]
    feats = [f for f in featurize(tr, cache, cfg["env"]) if f[2].any()]
    dev = featurize(load_split(cfg["data"], "dev")[: scfg["eval_subset"]], cache, cfg["env"])
    log.info("train %d/%d examples with a positive candidate; featurize %.0fs",
             len(feats), len(tr), time.time() - t0)
    if not feats:
        raise SystemExit("no training example has a candidate containing the gold answer")
    scorer = Scorer(feats[0][1].shape[1], scfg["hidden"], scfg["dropout"])
    opt = torch.optim.AdamW(scorer.parameters(), lr=scfg["lr"], weight_decay=scfg["weight_decay"])
    logger = CSVLogger(rm.run_dir)
    bs, seen, step, next_eval, losses = scfg["batch_size"], 0, 0, scfg["eval_every_examples"], []
    order: list[int] = []
    while seen < scfg["budget_examples"]:
        if len(order) < bs:
            order = list(range(len(feats)))
            rng.shuffle(order)
        idx, order = order[:bs], order[bs:]
        loss = sum(marginal_nll(scorer(torch.from_numpy(feats[i][1])), torch.from_numpy(feats[i][2]))
                   for i in idx) / len(idx)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        opt.step()
        step += 1
        seen += len(idx)
        losses.append(loss.item())
        if seen >= next_eval or seen >= scfg["budget_examples"]:
            next_eval += scfg["eval_every_examples"]
            rec = _recall_top_n(scorer, dev, scfg["top_n"])
            logger.log({"step": step, "examples_seen": seen, "train_loss": float(np.mean(losses[-50:])),
                        "dev_pos_in_top_n": rec, "elapsed_s": time.time() - t0})
            log.info("examples %d loss %.4f dev pos-in-top%d %.3f", seen, np.mean(losses[-50:]),
                     scfg["top_n"], rec)
    torch.save({"state": scorer.state_dict(), "d_in": feats[0][1].shape[1], "scfg": scfg},
               rm.run_dir / "checkpoint.pt")
    rm.meta["train_time_s"] = time.time() - t0
    rm.meta["examples_seen"] = seen
    rm.finish("done")
    return rm.run_dir


def load_scorer(run_dir: Path) -> tuple[Scorer, dict]:
    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    s = Scorer(ck["d_in"], ck["scfg"]["hidden"], ck["scfg"]["dropout"])
    s.load_state_dict(ck["state"])
    return s.eval(), ck["scfg"]


def predict(run_dir: Path, cfg: dict, exs: list[Example], head, cache: EmbeddingCache):
    """Top-n selection -> frozen answer head. Returns per-example prediction dicts (metrics.aggregate input).

    The selection is an unordered set (rank order); trace_info marks every entity legal and non-redundant,
    so `path_validity_strict` = fraction of selected entities that lie on the gold path.
    """
    scorer, scfg = load_scorer(run_dir)
    sels = []
    for e in exs:
        refs, x = candidate_pool(e, cache.example_embs(e), cfg["env"])
        sels.append(select(scorer, refs, x, scfg["top_n"]))
    preds = head.answer_refs(list(zip(exs, sels)))
    return [{"id": e.id, "question": e.question, "pred": p, "gold": e.answer, "path": s,
             "gold_path": list(e.gold_path or []),
             "trace_info": [{"entity": r, "legal": True, "redundant": False} for r in s]}
            for e, p, s in zip(exs, preds, sels)]
