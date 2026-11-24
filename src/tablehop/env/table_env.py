"""TableEnv: one episode = pick entities (hops) until STOP or the max-hop limit (specs/07, 02).

Action = index into the current candidate list. Illegal action (masked, incl. STOP at t=0): the episode
ends at once with HopInfo.legal=False and nothing is added to the trace. Selecting the H-th entity forces a stop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tablehop.data.schema import EntityRef, Example
from tablehop.env.candidates import CandidateGen, Candidates
from tablehop.models.encoder import EmbeddingCache, ExampleEmbs
from tablehop.rl.reward import schema_ok


@dataclass
class State:
    q_emb: np.ndarray  # (D,)
    trace_embs: np.ndarray  # (t, D)
    cand_embs: np.ndarray  # (N, D)  STOP row = 0
    rel_ids: np.ndarray  # (N,)
    mask: np.ndarray  # (N,) bool
    cand_refs: list  # N entries, EntityRef or None (STOP)
    t: int
    cand_feats: np.ndarray | None = None  # (N, N_FEAT) lexical/retrieval features (spec change R2)


@dataclass
class HopInfo:
    entity: EntityRef | None  # None = STOP
    on_gold: bool
    redundant: bool
    schema_ok: bool
    legal: bool
    stop: bool = False


class TableEnv:
    def __init__(self, example: Example, cache: EmbeddingCache | None, cfg: dict,
                 embs: ExampleEmbs | None = None):
        self.ex, self.cfg = example, cfg
        self.H = int(cfg["env"]["max_hops"])
        self.cues = cfg["reward"]["schema_cues"]
        embs = embs if embs is not None else cache.example_embs(example)
        self.gen = CandidateGen(example, embs, cfg["env"])
        self.gold = list(example.gold_path or [])
        self.reset()

    def reset(self) -> State:
        self.t = 0
        self.done = False
        self.forced_stop = False
        self.trace: list[HopInfo] = []
        self.visited: set[EntityRef] = set()
        self._max_gold = -1
        self._embs: list[np.ndarray] = []
        self._cands = self.gen.hop0()
        return self._state()

    @property
    def entities(self) -> list[EntityRef]:
        return [h.entity for h in self.trace]

    def _state(self) -> State:
        c: Candidates = self._cands
        d = self.gen.embs.q.shape[0]
        trace = np.stack(self._embs) if self._embs else np.zeros((0, d), np.float32)
        return State(self.gen.embs.q, trace, self.gen.embed(c.refs), c.rels, c.mask, c.refs, self.t,
                     self.gen.features(c.refs))

    def step(self, a: int) -> tuple[State, HopInfo, bool]:
        if self.done:
            raise RuntimeError("step() on a finished episode")
        c = self._cands
        if not 0 <= a < len(c.refs):
            raise IndexError(f"action {a} outside candidate list of {len(c.refs)}")
        ref = c.refs[a]
        if not c.mask[a]:
            self.done = True
            return self._state(), HopInfo(ref, False, False, True, False, stop=ref is None), True
        if ref is None:
            self.done = True
            return self._state(), HopInfo(None, False, False, True, True, stop=True), True
        redundant = ref in self.visited
        on_gold = False
        if ref in self.gold and not redundant:  # a revisit never re-earns the hop reward (anti-loop)
            idx = self.gold.index(ref)
            if idx >= self._max_gold:  # order-consistent (spec 02)
                on_gold, self._max_gold = True, idx
        info = HopInfo(ref, on_gold, redundant, schema_ok(ref, self.ex, self.cues), True)
        self.visited.add(ref)
        self.trace.append(info)
        self._embs.append(self.gen.embed([ref])[0])
        self.t += 1
        if self.t >= self.H:
            self.done = self.forced_stop = True
        else:
            self._cands = self.gen.next(ref)
        return self._state(), info, self.done
