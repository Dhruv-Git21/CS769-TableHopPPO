"""Candidate generation (specs/07). STOP is always the LAST candidate (ref None); illegal at t=0.

Relation ids: start=0, same_row=1, same_col=2, link=3, stop=4.
hop 0:   top-K cells by  w*bm25_norm + (1-w)*cos(question, cell).
hop t>0: cells in the current entity's row, top `col_topk` cells of its column by cosine, the linked
         passage if the current entity is a linked cell, STOP.  Unrestricted (A4): top-K cells of the
         whole table by cosine + linked passage + STOP.
Current entity = a passage -> its source cell's (row, col) is used for row/column neighbours; no link.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from tablehop.data.gold_paths import tokens
from tablehop.data.schema import EntityRef, Example
from tablehop.env.render import render_cell
from tablehop.models.encoder import ExampleEmbs

REL = {"start": 0, "same_row": 1, "same_col": 2, "link": 3, "stop": 4}
_TOK = re.compile(r"\w+")
N_FEAT = 4  # per-candidate lexical/retrieval features, see CandidateGen.features


@dataclass
class Candidates:
    refs: list[EntityRef | None]  # last is STOP (None)
    rels: np.ndarray  # (N,) int64
    mask: np.ndarray  # (N,) bool


class CandidateGen:
    """Per-example precomputation: cosine map (R, C), BM25-normalised map (R, C), hop-0 top-K."""

    def __init__(self, ex: Example, embs: ExampleEmbs, env_cfg: dict):
        self.ex, self.embs = ex, embs
        self.K, self.col_topk = int(env_cfg["K"]), int(env_cfg["col_topk"])
        self.unrestricted = bool(env_cfg["unrestricted"])
        t = ex.table
        self.R = len(t.rows)
        self.C = len(t.rows[0]) if t.rows else 0
        self.cos = embs.cells @ embs.q  # (R, C)
        self.bm25 = self._bm25_norm()
        w = float(env_cfg["bm25_weight"])
        self.hop0_score = w * self.bm25 + (1 - w) * self.cos
        self._qtok = tokens(ex.question)
        self._feat_cache: dict = {}

    def _bm25_norm(self) -> np.ndarray:
        t = self.ex.table
        corpus = [_TOK.findall(render_cell(t, r, c).lower()) for r in range(self.R) for c in range(self.C)]
        if not corpus:
            return np.zeros((self.R, self.C))
        s = np.asarray(BM25Okapi(corpus).get_scores(_TOK.findall(self.ex.question.lower())), dtype=np.float64)
        rng = s.max() - s.min()
        s = (s - s.min()) / rng if rng > 0 else np.zeros_like(s)
        return s.reshape(self.R, self.C)

    def _cell(self, r: int, c: int) -> EntityRef:
        return EntityRef("cell", self.ex.table.id, r, c)

    def _topk_cells(self, score: np.ndarray, k: int) -> list[tuple[int, int]]:
        flat = np.argsort(-score, axis=None, kind="stable")[:k]
        return [(int(i // self.C), int(i % self.C)) for i in flat]

    def hop0(self) -> Candidates:
        refs: list[EntityRef | None] = [self._cell(r, c) for r, c in self._topk_cells(self.hop0_score, self.K)]
        return self._pack(refs, [REL["start"]] * len(refs), t=0)

    def next(self, cur: EntityRef) -> Candidates:
        """Candidates after selecting `cur` (cell or passage), t >= 1."""
        r0, c0 = cur.row, cur.col
        refs: list[EntityRef | None] = []
        rels: list[int] = []
        seen: set[tuple[int, int]] = set()

        def add(r: int, c: int, rel: str) -> None:
            if (r, c) not in seen:
                seen.add((r, c))
                refs.append(self._cell(r, c))
                rels.append(REL[rel])

        if self.unrestricted:
            for r, c in self._topk_cells(self.cos, self.K):
                add(r, c, "same_row" if r == r0 else "same_col" if c == c0 else "start")
        else:
            for c in range(self.C):
                add(r0, c, "same_row")
            col = np.argsort(-self.cos[:, c0], kind="stable")[: self.col_topk]
            for r in col:
                add(int(r), c0, "same_col")
        if cur.kind == "cell":
            link = self.ex.table.rows[r0][c0].link
            if link is not None:
                refs.append(EntityRef("passage", self.ex.table.id, r0, c0, link))
                rels.append(REL["link"])
        return self._pack(refs, rels, t=1)

    def _pack(self, refs: list, rels: list[int], t: int) -> Candidates:
        refs = [*refs, None]
        rels_a = np.asarray([*rels, REL["stop"]], dtype=np.int64)
        mask = np.ones(len(refs), dtype=bool)
        if t == 0:
            mask[-1] = False  # STOP illegal at t=0
        return Candidates(refs, rels_a, mask)

    def _overlap(self, text: str) -> float:
        return len(self._qtok & tokens(text)) / len(self._qtok) if self._qtok else 0.0

    def features(self, refs: list[EntityRef | None]) -> np.ndarray:
        """(N, N_FEAT) float32, STOP -> zeros (spec change R2, STATUS.md). Same for every RL method:
        [cos(q, entity), bm25_norm (cells; 0 for passages), question-token overlap with the entity text,
        question-token overlap with the column header (cells; 0 for passages)]."""
        out = np.zeros((len(refs), N_FEAT), dtype=np.float32)
        t = self.ex.table
        for i, e in enumerate(refs):
            if e is None:
                continue
            f = self._feat_cache.get(e)
            if f is None:
                if e.kind == "passage":
                    txt = self.ex.passages.get(e.passage_id, "")
                    pe = self.embs.passages.get(e.passage_id)
                    f = (float(pe @ self.embs.q) if pe is not None else 0.0, 0.0, self._overlap(txt), 0.0)
                else:
                    hdr = t.header[e.col] if e.col < len(t.header) else ""
                    f = (float(self.cos[e.row, e.col]), float(self.bm25[e.row, e.col]),
                         self._overlap(t.rows[e.row][e.col].text), self._overlap(hdr))
                self._feat_cache[e] = f
            out[i] = f
        return out

    def embed(self, refs: list[EntityRef | None]) -> np.ndarray:
        """(N, D) embeddings; STOP -> zeros."""
        d = self.embs.q.shape[0]
        out = np.zeros((len(refs), d), dtype=np.float32)
        for i, e in enumerate(refs):
            if e is None:
                continue
            out[i] = self.embs.passages[e.passage_id] if e.kind == "passage" else self.embs.cells[e.row, e.col]
        return out


def first_gold_in_hop0(gen: CandidateGen) -> bool | None:
    """Is the first gold entity among the hop-0 candidates? None if the example has no gold path."""
    gp = gen.ex.gold_path
    if not gp:
        return None
    return gp[0] in set(gen.hop0().refs)
