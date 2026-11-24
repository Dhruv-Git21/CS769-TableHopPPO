"""Masked categorical policy + value head (specs/07 "Policy and RL"). ~0.3M params, CPU.

s_t   = GRU over trace embeddings, initial state = tanh(Wq q)            (B, H)
logit = (W s_t) . c_i + rel_emb[rel_i] . u + beta * cos(q, c_i)
        + v . f_i + tanh(F f_i) . g(s_t)        f_i = candidate features (spec change R2); illegal -> -1e9
value = MLP(s_t)
Shapes: q (B,D), trace (B,T,D) + tlen (B,), cands (B,N,D), rel (B,N) long, mask (B,N) bool, feats (B,N,F).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

NEG = -1e9


class Policy(nn.Module):
    def __init__(self, d_emb: int = 384, hidden: int = 256, n_rel: int = 5, rel_dim: int = 16,
                 n_feat: int = 4, feat_dim: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(d_emb, hidden)
        self.gru = nn.GRU(d_emb, hidden, batch_first=True)
        self.w = nn.Linear(hidden, d_emb, bias=False)
        self.rel_emb = nn.Embedding(n_rel, rel_dim)
        self.u = nn.Linear(hidden, rel_dim, bias=False)
        self.beta = nn.Parameter(torch.tensor(5.0))  # learned scale of the question-candidate cosine
        self.n_feat = n_feat
        self.feat_lin = nn.Linear(n_feat, 1, bias=False)  # state-independent feature score
        self.feat_emb = nn.Linear(n_feat, feat_dim)
        self.g = nn.Linear(hidden, feat_dim, bias=False)  # state-dependent feature weighting
        self.value = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))

    def state(self, q, trace, tlen):
        h0 = torch.tanh(self.q_proj(q))
        if trace.shape[1] == 0:
            return h0
        out, _ = self.gru(trace, h0[None].contiguous())
        seq = torch.cat([h0[:, None], out], dim=1)  # (B, T+1, H); index tlen = state after tlen hops
        return seq[torch.arange(q.shape[0]), tlen]

    def forward(self, q, trace, tlen, cands, rel, mask, feats=None):
        """-> (masked logits (B,N), value (B,))."""
        s = self.state(q, trace, tlen)
        logits = torch.einsum("bd,bnd->bn", self.w(s), cands)
        logits = logits + torch.einsum("bnr,br->bn", self.rel_emb(rel), self.u(s))
        logits = logits + self.beta * torch.einsum("bd,bnd->bn", q, cands)
        if feats is not None:
            logits = logits + self.feat_lin(feats).squeeze(-1)
            logits = logits + torch.einsum("bnf,bf->bn", torch.tanh(self.feat_emb(feats)), self.g(s))
        return logits.masked_fill(~mask, NEG), self.value(s).squeeze(-1)


def collate(steps: list[dict]) -> dict[str, torch.Tensor]:
    """Pad a list of step dicts (numpy: q, trace, cands, rel, mask) into batch tensors."""
    b, d = len(steps), steps[0]["q"].shape[0]
    tmax = max(s["trace"].shape[0] for s in steps)
    nmax = max(s["cands"].shape[0] for s in steps)
    q = np.stack([s["q"] for s in steps]).astype(np.float32)
    trace = np.zeros((b, tmax, d), np.float32)
    cands = np.zeros((b, nmax, d), np.float32)
    rel = np.zeros((b, nmax), np.int64)
    mask = np.zeros((b, nmax), bool)
    nf = next((s["feats"].shape[1] for s in steps if s.get("feats") is not None), 4)
    feats = np.zeros((b, nmax, nf), np.float32)
    tlen = np.zeros(b, np.int64)
    for i, s in enumerate(steps):
        t, n = s["trace"].shape[0], s["cands"].shape[0]
        trace[i, :t], cands[i, :n], rel[i, :n], mask[i, :n], tlen[i] = s["trace"], s["cands"], s["rel"], s["mask"], t
        if s.get("feats") is not None:
            feats[i, :n] = s["feats"]
    return {k: torch.from_numpy(v) for k, v in
            {"q": q, "trace": trace, "tlen": tlen, "cands": cands, "rel": rel, "mask": mask,
             "feats": feats}.items()}


def run_policy(policy: Policy, batch: dict[str, torch.Tensor]):
    return policy(batch["q"], batch["trace"], batch["tlen"], batch["cands"], batch["rel"], batch["mask"],
                  batch.get("feats"))
