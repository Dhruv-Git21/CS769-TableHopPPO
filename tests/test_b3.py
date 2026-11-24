import hashlib

import numpy as np
import torch

from tablehop.baselines import soft_region as sr
from tablehop.data.synthetic import generate
from tablehop.models.encoder import EmbeddingCache
from tablehop.utils.config import get_config

D = 16


def stub_encoder(texts):
    out = []
    for t in texts:
        seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
        out.append(np.random.default_rng(seed).normal(size=D))
    return np.asarray(out, dtype=np.float32)


def test_marginal_nll_value():
    s = torch.tensor([1.0, 2.0, 3.0])
    pos = torch.tensor([False, True, True])
    p = torch.softmax(s, 0)
    assert abs(float(sr.marginal_nll(s, pos)) + float(torch.log(p[1] + p[2]))) < 1e-5


def test_pool_and_positives():
    cfg = get_config("configs/smoke_b3.yaml")
    cache = EmbeddingCache(cache_dir=None, encoder=stub_encoder)
    ex = generate(1, 0)[0]
    refs, x = sr.candidate_pool(ex, cache.example_embs(ex), cfg["env"])
    assert len(refs) == len(set(refs)) and x.shape[0] == len(refs)
    assert x.shape[1] == 3 * D + 3 + sr.N_REL + 2
    pos = sr.positives(ex, refs)
    assert pos.dtype == bool and len(pos) == len(refs)


def test_train_improves_and_predict(tmp_path, monkeypatch):
    cfg = dict(get_config("configs/smoke_b3.yaml"))
    cfg["paths"] = {**cfg["paths"], "results_dir": str(tmp_path)}
    cache = EmbeddingCache(cache_dir=None, encoder=stub_encoder)
    monkeypatch.setattr(sr, "make_cache", lambda c, d: cache)  # avoid loading bge
    run = sr.train(cfg, 0)
    assert (run / "checkpoint.pt").exists() and (run / "train_log.csv").exists()

    class Head:
        def answer_refs(self, items):
            return ["x"] * len(items)

    exs = generate(4, 1)
    recs = sr.predict(run, cfg, exs, Head(), cache)
    assert len(recs) == 4 and all(len(r["path"]) == 3 for r in recs)
