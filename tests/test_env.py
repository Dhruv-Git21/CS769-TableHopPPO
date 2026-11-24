import hashlib

import numpy as np
import pytest

from tablehop.data.synthetic import generate
from tablehop.env.candidates import REL
from tablehop.env.table_env import TableEnv
from tablehop.models.encoder import EmbeddingCache
from tablehop.utils.config import get_config

CFG = get_config("configs/base.yaml")
D = 16


def stub_encoder(texts):
    out = []
    for t in texts:
        seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
        out.append(np.random.default_rng(seed).normal(size=D))
    return np.asarray(out, dtype=np.float32)


@pytest.fixture(scope="module")
def cache():
    return EmbeddingCache(cache_dir=None, encoder=stub_encoder)


def cfg_with(**env):
    c = get_config("configs/base.yaml")
    c["env"].update(env)
    return c


def legal_idx(state):
    return [i for i, m in enumerate(state.mask) if m]


def test_cache_recompute_nothing(cache):
    exs = generate(2, 0)
    cache.example_embs(exs[0])
    n = cache.n_computed
    cache.example_embs(exs[0])
    assert cache.n_computed == n
    e = cache.embed(["abc"])
    assert abs(np.linalg.norm(e[0]) - 1) < 1e-2


def test_stop_illegal_at_t0_and_last(cache):
    ex = generate(1, 0)[0]
    env = TableEnv(ex, cache, CFG)
    s = env.reset()
    assert s.cand_refs[-1] is None and s.rel_ids[-1] == REL["stop"]
    assert not s.mask[-1]
    assert s.t == 0 and s.trace_embs.shape == (0, D)
    _, info, done = env.step(len(s.mask) - 1)
    assert done and not info.legal and info.stop and env.trace == []


def test_masked_softmax_never_samples_illegal(cache):
    rng = np.random.default_rng(0)
    for ex in generate(5, 1):
        env = TableEnv(ex, cache, CFG)
        s = env.reset()
        while True:
            logits = rng.normal(size=len(s.mask))
            logits[~s.mask] = -np.inf
            p = np.exp(logits - logits.max())
            p /= p.sum()
            assert p[~s.mask].sum() == 0
            a = int(rng.choice(len(p), p=p))
            assert s.mask[a]
            s, info, done = env.step(a)
            assert info.legal
            if done:
                break


def test_forced_stop_at_H(cache):
    ex = generate(1, 2)[0]
    env = TableEnv(ex, cache, cfg_with(max_hops=2))
    s = env.reset()
    s, _, done = env.step(0)
    assert not done
    a = next(i for i in legal_idx(s) if s.cand_refs[i] is not None)
    s, _, done = env.step(a)
    assert done and env.forced_stop and s.t == 2
    with pytest.raises(RuntimeError):
        env.step(0)


def test_stop_legal_after_t0_and_visited_redundant(cache):
    ex = generate(1, 3)[0]
    env = TableEnv(ex, cache, CFG)
    s = env.reset()
    first = s.cand_refs[0]
    s, info, _ = env.step(0)
    assert not info.redundant
    assert s.mask[-1]
    j = s.cand_refs.index(first)  # own cell is a same_row candidate: legal but redundant
    s, info, _ = env.step(j)
    assert info.legal and info.redundant


def test_link_only_from_linked_cells(cache):
    for ex in generate(6, 4):
        env = TableEnv(ex, cache, CFG)
        s = env.reset()
        for i in range(len(s.cand_refs) - 1):
            e = TableEnv(ex, cache, CFG)
            e.reset()
            s2, info, _ = e.step(i)
            cell = ex.table.rows[info.entity.row][info.entity.col]
            links = [k for k, r in enumerate(s2.rel_ids) if r == REL["link"]]
            assert len(links) == (1 if cell.link else 0)
            for k in links:
                assert s2.cand_refs[k].passage_id == cell.link and s2.cand_refs[k].kind == "passage"


def test_unrestricted_changes_candidate_set(cache):
    ex = generate(1, 5)[0]
    a = TableEnv(ex, cache, CFG)
    b = TableEnv(ex, cache, cfg_with(unrestricted=True, K=4))
    sa, sb = a.reset(), b.reset()
    sa, _, _ = a.step(0)
    sb, _, _ = b.step(0)
    assert set(sa.cand_refs) != set(sb.cand_refs)
    cells_b = [r for r in sb.cand_refs if r is not None and r.kind == "cell"]
    assert len(cells_b) == 4


def test_gold_path_reachable_stepwise(cache):
    for seed in range(3):
        for ex in generate(20, seed):
            env = TableEnv(ex, cache, CFG)
            s = env.reset()
            for k, g in enumerate(ex.gold_path):
                assert g in s.cand_refs, (ex.id, k)
                s, info, done = env.step(s.cand_refs.index(g))
                assert info.legal and info.on_gold and not info.redundant
            if not done:
                _, info, done = env.step(len(s.mask) - 1)  # STOP
                assert done and info.legal


def test_gold_reachable_with_real_hop0_K(cache):
    # hop-0 K=32 covers the whole small synthetic table (<=8x4 cells)
    for ex in generate(10, 7):
        env = TableEnv(ex, cache, CFG)
        assert ex.gold_path[0] in env.reset().cand_refs


def test_order_consistent_on_gold(cache):
    ex = generate(1, 0)[0]
    env = TableEnv(ex, cache, CFG)
    s = env.reset()
    gp = ex.gold_path
    s, i1, _ = env.step(s.cand_refs.index(gp[0]))
    s, i2, _ = env.step(s.cand_refs.index(gp[1]))
    assert i1.on_gold and i2.on_gold
    s, i3, _ = env.step(s.cand_refs.index(gp[0]))  # back to an earlier gold entity: not order-consistent
    assert i3.redundant and not i3.on_gold


def test_revisiting_gold_entity_is_not_on_gold_again(cache):
    """Regression: looping on a gold cell must not re-earn the hop reward."""
    ex = generate(1, 0)[0]
    env = TableEnv(ex, cache, cfg_with(K=10_000))  # all cells at hop 0
    s = env.reset()
    first = ex.gold_path[0]
    s, info, _ = env.step(s.cand_refs.index(first))
    assert info.on_gold
    s, info2, _ = env.step(s.cand_refs.index(first))  # own cell is a same_row candidate
    assert info2.redundant and not info2.on_gold


def test_row_evidence_mode_dedupes_rows_and_keeps_passages():
    from tablehop.data.schema import EntityRef
    from tablehop.env.render import render_evidence

    ex = next(e for e in generate(20, 0) if any(r.kind == "passage" for r in e.gold_path))
    t = ex.table
    p = next(r for r in ex.gold_path if r.kind == "passage")
    refs = [EntityRef("cell", t.id, 0, 0), EntityRef("cell", t.id, 0, 1), p]
    ev = render_evidence(refs, ex, "row")
    assert len(ev) == 2 and ev[0].startswith(t.title + " | ") and all(h in ev[0] for h in t.header)
    assert ev[1] == render_evidence([p], ex)[0]
    with pytest.raises(ValueError):
        render_evidence(refs, ex, "bogus")
