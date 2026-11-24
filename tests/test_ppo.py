"""Spec 02 PPO tests 1-5 (CPU, synthetic)."""
import numpy as np
import torch

from tablehop.data.load import load_split
from tablehop.models.policy import NEG, Policy, collate, run_policy
from tablehop.rl.ppo import clipped_loss, gae, update
from tablehop.rl.reward import RewardFn
from tablehop.rl.rollout import EnvPool, rollout
from tablehop.rl.runner import StubSyntheticHead, build_cache_head, evaluate_policy
from tablehop.utils.config import get_config

D = 8


def _step(q, cands, action, logp, rel=None, mask=None, reward=0.0):
    n = cands.shape[0]
    return {"q": q, "trace": np.zeros((0, D), np.float32), "cands": cands,
            "rel": np.zeros(n, np.int64) if rel is None else rel,
            "mask": np.ones(n, bool) if mask is None else mask, "action": action, "logp": logp,
            "value": 0.0, "reward": reward}


def test_bandit_solved():
    """3-armed contextual bandit: best arm = the one whose embedding matches the context."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    pol = Policy(D, 32)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-3)
    pcfg = {"epochs": 4, "minibatch": 64, "clip_eps": 0.2, "value_coef": 0.5, "grad_clip": 0.5}
    eye = np.eye(D, dtype=np.float32)[:3]
    for _ in range(60):
        steps, rets, advs = [], [], []
        for _ in range(64):
            k = int(rng.integers(3))
            q = eye[k] + 0.05 * rng.normal(size=D).astype(np.float32)
            s = _step(q, eye, 0, 0.0)
            with torch.no_grad():
                lg, v = run_policy(pol, collate([s]))
            lp = torch.log_softmax(lg, -1)[0]
            a = int(torch.multinomial(lp.exp(), 1))
            s.update(action=a, logp=float(lp[a]), value=float(v[0]), reward=float(a == k))
            steps.append(s)
            advs.append(s["reward"] - s["value"])
            rets.append(s["reward"])
        update(pol, opt, steps, np.array(advs), np.array(rets), pcfg, 0.0, True, rng)
    ok = 0
    for k in range(3):
        for _ in range(20):
            q = eye[k] + 0.05 * rng.normal(size=D).astype(np.float32)
            lg, _ = run_policy(pol, collate([_step(q, eye, 0, 0.0)]))
            ok += int(lg.argmax()) == k
    assert ok / 60 >= 0.9


def test_masked_never_sampled():
    torch.manual_seed(0)
    pol = Policy(D, 16)
    q = np.random.randn(D).astype(np.float32)
    c = np.random.randn(6, D).astype(np.float32)
    mask = np.array([0, 1, 0, 0, 1, 0], bool)
    logits, _ = run_policy(pol, collate([_step(q, c, 0, 0.0, mask=mask)]))
    assert (logits[0, ~torch.from_numpy(mask)] <= NEG / 2).all()
    p = torch.softmax(logits, -1)
    draws = torch.multinomial(p.expand(2000, -1), 1).squeeze(-1)
    assert mask[draws.numpy()].all()


def test_gae_hand_example():
    # gamma=0.9, lam=0.5; r=[1,0,2], v=[0.5,0.4,0.3], V(s_T)=0
    adv, ret = gae([1.0, 0.0, 2.0], [0.5, 0.4, 0.3], 0.9, 0.5)
    d2 = 2.0 - 0.3
    d1 = 0.0 + 0.9 * 0.3 - 0.4
    d0 = 1.0 + 0.9 * 0.4 - 0.5
    a2 = d2
    a1 = d1 + 0.9 * 0.5 * a2
    a0 = d0 + 0.9 * 0.5 * a1
    np.testing.assert_allclose(adv, [a0, a1, a2])
    np.testing.assert_allclose(ret, np.array([a0, a1, a2]) + [0.5, 0.4, 0.3])


def test_ratio_one_and_gradient_sign():
    q = np.random.RandomState(0).randn(D).astype(np.float32)
    c = np.random.RandomState(1).randn(5, D).astype(np.float32)
    for sign in (1.0, -1.0):
        torch.manual_seed(0)
        pol = Policy(D, 16)
        batch = collate([_step(q, c, 2, 0.0)])
        with torch.no_grad():
            lg, _ = run_policy(pol, batch)
        old = torch.log_softmax(lg, -1)[:, 2]
        batch["action"] = torch.tensor([2])
        loss, st = clipped_loss(pol, batch, old, torch.tensor([sign]), None, 0.2, 0.0, 0.0)
        assert abs(st["ratio_mean"] - 1.0) < 1e-5 and st["clip_frac"] == 0.0
        loss.backward()
        with torch.no_grad():
            for p in pol.parameters():
                if p.grad is not None:
                    p -= 0.05 * p.grad  # one descent step on the loss
            lg2, _ = run_policy(pol, batch)
        after = torch.log_softmax(lg2, -1)[0, 2].item()
        assert (after - old.item()) * sign > 0  # +adv raises logp(action), -adv lowers it


def test_chain_task_two_hop_path_accuracy():
    """Synthetic 2-hop (player->team) templates: greedy policy follows the gold path >= 90% within CPU budget."""
    from tablehop.rl.runner import run

    cfg = dict(get_config("configs/smoke_h1.yaml"))
    cfg = get_config("configs/smoke_h1.yaml", [
        "data.synthetic_n.train=300", "rl.budget_trajectories=12800", "rl.eval_every=1000", "rl.eval_subset=40",
        "paths.results_dir=/tmp/_t08_tests", "env.max_hops=2"])
    from tablehop.data import synthetic
    orig = synthetic.TEMPLATES
    synthetic.TEMPLATES = ("2hop",)
    try:
        run_dir = run(dict(cfg), 0, "ppo")
        pol_ck = torch.load(run_dir / "checkpoint.pt", weights_only=False)
        from tablehop.rl.runner import make_policy
        pol = make_policy(cfg, pol_ck["d_emb"])
        pol.load_state_dict(pol_ck["policy"])
        cache, head = build_cache_head(cfg)
        dev = load_split(cfg["data"], "dev")[:100]
        recs = evaluate_policy(pol, dev, EnvPool(cache, cfg, 128), head, RewardFn(cfg))
        acc = np.mean([r["path"] == r["gold_path"] for r in recs])
    finally:
        synthetic.TEMPLATES = orig
    assert acc >= 0.9, acc


def test_rollout_no_illegal_and_stub_head():
    cfg = get_config("configs/smoke_h1.yaml")
    cache, head = build_cache_head(cfg)
    assert isinstance(head, StubSyntheticHead)
    pol = Policy(int(cfg["rl"]["stub_dim"]), 16)
    exs = load_split(cfg["data"], "dev")[:12]
    trajs = rollout(pol, exs, EnvPool(cache, cfg, 16), head, RewardFn(cfg))
    for t in trajs:
        assert all(h.legal for h in t.infos) and len(t.infos) >= 1 and t.steps[0]["mask"][t.steps[0]["action"]]


def test_checkpoint_resume():
    from tablehop.rl.runner import run

    ov = ["data.synthetic_n.train=64", "rl.eval_every=5", "rl.eval_subset=8", "paths.results_dir=/tmp/_t08_tests"]
    cfg = dict(get_config("configs/smoke_h1.yaml", [*ov, "rl.budget_trajectories=640"]))
    d = run(cfg, 0, "ppo")
    cfg2 = dict(get_config("configs/smoke_h1.yaml", [*ov, "rl.budget_trajectories=1280"]))
    cfg2["resume"] = str(d)
    d2 = run(cfg2, 0, "ppo")
    assert d2 == d
    ck = torch.load(d / "checkpoint.pt", weights_only=False)
    assert ck["update"] == 20 and ck["seen"] == 1280 and len(ck["rows"]) == 20
