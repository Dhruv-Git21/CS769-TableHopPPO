"""Demo (mode A and B) on synthetic data with hashed encoder, stub head and an untrained policy."""
import importlib.util
import json
import sys
from pathlib import Path

import torch

from tablehop.rl.runner import make_policy
from tablehop.utils.config import get_config

ROOT = Path(__file__).resolve().parents[1]


def _demo():
    spec = importlib.util.spec_from_file_location("demo_mod", ROOT / "scripts" / "demo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["demo_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_run(tmp_path: Path) -> Path:
    cfg = get_config(str(ROOT / "configs" / "smoke_h1.yaml"))
    cfg["data"]["synthetic_n"]["dev"] = 8
    run = tmp_path / "runs" / "h1_hoppo_fake"
    run.mkdir(parents=True)
    torch.manual_seed(0)
    pol = make_policy(cfg, int(cfg["rl"]["stub_dim"]))
    torch.save({"policy": pol.state_dict(), "d_emb": int(cfg["rl"]["stub_dim"])}, run / "checkpoint.pt")
    (run / "meta.json").write_text(json.dumps({"config": cfg, "method": "h1_hoppo", "status": "done"}, default=str))
    return run


def test_mode_a(tmp_path):
    demo = _demo()
    run = _fake_run(tmp_path)
    assert demo.newest_h1(tmp_path) == run
    out = tmp_path / "out"
    trs = demo.main(["--run", str(run), "--split", "dev", "--random", "2", "--seed", "1", "--m0",
                     "--save", str(out)])
    assert len(trs) == 2
    tr = trs[0]
    d = out / tr["id"].replace(":", "_")
    t = json.loads((d / "trace.json").read_text())
    for k in ("question", "hops", "pred", "gold", "em", "f1", "gold_path", "compare"):
        assert k in t
    h = t["hops"][0]
    assert {"hop", "text", "relation", "prob", "alternatives", "reward"} <= set(h)
    assert len(h["alternatives"]) <= 3 and 0 <= h["prob"] <= 1
    assert h["relation"] == "start"
    assert (d / f"case_{d.name}.png").stat().st_size > 0 and (d / f"case_{d.name}.pdf").exists()
    assert t["compare"][0]["method"] == "m0_oracle"


def test_mode_b(tmp_path):
    demo = _demo()
    run = _fake_run(tmp_path)
    csv = tmp_path / "t.csv"
    csv.write_text("Player,Team,City\nAlo Ber,Reds FC,Paris\nCid Dor,Blues FC,Rome\n")
    pas = tmp_path / "p.json"
    pas.write_text(json.dumps({"/wiki/Reds_FC": "Reds FC is a club founded in 1950."}))
    out = tmp_path / "out"
    trs = demo.main(["--run", str(run), "--table", str(csv), "--question", "Which city is Alo Ber's team in?",
                     "--passages", str(pas), "--save", str(out)])
    t = trs[0]
    assert t["gold"] is None and t["em"] is None and t["gold_path"] is None
    assert "reward" not in t["hops"][0]
    assert (out / "custom" / "case_custom.png").exists()


def test_case_selection():
    spec = importlib.util.spec_from_file_location("mcs", ROOT / "scripts" / "make_case_studies.py")
    sys.path.insert(0, str(ROOT / "scripts"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    c = lambda r: {"kind": "cell", "row": r, "col": 0, "passage_id": None}
    base = {"gold": ["x"], "gold_path": [c(0), c(1)]}
    assert mod.classify({**base, "pred": "x", "path": [c(0), c(1)]}) == "success"
    assert mod.classify({**base, "pred": "y", "path": [c(0), c(1)]}) == "answer_head_error"
    assert mod.classify({**base, "pred": "y", "path": [c(2), c(3)]}) == "failure"
    assert mod.classify({**base, "pred": "y", "path": [c(0)]}) == "hack_like"
    assert mod.classify({**base, "pred": "y", "path": [c(0), c(0)]}) == "hack_like"
