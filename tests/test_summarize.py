import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
HASH = "a" * 64


def fake_run(rd: Path, label: str, seed: int, acc: float, split="dev", env=HASH, status="done"):
    d = rd / "runs" / f"{label}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"run_id": d.name, "method": label, "seed": seed,
                                             "config": {}, "status": status}))
    n = 20
    preds = []
    for i in range(n):
        ok = i < int(acc * n)
        preds.append({"id": f"q{i}", "question": "q", "pred": "yes" if ok else "no", "gold": ["yes"],
                      "path": [], "gold_path": [], "trace_info": [{"legal": True, "redundant": False}]})
    (d / f"predictions_{split}.jsonl").write_text("\n".join(json.dumps(p) for p in preds))
    (d / f"metrics_{split}.json").write_text(json.dumps({
        "em": acc, "f1": acc, "path_validity_strict": 0.5, "hop_efficiency": 0.4, "mean_hops": 2.0,
        "n": n, "env_lock_sha256": env}))


def run_summ(rd: Path):
    return subprocess.run([PY, "scripts/summarize.py", "--results_dir", str(rd)], cwd=ROOT,
                          capture_output=True, text=True, check=False)


def test_summarize_fake(tmp_path):
    for s, (a, b) in enumerate([(0.8, 0.5), (0.75, 0.55)]):
        fake_run(tmp_path, "h1_hoppo", s, a)
        fake_run(tmp_path, "b1_sft", s, b)
        fake_run(tmp_path, "h1_hoppo", s, a - 0.2, split="ood_dev")
    fake_run(tmp_path, "b2a_grpo", 0, 0.9, status="running")  # not done -> ignored
    r = run_summ(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "paired h1_hoppo - b1_sft" in r.stdout and "b2a_grpo" not in r.stdout
    assert "no finished runs for b2a_grpo" in r.stderr  # skipped with a warning
    t = tmp_path / "tables"
    for n in ("main", "ablations", "ood", "paired"):
        assert (t / f"{n}.csv").exists() and (t / f"{n}.tex").exists()
    assert "\\newcommand{\\EmH" in (t / "macros.tex").read_text()
    assert "dev->ood_dev" in r.stdout


def test_summarize_env_mismatch(tmp_path):
    fake_run(tmp_path, "h1_hoppo", 0, 0.8)
    fake_run(tmp_path, "b1_sft", 0, 0.5, env="b" * 64)
    r = run_summ(tmp_path)
    assert r.returncode != 0 and "env_lock_sha256" in (r.stderr + r.stdout)


@pytest.mark.parametrize("script", ["make_figures"])
def test_figures_fake(tmp_path, script):
    fake_run(tmp_path, "h1_hoppo", 0, 0.8)
    fake_run(tmp_path, "b1_sft", 0, 0.5)
    r = subprocess.run([PY, f"scripts/{script}.py", "--results_dir", str(tmp_path)], cwd=ROOT,
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "figures" / "F2_main.png").exists()
    assert (tmp_path / "figures" / "F2_main.pdf").exists()
