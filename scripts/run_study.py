"""Tuning (T16) -> final runs x seeds (T17) -> dev/test/OOD evaluation (T18/T19). Resumable.

usage: python scripts/run_study.py [--phase tune|final|eval|all] [--lanes 2]
State: results/study/state.json (trial -> run dir, dev metrics, selection). Every trained method gets
the same number of trials (3, fixed grid in TRIALS, dev seed 0); ablations inherit H1's choice.
Selection: full-dev EM, ties -> F1. Non-selected trials are moved to results/runs_archive/ so that
summarize.py only sees final runs. Test/OOD splits are evaluated once per final run (evaluate.py guard).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
STATE = ROOT / "results/study/state.json"
RUNS = ROOT / "results/runs"
ARCHIVE = ROOT / "results/runs_archive"
LOGS = ROOT / "logs/study"
SEEDS = [0, 1, 2]
RL_GRID = [[], ["ppo.lr_head=1.0e-3"], ["ppo.entropy_coef=0.03"]]
TRIALS = {  # method -> list of override lists (trial 0 = spec default)
    "h1_hoppo": RL_GRID, "b2a_grpo": RL_GRID, "b2b_grpo_path": RL_GRID,
    "b3_soft": [[], ["soft.lr=3.0e-4"], ["soft.lr=3.0e-3"]],
    "b1_sft": [[], ["sft.lr=1.0e-4"], ["sft.lr=1.0e-3"]],
}
ABLATIONS = ["a1_terminal", "a2_noredund", "a3_noschema", "a4_unrestricted"]
EVAL_SPLITS = ["dev", "test", "ood_dev", "ood_test"]
LOCK = threading.Lock()


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"trials": {}, "chosen": {}, "final": {}}


def save(st: dict) -> None:
    with LOCK:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(st, indent=2))


def sh(args: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(args)}  [{time.strftime('%F %T')}]\n")
        f.flush()
        return subprocess.call(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)


def run_dir_from_log(log: Path) -> Path | None:
    """Last `run_dir: ...` line printed by train.py (safe with concurrent runs of one method)."""
    lines = [ln for ln in log.read_text().splitlines() if ln.startswith("run_dir: ")]
    return ROOT / lines[-1].split("run_dir: ", 1)[1].strip() if lines else None


def train(method: str, seed: int, overrides: list[str], tag: str) -> str | None:
    log = LOGS / f"{tag}.log"
    t0 = time.time()
    cmd = [PY, "scripts/train.py", "--config", f"configs/{method}.yaml", "--seed", str(seed)]
    if overrides:
        cmd += ["--override", *overrides]
    if sh(cmd, log) != 0:
        print(f"[study] FAILED train {tag}", flush=True)
        return None
    rd = run_dir_from_log(log)
    if rd is None or not rd.exists():
        return None
    rd = rd.resolve() if not str(rd).startswith(str(ROOT)) else rd
    sh([PY, "scripts/evaluate.py", "--run", str(rd), "--split", "dev"], log)
    print(f"[study] {tag} -> {rd.name} ({time.time() - t0:.0f}s) dev {dev_metrics(rd)}", flush=True)
    return str(rd.relative_to(ROOT))


def dev_metrics(rd: Path) -> dict | None:
    f = Path(rd) / "metrics_dev.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text())
    return {"em": m["em"], "f1": m["f1"]}


def run_lanes(jobs: list, lanes: int) -> None:
    """jobs: list of (lane_key, fn). Jobs with lane 'mps' run serially in their own lane; others share
    `lanes - 1` (min 1) worker threads."""
    if lanes <= 1:  # strictly sequential (parallel runs thrash a 16 GB laptop: measured 5-6x slowdown)
        for _, f in jobs:
            f()
        return
    mps = [f for k, f in jobs if k == "mps"]
    cpu = [f for k, f in jobs if k != "mps"]

    def worker(q):
        while True:
            with LOCK:
                if not q:
                    return
                f = q.pop(0)
            f()

    ts = [threading.Thread(target=worker, args=(mps,))]
    ts += [threading.Thread(target=worker, args=(cpu,)) for _ in range(max(1, lanes - 1))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def phase_tune(st: dict, lanes: int) -> None:
    jobs = []
    for m, grid in TRIALS.items():
        for i, ov in enumerate(grid):
            key = f"{m}/t{i}"
            if st["trials"].get(key, {}).get("run"):
                continue

            def job(m=m, i=i, ov=ov, key=key):
                rd = train(m, 0, ov, f"tune_{m}_t{i}")
                st["trials"][key] = {"overrides": ov, "run": rd, "dev": dev_metrics(ROOT / rd) if rd else None}
                save(st)
            jobs.append(("mps" if m == "b1_sft" else "cpu", job))
    run_lanes(jobs, lanes)
    for m, grid in TRIALS.items():
        res = [(k, v) for k, v in st["trials"].items() if k.startswith(m + "/") and v.get("dev")]
        if len(res) != len(grid):
            print(f"[study] {m}: only {len(res)}/{len(grid)} trials finished; no selection", flush=True)
            continue
        k, v = max(res, key=lambda kv: (kv[1]["dev"]["em"], kv[1]["dev"]["f1"]))
        st["chosen"][m] = {"trial": k, "overrides": v["overrides"], "run_s0": v["run"]}
        print(f"[study] chosen {m}: {k} {v['overrides']} dev {v['dev']}", flush=True)
    for a in ABLATIONS:
        if "h1_hoppo" in st["chosen"]:
            ov = st["chosen"]["h1_hoppo"]["overrides"]
            st["chosen"][a] = {"trial": "inherits h1_hoppo", "overrides": ov,
                               "run_s0": st.get("ablation_s0_default", {}).get(a) if not ov else None}
    save(st)


def phase_final(st: dict, lanes: int) -> None:
    jobs = []
    for m, ch in st["chosen"].items():
        for s in SEEDS:
            key = f"{m}/s{s}"
            if st["final"].get(key):
                continue
            if s == 0 and ch.get("run_s0"):
                st["final"][key] = ch["run_s0"]
                continue

            def job(m=m, s=s, key=key, ov=ch["overrides"]):
                st["final"][key] = train(m, s, ov, f"final_{m}_s{s}")
                save(st)
            jobs.append(("mps" if m == "b1_sft" else "cpu", job))
    save(st)
    run_lanes(jobs, lanes)
    # archive every run dir that is not a final run (tuning losers, pre-existing seed-0 runs)
    keep = {Path(v).name for v in st["final"].values() if v} | {"m0_oracle"}
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for p in RUNS.iterdir():
        if p.is_dir() and p.name not in keep:
            shutil.move(str(p), str(ARCHIVE / p.name))
            print(f"[study] archived {p.name}", flush=True)


def phase_eval(st: dict, lanes: int) -> None:
    jobs = []
    for split in EVAL_SPLITS:
        if not (RUNS / "m0_oracle" / f"metrics_{split}.json").exists():
            jobs.append(("cpu", lambda split=split: sh([PY, "scripts/evaluate.py", "--method", "m0", "--split",
                                                        split], LOGS / f"eval_m0_{split}.log")))
    for key, rd in st["final"].items():
        if not rd:
            continue
        for split in EVAL_SPLITS:
            if (ROOT / rd / f"metrics_{split}.json").exists():
                continue
            lane = "mps" if key.startswith("b1_sft") else "cpu"
            jobs.append((lane, lambda rd=rd, split=split, key=key: sh(
                [PY, "scripts/evaluate.py", "--run", rd, "--split", split],
                LOGS / f"eval_{key.replace('/', '_')}_{split}.log")))
    run_lanes(jobs, lanes)
    m0 = RUNS / "m0_oracle" / "meta.json"
    if not m0.exists():
        m0.write_text(json.dumps({"run_id": "m0_oracle", "method": "m0", "seed": 0, "status": "done",
                                  "config": {"exp": "m0"}, "note": "no training; frozen answer head on gold paths"}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["tune", "final", "eval", "all"])
    ap.add_argument("--lanes", type=int, default=1)
    a = ap.parse_args()
    st = load()
    if a.phase in ("tune", "all"):
        phase_tune(st, a.lanes)
    if a.phase in ("final", "all"):
        phase_final(st, a.lanes)
    if a.phase in ("eval", "all"):
        phase_eval(st, a.lanes)
    print("[study] done", time.strftime("%F %T"), flush=True)


if __name__ == "__main__":
    main()
