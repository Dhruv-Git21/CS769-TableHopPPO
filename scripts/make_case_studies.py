"""Case studies for figure F8 from DEV predictions only (fixed rules; never test).

usage: python scripts/make_case_studies.py [--run results/runs/h1_hoppo_...] [--results_dir results]
Rules over predictions_dev.jsonl of the run, ids sorted ascending: up to 3 successes (F1 = 1 and the selected
entity set equals the gold set), 3 failures (F1 = 0), 2 reward-hack-like (STOP after <= 1 hop, or a repeated
entity), 2 wrong-answer-on-gold-path (full gold set visited but F1 < 1). Runs demo.main --save for each, copies
figures to results/figures/F8_case_<k>_<kind>.{png,pdf}, writes results/case_studies.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import demo

from tablehop.eval.metrics import f1

RULES = (("success", 3), ("failure", 3), ("hack_like", 2), ("answer_head_error", 2))


def key(e: dict) -> tuple:
    return (e["kind"], e["row"], e["col"], e.get("passage_id"))


def classify(r: dict) -> str | None:
    path, gold = [key(e) for e in r["path"]], {key(e) for e in r["gold_path"] or []}
    score = f1(r["pred"], r["gold"]) if r["gold"] else 0.0
    if len(path) <= 1 or len(set(path)) < len(path):
        return "hack_like"
    if gold and score == 1.0 and set(path) == gold:
        return "success"
    if gold and gold <= set(path) and score < 1.0:
        return "answer_head_error"
    if score == 0.0:
        return "failure"
    return None


def select(recs: list[dict]) -> list[tuple[str, str]]:
    by: dict[str, list[str]] = {k: [] for k, _ in RULES}
    for r in sorted(recs, key=lambda r: r["id"]):
        k = classify(r)
        if k:
            by[k].append(r["id"])
    return [(k, i) for k, n in RULES for i in by[k][:n]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--results_dir", default="results")
    a = ap.parse_args()
    rd = Path(a.results_dir)
    run = Path(a.run) if a.run else demo.newest_h1(rd)
    pf = run / "predictions_dev.jsonl"
    if not pf.exists():
        raise SystemExit(f"{pf} missing: run scripts/evaluate.py --run {run} --split dev first")
    recs = [json.loads(line) for line in pf.open() if line.strip()]
    lines = [f"# Case studies (dev, run {run.name})\n",
             "Selection: fixed rules over dev predictions (see scripts/make_case_studies.py). Display only.\n"]
    (rd / "figures").mkdir(parents=True, exist_ok=True)
    for k, (kind, eid) in enumerate(select(recs), 1):
        tr = demo.main(["--run", str(run), "--split", "dev", "--id", eid, "--save", str(rd / "demo"),
                        "--results_dir", a.results_dir])[0]
        src = rd / "demo" / eid.replace(":", "_").replace("/", "_")
        for ext in ("png", "pdf"):
            shutil.copy(src / f"case_{src.name}.{ext}", rd / "figures" / f"F8_case_{k}_{kind}.{ext}")
        lines += [f"## {k}. {kind}: {eid}", f"- Question: {tr['question']}",
                  f"- Predicted: {tr['pred']!r}; gold: {tr['gold']}; hops: {len(tr['path'])}",
                  f"- Figure: figures/F8_case_{k}_{kind}.png\n"]
    (rd / "case_studies.md").write_text("\n".join(lines))
    print(f"wrote {rd / 'case_studies.md'}")


if __name__ == "__main__":
    main()
