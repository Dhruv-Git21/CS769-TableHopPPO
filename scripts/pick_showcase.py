"""Pick demo questions from DEV predictions of a finished H1 run (never from test). Display only.

usage: python scripts/pick_showcase.py [--run results/runs/<h1 run>] [--b1 results/runs/<b1 run>] [--n 8]
Writes results/demo/showcase.json: categories
  "h1_wins"    H1 exact match via a table->passage hop, B1 (SFT) wrong
  "both_right" H1 and B1 both exact
  "h1_fails"   H1 wrong although the oracle path answers it (an honest failure case)
Selection is deterministic (sorted by id, first n per category).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tablehop.eval.metrics import exact_match


def load_preds(run: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, (run / "predictions_dev.jsonl").read_text().splitlines()) if r}


def newest(pattern: str) -> Path:
    c = sorted(Path("results/runs").glob(pattern))
    if not c:
        raise SystemExit(f"no run matching {pattern}")
    return c[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--b1")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="results/demo/showcase.json")
    a = ap.parse_args()
    h1 = Path(a.run) if a.run else newest("h1_hoppo_hybridqa_s0_*")
    b1 = Path(a.b1) if a.b1 else newest("b1_sft_hybridqa_s0_*")
    m0 = Path("results/runs/m0_oracle")
    H, B = load_preds(h1), load_preds(b1)
    M = load_preds(m0) if (m0 / "predictions_dev.jsonl").exists() else {}
    cats: dict[str, list] = {"h1_wins": [], "both_right": [], "h1_fails": []}
    for qid in sorted(H):
        h = H[qid]
        he = exact_match(h["pred"], h["gold"])
        be = exact_match(B[qid]["pred"], h["gold"]) if qid in B else None
        me = exact_match(M[qid]["pred"], h["gold"]) if qid in M else None
        passage_hop = any(p.get("kind") == "passage" for p in h["path"])
        item = {"id": qid, "question": h["question"], "gold": h["gold"], "h1_pred": h["pred"],
                "b1_pred": B.get(qid, {}).get("pred"), "m0_pred": M.get(qid, {}).get("pred")}
        if he and passage_hop and be == 0:
            cats["h1_wins"].append(item)
        elif he and be == 1:
            cats["both_right"].append(item)
        elif not he and me == 1:
            cats["h1_fails"].append(item)
    out = {"h1_run": h1.name, "b1_run": b1.name, "split": "dev",
           "counts": {k: len(v) for k, v in cats.items()}, "n_dev": len(H),
           **{k: v[: a.n] for k, v in cats.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out["counts"]), "->", a.out)
    for k in cats:
        for it in out[k][:4]:
            print(f"[{k}] {it['id']}: {it['question'][:90]} | gold {it['gold']} | H1 {it['h1_pred']!r} | B1 {it['b1_pred']!r}")


if __name__ == "__main__":
    main()
