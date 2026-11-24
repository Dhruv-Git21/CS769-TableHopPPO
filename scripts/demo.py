"""Inference demo: hop-by-hop trace of a trained Hop-PPO policy on one question (display only).

Mode A: --split dev|test|ood_test --id <id>   or   --random N --seed S
Mode B: --table t.csv|.json --question "..." [--passages passages.json]
Mode C: --ask <example id> [--split dev] --question "..."  : a NEW question over that example's table and passages
        (no gold answer; shows the retrieved evidence and the answer). Without --question: interactive prompt.
Reads finished runs only; writes nothing to TEST_LOG. Never use it to select models or tune anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from dataclasses import replace

from tablehop.data.hybridqa import infer_dtype
from tablehop.data.load import load_split
from tablehop.data.schema import Cell, Example, Table
from tablehop.env.render import passage_title, render_evidence
from tablehop.eval.trace_viz import render_figure, trace_example
from tablehop.rl.reward import RewardFn
from tablehop.rl.rollout import EnvPool
from tablehop.rl.runner import build_cache_head, make_policy


def newest_h1(results_dir: Path) -> Path:
    runs = []
    for mp in sorted(results_dir.glob("runs/h1_hoppo_*/meta.json")):
        if json.loads(mp.read_text()).get("status") == "done":
            runs.append(mp.parent)
    if not runs:
        raise SystemExit("no finished h1_hoppo run under results/runs; pass --run")
    return runs[-1]


def load_custom(table_path: str, question: str, passages_path: str | None) -> Example:
    """Build an Example from a csv (first row = header) or json ({title, header, rows} / list of rows)."""
    p = Path(table_path)
    title, raw = p.stem, []
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        if isinstance(d, dict):
            title, raw = d.get("title", title), [d["header"], *d["rows"]]
        else:
            raw = d
    else:
        with open(p, newline="") as f:
            raw = list(csv.reader(f))
    passages = json.loads(Path(passages_path).read_text()) if passages_path else {}
    by_title = {passage_title(k).lower(): k for k in passages}
    header = [str(h) for h in raw[0]]
    rows = []
    for r, row in enumerate(raw[1:]):
        out = []
        for c, v in enumerate(row):
            text, link = (v.get("text", ""), v.get("link")) if isinstance(v, dict) else (str(v), None)
            link = link or by_title.get(text.strip().lower())
            out.append(Cell(text, r, c, link, infer_dtype(text)))
        rows.append(out)
    tab = Table("custom", title, "", header, rows)
    return Example("custom", "custom", "custom", question, tab, passages, [], "none")


def ask_example(base: Example, question: str, k: int = 0) -> Example:
    """Same table and passages as `base`, a user question, no gold answer or path."""
    return replace(base, id=f"{base.id}:ask{k}", question=question, answer=[], gold_path=None,
                   gold_path_quality="none")


def find_example(cfg: dict, split: str, ex_id: str) -> Example:
    sel = [e for e in load_split(cfg["data"], split) if e.id == ex_id]
    if not sel:
        raise SystemExit(f"id {ex_id} not in split {split}")
    return sel[0]


def pick_examples(args, cfg: dict) -> list[Example]:
    if args.table:
        return [load_custom(args.table, args.question, args.passages)]
    if args.ask:
        return [ask_example(find_example(cfg, args.split, args.ask), args.question)]
    exs = load_split(cfg["data"], args.split)
    if args.id:
        sel = [e for e in exs if e.id == args.id]
        if not sel:
            raise SystemExit(f"id {args.id} not in split {args.split}")
        return sel
    return random.Random(args.seed).sample(exs, min(args.random, len(exs)))


def load_policy(run_dir: Path):
    meta = json.loads((run_dir / "meta.json").read_text())
    cfg = meta["config"]
    cache, head = build_cache_head(cfg)
    ck = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    pol = make_policy(cfg, ck["d_emb"])
    pol.load_state_dict(ck["policy"])
    return cfg, pol, EnvPool(cache, cfg, 8), head, RewardFn(cfg)


def compare_one(run_dir: Path | None, ex: Example, m0_head=None) -> dict:
    """One baseline answer for `ex` using scripts/evaluate.py predictors (or m0 when run_dir is None)."""
    if run_dir is None:
        gp = list(ex.gold_path or [])
        pred = m0_head.answer([(ex.id, ex.question, render_evidence(gp, ex, getattr(m0_head, "evidence_mode", "entity")))])[0]
        recs, name = [{"pred": pred, "path": gp}], "m0_oracle"
    else:
        import evaluate as ev

        meta = json.loads((run_dir / "meta.json").read_text())
        cfg, name = meta["config"], meta["method"]
        if name == "b3_soft":
            recs, _ = ev.predict_b3(run_dir, cfg, [ex])
        elif name in ev.POLICY_METHODS:
            recs, _ = ev.predict_policy(run_dir, cfg, [ex])
        else:
            recs, _ = ev.predict_b1(run_dir, cfg, [ex])
    r = recs[0]
    from tablehop.eval.metrics import exact_match, f1
    return {"method": name, "run": run_dir.name if run_dir else name, "pred": r["pred"], "hops": len(r["path"]),
            "em": exact_match(r["pred"], ex.answer) if ex.answer else None,
            "f1": f1(r["pred"], ex.answer) if ex.answer else None}


def show_table(ex: Example, gold_cells: set | None = None, max_rows: int = 8) -> None:
    gold_cells = gold_cells or set()
    print(f"Table: {ex.table.title}  ({len(ex.table.rows)} rows, {len(ex.passages)} linked passages)")
    print(" | ".join(ex.table.header))
    for r, row in enumerate(ex.table.rows[:max_rows]):
        print(" | ".join(f"*{c.text}*" if (r, k) in gold_cells else c.text for k, c in enumerate(row)))
    if len(ex.table.rows) > max_rows:
        print(f"... ({len(ex.table.rows) - max_rows} more rows)")
    if gold_cells:
        print("(* = gold cell)")


def show(ex: Example, tr: dict, comps: list[dict], table: bool = True) -> None:
    out = print
    gold_cells = {(e["row"], e["col"]) for e in (tr["gold_path"] or []) if e["kind"] == "cell"}
    out(f"\nQuestion [{ex.id}]: {ex.question}")
    if table:
        show_table(ex, gold_cells)
    for h in tr["hops"]:
        txt = "STOP" if h["entity"] is None else (h["text"][:160] + ("..." if len(h["text"]) > 160 else ""))
        out(f"\nHop {h['hop']}: {txt}  [{h['relation']}]  p={h['prob']:.3f}")
        out("  alternatives: " + "; ".join(f"{short(a['text'])} ({a['prob']:.3f})" for a in h["alternatives"]))
        if "reward" in h:
            rw = h["reward"]
            out(f"  reward: hop {rw['hop']:+.2f} redundancy {rw['redundancy']:+.2f} "
                f"schema {rw['schema']:+.2f} step {rw['step']:+.2f}")
    out("\nEvidence read by the answer model (from the selected entities):")
    for i, e in enumerate(tr.get("evidence") or [], 1):
        out(f"  [{i}] {e[:300]}{'...' if len(e) > 300 else ''}")
    gold = " / ".join(tr["gold"]) if tr["gold"] else "n/a"
    score = f"  EM {tr['em']:.0f} F1 {tr['f1']:.2f}" if tr["f1"] is not None else ""
    out(f"\nANSWER: {tr['pred']!r}   (gold: {gold}{score})")
    for c in comps:
        s = f" EM {c['em']:.0f} F1 {c['f1']:.2f}" if c["f1"] is not None else ""
        out(f"  {c['method']:<14} {c['pred']!r} hops={c['hops']}{s}")


def short(text: str | None) -> str:
    """Compact label for a rendered entity: 'header = cell' for cells, 'passage: Title' for passages."""
    if not text:
        return "STOP"
    parts = text.split(" | ")
    if len(parts) >= 4:
        return f"{parts[1]} = {parts[-1]}"[:60]
    return ("passage: " + text.split(" : ", 1)[0])[:60]


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--m0", action="store_true")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "ood_dev", "ood_test"])
    ap.add_argument("--id")
    ap.add_argument("--random", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--table")
    ap.add_argument("--ask", help="example id whose table/passages are used for a new --question")
    ap.add_argument("--question")
    ap.add_argument("--passages")
    ap.add_argument("--save")
    ap.add_argument("--results_dir", default="results")
    a = ap.parse_args(argv)
    if not (a.table or a.id or a.random or a.ask):
        raise SystemExit("give --table/--question, --ask ID [--question], --id, or --random N")
    if a.table and not a.question:
        raise SystemExit("--table needs --question")
    run_dir = Path(a.run) if a.run else newest_h1(Path(a.results_dir))
    cfg, pol, pool, head, rfn = load_policy(run_dir)
    print(f"model: {run_dir.name}")
    if a.ask and not a.question:  # interactive: many questions over one table, model loaded once
        base = find_example(cfg, a.split, a.ask)
        show_table(base)
        for k in range(1, 1000):
            try:
                q = input("\nAsk a question about this table (empty line to quit): ").strip()
            except EOFError:
                break
            if not q:
                break
            ex = ask_example(base, q, k)
            show(ex, trace_example(pol, ex, pool, head, rfn), [], table=False)
        return []
    traces = []
    for ex in pick_examples(a, cfg):
        tr = trace_example(pol, ex, pool, head, rfn)
        tr["run"] = run_dir.name
        comps = [compare_one(Path(c), ex, None) for c in a.compare]
        if a.m0:
            comps.append(compare_one(None, ex, head))
        tr["compare"] = comps
        show(ex, tr, comps)
        if a.save:
            d = Path(a.save) / ex.id.replace(":", "_").replace("/", "_")
            d.mkdir(parents=True, exist_ok=True)
            (d / "trace.json").write_text(json.dumps(tr, indent=2, ensure_ascii=False))
            tag = d.name
            render_figure(ex, tr, d / f"case_{tag}")
            print(f"saved {d}")
        traces.append(tr)
    return traces


if __name__ == "__main__":
    main()
