"""Hop-by-hop greedy trace of a policy on one example, plus the case-study figure (display only).

trace_example -> dict with question, hops (entity, text, relation, prob, alternatives, reward parts), pred,
gold, em, f1. render_figure draws the table grid with selected cells (orange, numbered), gold cells (teal
outline), arrows between hops and the linked passage as a side box. Never used for model selection.
"""
from __future__ import annotations

import itertools
import textwrap
from pathlib import Path

import torch

from tablehop.data.schema import EntityRef, Example
from tablehop.env.candidates import REL
from tablehop.env.render import render_entity, render_evidence
from tablehop.eval.metrics import exact_match, f1
from tablehop.models.policy import collate, run_policy

REL_NAME = {v: k for k, v in REL.items()}
ORANGE, LORANGE, TEAL, NAVY, CREAM, CARD, BORDER, TEXT = (
    "#B4501A", "#F2B48A", "#1F6F8B", "#14213D", "#F6F5F0", "#FBFAF6", "#DAD7CC", "#3D4757")


def ref_dict(r: EntityRef | None) -> dict | None:
    return None if r is None else {"kind": r.kind, "row": r.row, "col": r.col, "passage_id": r.passage_id}


def trace_example(policy, ex: Example, pool, head, rfn) -> dict:
    """Greedy rollout with per-hop probabilities; reward parts only when a gold path exists."""
    env = pool.make(ex)
    st, hops, done, w = env._state(), [], False, rfn.w
    gold = bool(ex.gold_path) and ex.gold_path_quality != "none"
    policy.eval()
    while not done:
        step = {"q": st.q_emb, "trace": st.trace_embs, "cands": st.cand_embs, "rel": st.rel_ids, "mask": st.mask, "feats": st.cand_feats}
        with torch.no_grad():
            logits, _ = run_policy(policy, collate([step]))
        p = torch.softmax(logits[0, : len(st.cand_refs)], -1)
        order = torch.argsort(-p).tolist()
        a = order[0]
        alts = [{"entity": ref_dict(st.cand_refs[i]), "text": _txt(st.cand_refs[i], ex), "prob": float(p[i])}
                for i in order[1:4]]
        rel = REL_NAME[int(st.rel_ids[a])]
        st, info, done = env.step(a)
        hop = {"hop": len(hops), "entity": ref_dict(info.entity), "text": _txt(info.entity, ex),
               "relation": rel, "prob": float(p[a]), "alternatives": alts, "stop": info.stop and info.legal}
        if gold and info.entity is not None:
            hop["reward"] = {"hop": w["w_hop"] * float(info.on_gold), "redundancy": -w["w_red"] * float(info.redundant),
                             "schema": 0.0, "step": -w["w_step"], "on_gold": info.on_gold}
        hops.append(hop)
    infos = [h for h in env.trace]
    if gold and infos and not infos[-1].schema_ok:
        sel = [h for h in hops if h["entity"] is not None]
        sel[-1]["reward"]["schema"] = -w["w_schema"]
    ents = env.entities
    evidence = render_evidence(ents, ex, getattr(head, "evidence_mode", "entity"))
    pred = head.answer([(ex.id, ex.question, evidence)])[0]
    return {"id": ex.id, "question": ex.question, "hops": hops, "path": [ref_dict(e) for e in ents],
            "evidence": evidence, "pred": pred, "gold": list(ex.answer) if ex.answer else None,
            "gold_path": [ref_dict(e) for e in ex.gold_path] if gold else None,
            "em": exact_match(pred, ex.answer) if ex.answer else None,
            "f1": f1(pred, ex.answer) if ex.answer else None}


def _txt(r: EntityRef | None, ex: Example) -> str:
    return "STOP" if r is None else render_entity(r, ex)


def _wrap(s: str, n: int, lines: int = 2) -> str:
    out = textwrap.wrap(s, n) or [""]
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1][: n - 1] + "..."
    return "\n".join(out)


def render_figure(ex: Example, trace: dict, out_base: Path, max_rows: int = 12) -> list[Path]:
    """Write <out_base>.png and .pdf (200 dpi); returns the paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    sel = [h["entity"] for h in trace["hops"] if h["entity"] is not None]
    gold = trace["gold_path"] or []
    cells = {(e["row"], e["col"]) for e in sel + gold}
    nr, nc = len(ex.table.rows), len(ex.table.header)
    rows = sorted({r for r, _ in cells} | set(range(min(nr, max_rows))))[: max(max_rows, len({r for r, _ in cells}))]
    rpos = {r: i for i, r in enumerate(rows)}
    pas = [e for e in sel if e["kind"] == "passage"]
    cw, ch = 1.9, 0.62
    fig_w = nc * cw + (3.2 if pas else 0.2)
    fig, ax = plt.subplots(figsize=(fig_w, (len(rows) + 1) * ch + 1.6), facecolor=CREAM)
    ax.set_xlim(0, nc * cw + (3.2 if pas else 0.2))
    ax.set_ylim(-(len(rows) + 1) * ch - 0.1, 0.6)
    ax.axis("off")
    for c, h in enumerate(ex.table.header):
        ax.add_patch(Rectangle((c * cw, -ch), cw, ch, fc=NAVY, ec=BORDER))
        ax.text(c * cw + cw / 2, -ch / 2, _wrap(h, 22, 1), ha="center", va="center", color="white", fontsize=7)
    centre = {}
    for r in rows:
        y0 = -(rpos[r] + 2) * ch
        for c in range(nc):
            on_sel = [i for i, e in enumerate(sel) if e["kind"] == "cell" and (e["row"], e["col"]) == (r, c)]
            fc = LORANGE if on_sel else (CARD if rpos[r] % 2 == 0 else "#ECEAE0")
            is_gold = any(e["kind"] == "cell" and (e["row"], e["col"]) == (r, c) for e in gold)
            ax.add_patch(Rectangle((c * cw, y0), cw, ch, fc=fc, ec=TEAL if is_gold else BORDER,
                                   lw=2.4 if is_gold else 0.6, zorder=3 if is_gold else 1))
            ax.text(c * cw + cw / 2, y0 + ch / 2, _wrap(ex.table.rows[r][c].text, 24), ha="center", va="center",
                    fontsize=6.5, color=TEXT, zorder=4)
            if on_sel:
                ax.text(c * cw + 0.08, y0 + ch - 0.07, str(on_sel[0] + 1), ha="left", va="top", fontsize=8,
                        fontweight="bold", color=ORANGE, zorder=5)
            centre[(r, c)] = (c * cw + cw / 2, y0 + ch / 2)
    pos = []
    for i, e in enumerate(sel):
        if e["kind"] == "cell":
            pos.append(centre[(e["row"], e["col"])])
        else:
            bx, by = nc * cw + 0.15, -2 * ch
            ax.add_patch(Rectangle((bx, by - 2.2), 2.9, 2.2 + ch, fc=CARD, ec=ORANGE, lw=1.5))
            title = e["passage_id"].rsplit("/", 1)[-1].replace("_", " ")
            body = _wrap(ex.passages.get(e["passage_id"], ""), 44, 9)
            ax.text(bx + 0.1, by + ch - 0.08, f"{i + 1}. {title}", fontsize=7.5, fontweight="bold", color=ORANGE, va="top")
            ax.text(bx + 0.1, by + ch - 0.45, body, fontsize=6.3, color=TEXT, va="top")
            pos.append((bx, by + ch / 2))
    for a, b in itertools.pairwise(pos):
        if a != b:
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=ORANGE, lw=1.6,
                                         connectionstyle="arc3,rad=0.2", zorder=6))
    gold_ans = " / ".join(trace["gold"]) if trace["gold"] else "n/a (custom input)"
    cap = f"Q: {trace['question']}\nPredicted: {trace['pred']!r}    Gold: {gold_ans}"
    if trace["f1"] is not None:
        cap += f"    EM {trace['em']:.0f}  F1 {trace['f1']:.2f}"
    fig.text(0.01, 0.01, "\n".join(textwrap.wrap(cap.replace("\n", "  ||  "), 150)).replace("  ||  ", "\n"),
             fontsize=8, color=NAVY, va="bottom")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.14)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=200, facecolor=CREAM)
        paths.append(p)
    plt.close(fig)
    return paths
