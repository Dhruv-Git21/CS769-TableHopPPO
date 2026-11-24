"""Results comparison table as an image (for forms/slides): results/figures/results_table.png.

usage: python scripts/make_results_image.py
Reads results/tables/main.csv and ablations.csv (written by scripts/summarize.py). Shows the split used,
seeds, method names, EM (with 95% CI), F1, strict path validity and mean hops. The best learned value in each
column is highlighted (bold, green); the oracle row is a reference ceiling and is excluded from "best".
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables"
NAMES = {"m0": "Oracle path (gold evidence, upper bound)", "b1_sft": "B1  Full-table reader (SFT)",
         "b2a_grpo": "B2a Answer-only RL (GRPO)", "b2b_grpo_path": "B2b Answer + path RL (GRPO)",
         "b3_soft": "B3  Soft selection", "h1_hoppo": "H1  Hop-PPO (ours)",
         "a1_terminal": "A1  Ours, answer reward only", "a2_noredund": "A2  Ours, no repeat penalty",
         "a3_noschema": "A3  Ours, no type check", "a4_unrestricted": "A4  Ours, any-cell moves"}


def num(s: str) -> float | None:
    m = re.match(r"\s*([0-9.]+)", s or "")
    return float(m.group(1)) if m else None


def main() -> None:
    rows = []
    for name in ("main", "ablations"):
        for r in csv.DictReader((T / f"{name}.csv").open()):
            rows.append(r)
    split = "test" if any(r["split"] == "test" for r in rows) else "dev"
    rows = [r for r in rows if r["split"] == split]
    seen, uniq = set(), []
    for r in rows:
        if r["method"] not in seen:
            seen.add(r["method"])
            uniq.append(r)
    order = [m for m in NAMES if m in seen]
    by = {r["method"]: r for r in uniq}
    cols = ["Exact match % ↑", "EM 95% CI", "F1 % ↑", "Path valid % ↑", "Avg hops"]
    table, vals = [], {c: {} for c in cols}
    for m in order:
        r = by[m]
        em, f1, pv, hops = num(r["EM mean±std"]), num(r["F1"]), num(r["strict validity"]), num(r["mean hops"])
        ci = re.findall(r"[0-9.]+", r["EM 95% CI"])
        cells = [f"{100 * em:.1f}", f"[{100 * float(ci[0]):.1f}, {100 * float(ci[1]):.1f}]" if ci else "n/a",
                 f"{100 * f1:.1f}", "n/a" if pv is None else f"{100 * pv:.1f}", "n/a" if hops is None else f"{hops:.2f}"]
        table.append([NAMES[m], *cells])
        if m != "m0":
            for c, v in zip(cols, (em, None, f1, pv, None)):
                if v is not None:
                    vals[c][m] = v
    best = {c: max(v, key=v.get) for c, v in vals.items() if v}
    seeds = by.get("h1_hoppo", uniq[0])["n_seeds"]
    fig, ax = plt.subplots(figsize=(13, 0.42 * (len(table) + 1) + 0.9), dpi=200)
    ax.axis("off")
    tb = ax.table(cellText=table, colLabels=["Method", *cols], cellLoc="center", bbox=[0, 0.06, 1, 0.94],
                  colWidths=[0.37, 0.13, 0.15, 0.1, 0.14, 0.11])
    tb.auto_set_font_size(False)
    tb.set_fontsize(10.5)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if i == 0:
            cell.set_facecolor("#14213D")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            continue
        m = order[i - 1]
        if j == 0:
            cell.get_text().set_ha("left")
            cell._loc = "left"
        if m == "m0":
            cell.set_facecolor("#F3F4F6")
            cell.get_text().set_style("italic")
        elif m == "h1_hoppo":
            cell.set_facecolor("#FCE7D8")
        if j >= 1 and best.get(cols[j - 1]) == m:
            cell.set_facecolor("#D9F2DD")
            cell.get_text().set_weight("bold")
            cell.get_text().set_color("#166534")
        if j == 0 and m == "h1_hoppo":
            cell.get_text().set_weight("bold")
    ax.set_title(f"TableHop-PPO on HybridQA ({split} split, {seeds} seed{'s' if seeds != '1' else ''}): "
                 "comparison with baselines and ablations", fontsize=12, weight="bold", pad=8)
    ax.text(0.5, 0.0, "Green bold = best learned method per column (oracle excluded: it reads the gold evidence). "
             "Orange row = our method. ↑ higher is better. 95% CI by bootstrap over questions. "
             "Path metrics are n/a for SFT (no hops).", ha="center", va="top", fontsize=8.5, color="#374151",
            transform=ax.transAxes)
    out = ROOT / "results/figures/results_table.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out} ({split}, best: {best})")


if __name__ == "__main__":
    main()
