"""F2-F7, F9, F10 (specs/03) from results/runs -> results/figures/*.{pdf,png} (200 dpi). Skips what has no data.

usage: python scripts/make_figures.py [--results_dir results] [--ref h1_hoppo]
Error taxonomy (F9) uses only what predictions store: wrong first hop, wrong later hop, right path but wrong
answer (covers answer-head errors), no-gold-path. Candidate-recall failures need candidate lists, not stored.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from summarize import ABLATIONS, ORDER, aggregate_method, label_of, load_runs  # noqa: F401

OURS, TEAL, GREY, NAVY, LTEAL = "#B4501A", "#1F6F8B", "#6B7280", "#14213D", "#7FB3C4"
BASE_COLORS = [TEAL, GREY, NAVY, LTEAL]
CREAM = "#F6F5F0"


FIXED = {"h1_hoppo": OURS, "b1_sft": TEAL, "b2a_grpo": GREY, "b2b_grpo_path": NAVY, "b3_soft": LTEAL,
         "m0": "#2E7D32", "a1_terminal": "#6A4C93", "a2_noredund": "#C9A227", "a3_noschema": "#8D6E63",
         "a4_unrestricted": "#E07A5F"}


DISPLAY = {"m0": "Oracle path", "b1_sft": "Full-table reader (SFT)", "b2a_grpo": "Answer-only RL (GRPO)",
           "b2b_grpo_path": "Answer+path RL (GRPO)", "b3_soft": "Soft selection", "h1_hoppo": "Hop-PPO (ours)",
           "a1_terminal": "Ours: answer reward only", "a2_noredund": "Ours: no repeat penalty",
           "a3_noschema": "Ours: no type check", "a4_unrestricted": "Ours: any-cell moves"}


def disp(m: str, wrap: bool = False) -> str:
    """Plain-language method name for figures (internal ids stay in file names and tables)."""
    d = DISPLAY.get(m, m)
    if wrap and len(d) > 11 and " " in d:  # break the longest name roughly in half
        words, best = d.split(" "), None
        for k in range(1, len(d.split(" "))):
            a, b = " ".join(words[:k]), " ".join(words[k:])
            if best is None or max(len(a), len(b)) < max(len(best[0]), len(best[1])):
                best = (a, b)
        d = "\n".join(best)
    return d


def color_of(m: str, i: int) -> str:
    """One fixed color per method across all figures."""
    return FIXED.get(m, BASE_COLORS[i % len(BASE_COLORS)])


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=CREAM)
    plt.close(fig)
    print(f"wrote {name}")


def newfig(w=6.0, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def key(e) -> str:
    return json.dumps(e, sort_keys=True)


def preds(run: dict, split: str) -> list[dict]:
    f = run["dir"] / f"predictions_{split}.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def bars(ax, labels, vals, cis, colors):
    x = np.arange(len(labels))
    err = np.array([[v - c[0], c[1] - v] if c else [0, 0] for v, c in zip(vals, cis)]).T
    ax.bar(x, vals, color=colors, yerr=np.clip(err, 0, None), capsize=3, ecolor=NAVY)
    ax.set_xticks(x, [disp(m, wrap=True) for m in labels], rotation=0, ha="center", fontsize=8)


def main_split(agg, labels):
    for s in ("test", "dev"):
        if any(agg.get((m, s)) for m in labels):
            return s
    return None


def f2_f5(agg, labels, out, ref):
    s = main_split(agg, labels)
    if not s:
        return
    for name, ms in (("F2_main", [m for m in labels if m not in ABLATIONS]),
                     ("F5_ablations", [m for m in labels if m in ABLATIONS or m == ref])):
        ms = [m for m in ms if agg.get((m, s))]
        if not ms:
            continue
        fig, ax = newfig()
        bars(ax, ms, [agg[(m, s)]["em"][0] for m in ms], [agg[(m, s)]["em_ci"] for m in ms],
             [color_of(m, i) for i, m in enumerate(ms)])
        ax.set_ylabel(f"exact-match accuracy ({s}; bars = 95% CI)")
        save(fig, out, name)


def f3(by, out):
    fig, ax = newfig()
    drawn = False
    for i, (m, runs) in enumerate(sorted(by.items())):
        if m in ABLATIONS:  # main methods only; ablation curves would clutter the panel
            continue
        curves = []
        for r in runs:
            f = r["dir"] / "train_log.csv"
            if not f.exists():
                continue
            rows = list(csv.DictReader(f.open()))
            xk = next((k for k in ("trajectories_seen", "trajectories", "examples_seen", "step") if rows and k in rows[0]), None)
            yk = next((k for k in ("dev_em", "dev_em_subset") if rows and k in rows[0]), None)
            if xk and yk:
                curves.append({float(r_[xk]): float(r_[yk]) for r_ in rows if r_[yk] not in ("", None)})
        if not curves:
            continue
        xs = sorted(set.intersection(*(set(c) for c in curves)) or set().union(*curves))
        ys = np.array([[c[x] for x in xs if x in c] for c in curves if all(x in c for x in xs)] or [[]])
        if ys.size == 0:
            continue
        mu = ys.mean(0)
        ci = 1.96 * ys.std(0, ddof=1) / np.sqrt(len(ys)) if len(ys) > 1 else np.zeros_like(mu)
        ax.plot(xs, mu, color=color_of(m, i), label=disp(m))
        ax.fill_between(xs, mu - ci, mu + ci, color=color_of(m, i), alpha=0.2)
        drawn = True
    if drawn:
        ax.set_xlabel("training progress (questions practised)")
        ax.set_ylabel("dev exact-match accuracy")
        ax.legend(frameon=False)
        save(fig, out, "F3_learning_curves")
    else:
        plt.close(fig)


def hop_stats(runs, split):
    """per-hop strict validity lists, hop counts, taxonomy Counter."""
    valid: dict[int, list[float]] = {}
    hops, tax = Counter(), Counter()
    for r in runs:
        for p in preds(r, split):
            path, gold = p.get("path") or [], {key(g) for g in p.get("gold_path") or []}
            ti = p.get("trace_info") or []
            hops[len(path)] += 1
            for t, e in enumerate(path):
                ok = t < len(ti) and ti[t]["legal"] and not ti[t]["redundant"] and key(e) in gold
                valid.setdefault(t + 1, []).append(float(ok))
            correct = p["pred"].strip().lower() in [g.strip().lower() for g in (
                p["gold"] if isinstance(p["gold"], list) else [p["gold"]])]
            if correct:
                continue
            gp = [key(g) for g in p.get("gold_path") or []]
            if not gp or not path:
                tax["no gold path / no hops"] += 1
            elif key(path[0]) != gp[0]:
                tax["wrong first hop"] += 1
            elif any(key(e) not in gold for e in path[1:]):
                tax["wrong later hop"] += 1
            else:
                tax["right path, wrong answer"] += 1
    return valid, hops, tax


def f4_f7_f9(by, out):
    s = "test" if any("test" in r["metrics"] for rs in by.values() for r in rs) else "dev"
    fig4, a4 = newfig()
    fig7, a7 = newfig()
    stats, hasv = {}, False
    for i, (m, runs) in enumerate(sorted(by.items())):
        valid, hops, tax = hop_stats([r for r in runs if s in r["metrics"]], s)
        if not hops or sum(k for k in hops if k) == 0:
            continue
        stats[m] = tax
        if m in ABLATIONS:  # F4/F7 compare main methods; ablations appear in F5/F9
            continue
        xs = sorted(valid)
        a4.plot(xs, [np.mean(valid[x]) for x in xs], marker="o", color=color_of(m, i), label=disp(m))
        hasv = True
        ks = sorted(hops)
        tot = sum(hops.values())
        a7.plot(ks, [hops[k] / tot for k in ks], marker="s", color=color_of(m, i), label=disp(m))
    if hasv:
        a4.set_xlabel("hop number")
        a4.set_ylabel(f"share of hops on the gold path ({s})")
        a4.set_xticks(range(1, 5))
        a4.legend(frameon=False)
        save(fig4, out, "F4_hop_validity")
        a7.set_xlabel("hops taken")
        a7.set_ylabel("fraction of questions")
        a7.legend(frameon=False)
        save(fig7, out, "F7_hop_counts")
    else:
        plt.close(fig4), plt.close(fig7)
    if stats:
        cats = ["wrong first hop", "wrong later hop", "right path, wrong answer", "no gold path / no hops"]
        cols = [OURS, "#F2B48A", TEAL, GREY]
        order = [m for m in ORDER if m in stats][::-1]  # top-to-bottom in the usual method order
        fig, ax = newfig()
        left = np.zeros(len(order))
        for c, col in zip(cats, cols):
            v = np.array([stats[m][c] / max(sum(stats[m].values()), 1) for m in order])
            ax.barh([disp(m) for m in order], v, left=left, color=col, label=c)
            left += v
        ax.set_xlabel(f"share of wrong answers ({s})")
        ax.set_xlim(0, 1.0)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.45, 1.0))
        plt.setp(ax.get_yticklabels(), fontsize=8.5)
        save(fig, out, "F9_error_taxonomy")


def f6(agg, labels, out):
    pairs = [("test", "ood_test"), ("dev", "ood_dev")]
    pr = next((p for p in pairs if any(agg.get((m, p[0])) and agg.get((m, p[1])) for m in labels)), None)
    if not pr:
        return
    ms = [m for m in labels if agg.get((m, pr[0])) and agg.get((m, pr[1]))]
    fig, ax = newfig()
    x = np.arange(len(ms))
    ax.bar(x - 0.2, [agg[(m, pr[0])]["em"][0] for m in ms], 0.4, color=TEAL, label=f"ID ({pr[0]})")
    ax.bar(x + 0.2, [agg[(m, pr[1])]["em"][0] for m in ms], 0.4, color=OURS, label=f"OOD ({pr[1]})")
    for xi, m in zip(x, ms):
        g = agg[(m, pr[0])]["em"][0] - agg[(m, pr[1])]["em"][0]
        ax.text(xi, max(agg[(m, pr[0])]["em"][0], agg[(m, pr[1])]["em"][0]) + 0.01, f"gap {g:+.2f}",
                ha="center", fontsize=7)
    ax.set_xticks(x, [disp(m, wrap=True) for m in ms], rotation=0, ha="center", fontsize=8.5)
    ax.set_ylabel("exact-match accuracy")
    ax.legend(frameon=False)
    save(fig, out, "F6_id_vs_ood")


def f10(by, out, ref: str = "h1_hoppo", win: int = 25):
    """Reward components of the reference method over training (seed-0 run, rolling mean over `win`
    updates), plus the training-batch answer F1 for comparison."""
    runs = [r for r in by.get(ref, []) if (r["dir"] / "train_log.csv").exists()]
    if not runs:
        return
    r = sorted(runs, key=lambda x: x["seed"])[0]
    rows = list(csv.DictReader((r["dir"] / "train_log.csv").open()))
    names = {"r_hop": "reward for a correct hop", "r_red": "penalty for revisiting", "r_schema": "penalty for wrong answer type",
             "r_step": "cost per hop", "r_terminal": "reward for the final answer"}
    fig, ax = newfig()
    for j, (k, lab) in enumerate(names.items()):
        if k not in rows[0]:
            continue
        y = np.array([float(x[k]) if x[k] else np.nan for x in rows])
        sm = np.convolve(np.nan_to_num(y), np.ones(win) / win, mode="valid")
        ax.plot(np.arange(len(sm)) + win, sm, label=lab, color=(OURS, TEAL, GREY, NAVY, LTEAL)[j])
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xlabel("training update")
    ax.set_ylabel("mean reward per episode")
    ax.set_title(f"{disp(ref)}, seed {r['seed']} ({win}-update moving average)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out, "F10_reward_components")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--ref", default="h1_hoppo")
    a = ap.parse_args()
    rd = Path(a.results_dir)
    out = rd / "figures"
    runs = load_runs(rd)
    by: dict[str, list[dict]] = {}
    for r in runs:
        by.setdefault(r["label"], []).append(r)
    if not by:
        print("[warn] no finished runs; no figures", file=sys.stderr)
        return
    labels = [m for m in ORDER if m in by] + sorted(m for m in by if m not in ORDER)
    splits = ("dev", "test", "ood_dev", "ood_test")
    agg = {(m, s): aggregate_method(by[m], s) for m in labels for s in splits}
    agg = {k: v for k, v in agg.items() if v and v["em"]}
    f2_f5(agg, labels, out, a.ref)
    f3(by, out)
    f4_f7_f9(by, out)
    f6(agg, labels, out)
    f10(by, out)


if __name__ == "__main__":
    main()
