"""Console comparison + tables from results/runs/*/metrics_<split>.json (status done only).

usage: python scripts/summarize.py [--results_dir results] [--ref h1_hoppo]
Run label = config["exp"] if set, else meta["method"]. Runs whose env_lock_sha256 differs abort the script.
Writes results/tables/{main,ablations,ood,paired}.{csv,tex} and macros.tex. Missing methods are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from tablehop.eval.metrics import exact_match
from tablehop.eval.stats import bootstrap_ci, gap, paired_bootstrap, seed_summary

SPLITS = ["dev", "test", "ood_dev", "ood_test"]
ABLATIONS = ["a1_terminal", "a2_noredund", "a3_noschema", "a4_unrestricted"]
KEYS = ["em", "f1", "path_validity_strict", "hop_efficiency", "mean_hops"]
ORDER = ["m0", "b1_sft", "b2a_grpo", "b2b_grpo_path", "b3_soft", "h1_hoppo", *ABLATIONS]
TEX_NAMES = {"n_seeds": "seeds", "b2b_grpo_path": "B2b GRPO+path", "b2a_grpo": "B2a GRPO", "b1_sft": "B1 SFT", "b3_soft": "B3 soft",
             "h1_hoppo": "H1 Hop-PPO", "a1_terminal": "A1 terminal-only", "a2_noredund": "A2 no-redund.",
             "a3_noschema": "A3 no-schema", "a4_unrestricted": "A4 unrestricted", "m0": "M0 oracle"}
DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def label_of(meta: dict) -> str:
    return str(meta.get("config", {}).get("exp") or meta["method"])


def load_runs(rd: Path) -> list[dict]:
    """[{label, seed, run_dir, metrics{split: dict}}] for finished runs; aborts on mixed env hashes."""
    runs, hashes = [], {}
    for mp in sorted((rd / "runs").glob("*/meta.json")):
        meta = json.loads(mp.read_text())
        if meta.get("status") != "done":
            continue
        ms = {s: json.loads((mp.parent / f"metrics_{s}.json").read_text())
              for s in SPLITS if (mp.parent / f"metrics_{s}.json").exists()}
        if not ms:
            continue
        for m in ms.values():
            hashes.setdefault(m.get("env_lock_sha256", "missing"), []).append(mp.parent.name)
        runs.append({"label": label_of(meta), "seed": meta.get("seed", 0), "dir": mp.parent, "metrics": ms})
    if len(hashes) > 1:
        sys.exit(f"ERROR: env_lock_sha256 differs across runs: { {k[:8]: v[:3] for k, v in hashes.items()} }")
    return runs


def correctness(run: dict, split: str) -> dict[str, float]:
    """{question id: EM in {0,1}} from predictions_<split>.jsonl."""
    f = run["dir"] / f"predictions_{split}.jsonl"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = exact_match(r["pred"], r["gold"])
    return out


def aggregate_method(runs: list[dict], split: str) -> dict | None:
    rs = [r for r in runs if split in r["metrics"]]
    if not rs:
        return None
    row: dict = {"n_seeds": len(rs)}
    for k in KEYS:
        vals = [r["metrics"][split].get(k) for r in rs]
        vals = [v for v in vals if v is not None]
        row[k] = seed_summary(vals) if vals else None
    per_q: dict[str, list[float]] = {}
    for r in rs:
        for i, c in correctness(r, split).items():
            per_q.setdefault(i, []).append(c)
    row["per_q"] = {i: float(np.mean(v)) for i, v in per_q.items()}
    pooled = [c for v in per_q.values() for c in v]
    row["em_ci"] = bootstrap_ci(pooled)[1:] if pooled else None
    return row


def fmt(ms, nd=3) -> str:
    return "n/a" if ms is None else f"{ms[0]:.{nd}f}±{ms[1]:.{nd}f}"


def write_table(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    def esc(x) -> str:
        x = str(x)
        for k, v in TEX_NAMES.items():  # paper names in .tex only; csv keeps the run labels
            x = x.replace(k, v)
        return x.replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$").replace("->", r"$\rightarrow$")
    lines = [r"\begin{tabular}{" + "l" * len(header) + "}", r"\toprule",
             " & ".join(esc(h) for h in header) + r" \\", r"\midrule",
             *(" & ".join(esc(c) for c in r) + r" \\" for r in rows), r"\bottomrule", r"\end{tabular}"]
    path.with_suffix(".tex").write_text("\n".join(lines) + "\n")


def macro_name(*parts: str) -> str:
    s = "".join(p.title().replace("_", "") for p in parts)
    return "".join(DIGITS[int(c)] if c.isdigit() else c for c in s if c.isalnum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--ref", default="h1_hoppo")
    a = ap.parse_args()
    rd = Path(a.results_dir)
    runs = load_runs(rd)
    by: dict[str, list[dict]] = {}
    for r in runs:
        by.setdefault(r["label"], []).append(r)
    labels = [m for m in ORDER if m in by] + sorted(m for m in by if m not in ORDER)
    for m in ORDER:
        if m not in by and m != "m0":
            print(f"[warn] no finished runs for {m}; skipped", file=sys.stderr)
    tab = rd / "tables"
    agg = {(m, s): aggregate_method(by[m], s) for m in labels for s in SPLITS}
    macros, main_rows, abl_rows, paired_rows, ood_rows = [], [], [], [], []
    hdr = ["split", "method", "n_seeds", "EM mean±std", "EM 95% CI", "F1", "strict validity",
           "hop eff", "mean hops"]
    for s in SPLITS:
        present = [m for m in labels if agg[(m, s)]]
        if not present:
            continue
        print(f"\n== {s} ==\n" + " | ".join(hdr[1:]))
        for m in present:
            g = agg[(m, s)]
            ci = "n/a" if g["em_ci"] is None else f"[{g['em_ci'][0]:.3f}, {g['em_ci'][1]:.3f}]"
            row = [s, m, g["n_seeds"], fmt(g["em"]), ci, fmt(g["f1"]), fmt(g["path_validity_strict"]),
                   fmt(g["hop_efficiency"]), fmt(g["mean_hops"], 2)]
            print(" | ".join(map(str, row[1:])))
            (abl_rows if m in ABLATIONS else main_rows).append(row)
            if m == a.ref:
                abl_rows.append(row)
            if g["em"]:
                macros.append((macro_name("em", m, s), f"{100 * g['em'][0]:.1f}"))
                macros.append((macro_name("seeds", m, s), str(g["n_seeds"])))
            if g["em_ci"]:
                macros.append((macro_name("emlo", m, s), f"{100 * g['em_ci'][0]:.1f}"))
                macros.append((macro_name("emhi", m, s), f"{100 * g['em_ci'][1]:.1f}"))
            for key, name, scale, nd in (("f1", "fscore", 100, 1), ("path_validity_strict", "valid", 100, 1),
                                         ("hop_efficiency", "hopeff", 100, 1), ("mean_hops", "hops", 1, 2)):
                if g.get(key):
                    macros.append((macro_name(name, m, s), f"{scale * g[key][0]:.{nd}f}"))
        ref = agg.get((a.ref, s))
        if ref:
            for m in present:
                g = agg[(m, s)]
                common = sorted(set(ref["per_q"]) & set(g["per_q"]))
                if m == a.ref or not common:
                    continue
                pb = paired_bootstrap([ref["per_q"][i] for i in common], [g["per_q"][i] for i in common])
                print(f"  paired {a.ref} - {m}: diff {pb['diff']:+.3f} CI [{pb['lo']:+.3f}, {pb['hi']:+.3f}] p={pb['p']:.4f}")
                paired_rows.append([s, f"{a.ref} - {m}", len(common), f"{pb['diff']:+.3f}",
                                    f"[{pb['lo']:+.3f}, {pb['hi']:+.3f}]", f"{pb['p']:.4f}"])
                macros.append((macro_name("diff", m, s), f"{100 * pb['diff']:+.1f}"))
                macros.append((macro_name("difflo", m, s), f"{100 * pb['lo']:+.1f}"))
                macros.append((macro_name("diffhi", m, s), f"{100 * pb['hi']:+.1f}"))
    for m in labels:
        for i_s, o_s in (("dev", "ood_dev"), ("test", "ood_test")):
            gi, go = agg[(m, i_s)], agg[(m, o_s)]
            if gi and go and gi["em"] and go["em"]:
                d, rel = gap(gi["em"][0], go["em"][0])
                ood_rows.append([f"{i_s}->{o_s}", m, f"{gi['em'][0]:.3f}", f"{go['em'][0]:.3f}",
                                 f"{d:+.3f}", f"{100 * rel:.1f}%"])
                macros.append((macro_name("gap", m, o_s), f"{100 * d:.1f}"))
    if ood_rows:
        print("\n== ID -> OOD gap (EM) ==")
        for r in ood_rows:
            print(" | ".join(map(str, r)))
    write_table(tab / "main", hdr, main_rows)
    write_table(tab / "ablations", hdr, abl_rows)
    write_table(tab / "ood", ["pair", "method", "EM ID", "EM OOD", "gap abs", "gap rel"], ood_rows)
    write_table(tab / "paired", ["split", "comparison", "n questions", "diff EM", "95% CI", "p"], paired_rows)
    macros += extra_macros_and_tables(rd, tab)
    (tab / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros))
    print(f"\nwrote tables to {tab}")


def extra_macros_and_tables(rd: Path, tab: Path) -> list[tuple[str, str]]:
    """Data statistics, sanity study and tuning trials: macros + study.tex / datastats.tex."""
    out: list[tuple[str, str]] = []
    ds = tab / "data_stats.csv"
    if ds.exists():
        rows = {r["split"]: r for r in csv.DictReader(ds.open())}
        if "train" in rows:
            t = rows["train"]
            out += [("GoldCovTrain", f"{100 * float(t['frac_heuristic']):.1f}"),
                    ("PathLenTrain", f"{float(t['path_len_mean']):.2f}"),
                    ("LastEntAnsTrain", f"{100 * float(t['last_entity_contains_answer']):.1f}"),
                    ("PassageAnsTrain", f"{100 * int(t['ans_type_passage']) / int(t['n']):.1f}")]
        keep = ["n", "n_tables", "frac_heuristic", "path_len_mean", "ans_type_passage", "rows_mean", "cols_mean"]
        write_table(tab / "datastats", ["split", *keep], [[sp, *(r[k] for k in keep)] for sp, r in rows.items()])
    cr = tab / "candidate_recall.csv"
    if cr.exists():
        rows = {r["split"]: r for r in csv.DictReader(cr.open())}
        if "dev" in rows:
            out.append(("RecallKDev", f"{100 * float(rows['dev']['recall_at_32']):.1f}"))
    for mp in sorted((rd / "sanity").glob("*/metrics_dev.json")):
        m = json.loads(mp.read_text())
        out += [("SanityEm", f"{100 * m['em']:.1f}"), ("SanityValid", f"{100 * (m.get('path_validity_strict') or 0):.1f}")]
        break
    st = rd / "study" / "state.json"
    if st.exists():
        sj = json.loads(st.read_text())
        chosen = {v["trial"] for v in sj.get("chosen", {}).values()}
        rows = []
        for k, v in sorted(sj.get("trials", {}).items()):
            if v.get("dev"):
                rows.append([k.split("/")[0], k.split("/")[1], " ".join(v["overrides"]) or "default",
                             f"{100 * v['dev']['em']:.1f}", "yes" if k in chosen else ""])
        if rows:
            write_table(tab / "study", ["method", "trial", "setting", "dev EM", "chosen"], rows)
    return out


if __name__ == "__main__":
    main()
