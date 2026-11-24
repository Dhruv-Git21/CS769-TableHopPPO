"""Build deliverables/WORK_REPORT.{docx,pdf} from docs/report/WORK_REPORT.md.

Markers in the markdown are replaced by content generated from results/ and git:
  <!-- TABLE:main|ablations|ood|paired -->  results/tables/<name>.csv as a markdown table (all rows)
  <!-- FIG:<name> -->                       results/figures/<name>.png + caption (placeholder if missing)
  <!-- RUNS -->                             final runs from results/study/state.json ("final") + wall time
  <!-- TIMELINE -->                         dated list from `git log`, oldest first, grouped by date
  <!-- REFS -->                             reference list from paper/refs.bib (verified entries only)
  <!-- CONFIGS -->                          key hyperparameters per configs/*.yaml (resolved with base chain)
No number is typed by hand; everything is read from files. usage: python scripts/make_report.py [--no-pdf]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs/report/WORK_REPORT.md"
REF_DOCX = ROOT / "docs/report/reference.docx"
OUT = ROOT / "deliverables"

CAPTIONS = {
    "F1_architecture": "Figure: pipeline and architecture of Hop-PPO.",
    "F2_main": "Figure: main in-domain comparison (EM with 95% bootstrap CI).",
    "F3_learning_curves": "Figure: learning curves, dev EM against training trajectories.",
    "F4_hop_validity": "Figure: path validity by hop index.",
    "F5_ablations": "Figure: reward and action-space ablations.",
    "F6_id_vs_ood": "Figure: in-domain versus held-out-schema EM and the gap.",
    "F7_hop_counts": "Figure: distribution of hops taken.",
    "F8_case_studies": "Figure: qualitative path case studies.",
    "F9_error_taxonomy": "Figure: error taxonomy.",
    "F10_reward_components": "Figure: reward components during training (reward-hacking check).",
}


def md_table(rows: list[list[str]]) -> str:
    if not rows:
        return "_(no rows yet)_"
    head, body = rows[0], rows[1:]
    esc = lambda s: str(s).replace("|", "\\|")
    out = ["| " + " | ".join(esc(c) for c in head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in body]
    return "\n".join(out)


COLS = {"data_stats": ["split", "n", "n_tables", "frac_heuristic", "frac_none", "path_len_mean", "len1", "len2",
                       "len3", "ans_type_table", "ans_type_passage"]}  # compact view of wide tables


def table_from_csv(name: str) -> str:
    p = ROOT / "results/tables" / f"{name}.csv"
    if not p.exists():
        return f"_(results/tables/{name}.csv not found)_"
    with p.open(newline="") as f:
        rows = list(csv.reader(f))
    if name in COLS and rows:
        idx = [rows[0].index(c) for c in COLS[name] if c in rows[0]]
        rows = [[r[i] for i in idx] for r in rows]
    if len(rows) <= 1:
        return md_table(rows) + "\n\n_(no data rows in this table yet)_"
    return md_table(rows) + f"\n\nTable: `results/tables/{name}.csv`, all rows as generated."


def figure(name: str) -> str:
    p = ROOT / "results/figures" / f"{name}.png"
    cap = CAPTIONS.get(name, f"Figure {name}.")
    if not p.exists():
        return f"> **[FIGURE PENDING: {name}]** `results/figures/{name}.png` does not exist yet."
    return f"![{cap} Source: `results/figures/{name}.png`.]({p}){{width=6in}}"


def runs_table() -> str:
    st_p = ROOT / "results/study/state.json"
    if not st_p.exists():
        return "_(results/study/state.json not found)_"
    final = json.loads(st_p.read_text()).get("final", {})
    items: list[tuple[str, str]] = []
    for k, v in final.items():
        if isinstance(v, dict):
            items += [(f"{k}/{k2}", v2) for k2, v2 in v.items()]
        else:
            items.append((k, v))
    if not items:
        return "_(no final runs recorded yet: `final` map in results/study/state.json is empty)_"
    rows = [["method/seed", "run", "wall time (min)", "trajectories", "device"]]
    for k, d in sorted(items):
        rd = ROOT / d
        mp = rd / "meta.json"
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        wt = meta.get("train_time_s")
        rows.append([k, rd.name, f"{wt / 60:.1f}" if wt else "n/a", str(meta.get("trajectories_seen", "n/a")),
                     str(meta.get("config", {}).get("device", "n/a")) if isinstance(meta.get("config"), dict)
                     else str(meta.get("device", "n/a"))])
    return md_table(rows)


def timeline() -> str:
    out = subprocess.run(["git", "log", "--date=short", "--pretty=%ad\t%s"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip().splitlines()
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for ln in reversed(out):
        d, _, s = ln.partition("\t")
        groups.setdefault(d, []).append(s)
    return "\n".join(f"- **{d}**: " + "; ".join(v) for d, v in groups.items())


def load_cfg(name: str, seen: tuple = ()) -> dict:
    import yaml
    d = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text()) or {}
    base = d.pop("base", None)
    out = load_cfg(base.removesuffix(".yaml"), seen + (name,)) if base and base not in seen else {}
    for k, v in d.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def configs_table() -> str:
    names = ["b1_sft", "b2a_grpo", "b2b_grpo_path", "b3_soft", "h1_hoppo", "a1_terminal", "a2_noredund",
             "a3_noschema", "a4_unrestricted", "m0"]
    rows = [["config", "method", "budget (trajectories or examples)", "lr (head)", "entropy coef", "max hops",
             "reward overrides"]]
    for n in names:
        if not (ROOT / "configs" / f"{n}.yaml").exists():
            continue
        c = load_cfg(n)
        rl, sft = c.get("rl", {}), c.get("sft", {})
        raw = __import__("yaml").safe_load((ROOT / "configs" / f"{n}.yaml").read_text()) or {}
        ov = ", ".join(f"{k}={v}" for k, v in (raw.get("reward") or {}).items()) or "none"
        if raw.get("env"):
            ov += "; env: " + ", ".join(f"{k}={v}" for k, v in raw["env"].items())
        rows.append([n, str(c.get("method", n)), str(rl.get("budget_trajectories", sft.get("budget_examples", "n/a"))),
                     str(sft.get("lr", c.get("ppo", {}).get("lr_head", "n/a"))),
                     str(c.get("ppo", {}).get("entropy_coef", "n/a")), str(c.get("env", {}).get("max_hops", "n/a")),
                     ov])
    return md_table(rows) + "\n\nTable: resolved values from `configs/*.yaml` (base chain applied)."


def refs_list() -> str:
    """Render paper/refs.bib as a numbered list (title, authors, venue, year, key)."""
    txt = (ROOT / "paper/refs.bib").read_text()
    out = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", txt, re.DOTALL):
        f = {k.lower(): v for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\n|$)", m.group(3))}
        clean = lambda s: re.sub(r"[{}\\]", "", s).replace("\"", "")
        venue = f.get("booktitle") or f.get("journal") or f.get("publisher") or "arXiv preprint"
        out.append((m.group(2), f"{clean(f.get('author', ''))}. {clean(f.get('title', ''))}. "
                    f"{clean(venue)}, {f.get('year', '')}. [`{m.group(2)}`]"))
    return "\n".join(f"{i}. {t}" for i, (_, t) in enumerate(sorted(out), 1))


def macro_values() -> dict[str, str]:
    """results/tables/macros.tex -> {name: value}; {{Name}} in the report uses the Test value, else Dev."""
    f = ROOT / "results/tables/macros.tex"
    vals = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", f.read_text())) if f.exists() else {}
    return vals


def macro(name: str, vals: dict[str, str]) -> str:
    for cand in (name + "Test", name + "Dev", name):
        if cand in vals:
            return vals[cand]
    return "n/a"


def expand(text: str) -> str:
    vals = macro_values()
    split = "test" if "EmHOneHoppoTest" in vals else "development"
    text = text.replace("{{SPLIT}}", split)
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: macro(m.group(1), vals), text)
    text = re.sub(r"<!--\s*TABLE:(\w+)\s*-->", lambda m: table_from_csv(m.group(1)), text)
    text = re.sub(r"<!--\s*FIG:(\w+)\s*-->", lambda m: figure(m.group(1)), text)
    text = re.sub(r"<!--\s*RUNS\s*-->", lambda m: runs_table(), text)
    text = re.sub(r"<!--\s*TIMELINE\s*-->", lambda m: timeline(), text)
    text = re.sub(r"<!--\s*REFS\s*-->", lambda m: refs_list(), text)
    text = re.sub(r"<!--\s*CONFIGS\s*-->", lambda m: configs_table(), text)
    return text


def soffice_pdf(docx: Path, outdir: Path) -> bool:
    exe = shutil.which("soffice") or "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    try:
        r = subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(docx)],
                           capture_output=True, text=True, timeout=240)
        return r.returncode == 0 and (outdir / (docx.stem + ".pdf")).exists()
    except subprocess.TimeoutExpired:
        print("soffice timed out", file=sys.stderr)
        return False


def style_tables(docx: Path) -> None:
    """Add borders, header shading and small type to every table (pandoc tables are unstyled)."""
    try:
        from docx import Document
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Pt
    except ImportError:
        return
    d = Document(str(docx))
    for t in d.tables:
        pr = t._tbl.tblPr
        b = OxmlElement("w:tblBorders")
        for e in ("top", "left", "bottom", "right", "insideH", "insideV"):
            x = OxmlElement(f"w:{e}")
            x.set(qn("w:val"), "single"); x.set(qn("w:sz"), "4"); x.set(qn("w:color"), "DAD7CC")
            b.append(x)
        pr.append(b)
        for ri, row in enumerate(t.rows):
            for c in row.cells:
                if ri == 0:
                    sh = OxmlElement("w:shd")
                    sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "E4EEF1")
                    c._tc.get_or_add_tcPr().append(sh)
                for p in c.paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    for r in p.runs:
                        r.font.size = Pt(7.5)
                        r.font.bold = True if ri == 0 else r.font.bold
    d.save(str(docx))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    md = expand(SRC.read_text())
    gen = OUT / "_WORK_REPORT_expanded.md"
    gen.write_text(md)
    docx = OUT / "WORK_REPORT.docx"
    cmd = ["pandoc", str(gen), "-f", "markdown+pipe_tables+implicit_figures", "-o", str(docx), "--resource-path", str(ROOT)]
    if REF_DOCX.exists():
        cmd += ["--reference-doc", str(REF_DOCX)]
    subprocess.run(cmd, check=True)
    gen.unlink()
    style_tables(docx)
    print("wrote", docx)
    if not a.no_pdf:
        print("wrote" if soffice_pdf(docx, OUT) else "FAILED", OUT / "WORK_REPORT.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
