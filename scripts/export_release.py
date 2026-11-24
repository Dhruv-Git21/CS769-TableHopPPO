"""Export a clean, GitHub-ready copy of the project (code + configs + tests + docs + paper + results).

usage: python scripts/export_release.py [--out ../tablehop-ppo-release] [--with-h1-checkpoints]

Included: src/ scripts/ configs/ tests/ examples/ paper sources, docs/ (proposal, literature, report source),
deliverables/, data/splits/ (split ids + manifest), and results/: tables, figures, study state, TEST_LOG,
data report, demo traces, and for every final run its meta.json, metrics_*.json, train_log.csv and gzipped
predictions. Optional: the three final H1 policy checkpoints (~9 MB each) so the demo runs after the answer
head is re-trained (`scripts/train_answer_head.py --config configs/answer_head_row.yaml`, ~10 min).
Excluded: raw/processed data and embedding caches (re-created by scripts), model weights of the answer head
and B1 (~300 MB each), logs, virtualenvs, node_modules, LaTeX build files, and the internal orchestration
files used while building the project (CLAUDE.md, START_PROMPT.md, ORCHESTRATION_GUIDE.md, STATUS.md, tasks/,
specs/, .claude/). Design decisions R1-R4 are documented in docs/DESIGN_NOTES.md.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREES = ["src", "scripts", "configs", "tests", "examples", "docs", "deliverables", "data/splits"]
RESULT_DOCS = ["README.md"]
FILES = ["README.md", "Makefile", "pyproject.toml", "requirements.txt", "requirements-lock.txt", "LICENSE"]
PAPER = [".tex", ".bib", ".sty", ".bst", ".pdf"]
RESULT_TREES = ["tables", "figures", "study", "demo", "sanity"]
RESULT_FILES = ["TEST_LOG.md", "data_report.md"]
RUN_FILES = ["meta.json", "train_log.csv"]
DROP = ["deliverables/final_presentation_draft.pptx", "deliverables/code_structure_slide.pptx",
        "scripts/dev/g4_seed0.sh", "scripts/dev/stage1.sh"]  # superseded drafts / machine-specific one-offs
SKIP_DIRS = {"__pycache__", "node_modules", ".pytest_cache", ".ruff_cache", "checkpoint", ".DS_Store"}
GITIGNORE = """__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
node_modules/
.DS_Store
data/raw/
data/cache/
data/processed/
results/answer_head*/
results/runs/*/checkpoint/
results/smoke/
logs/
paper/*.aux
paper/*.log
paper/*.out
paper/*.bbl
paper/*.blg
"""


def ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in SKIP_DIRS or n.endswith((".pyc", ".aux", ".log", ".out", ".blg"))}


def final_runs() -> list[Path]:
    """Final runs from the study state; before the study finishes, every finished run in results/runs."""
    sf = ROOT / "results/study/state.json"
    st = json.loads(sf.read_text()) if sf.exists() else {}
    runs = [ROOT / v for v in st.get("final", {}).values() if v]
    if not runs:
        runs = [p.parent for p in sorted((ROOT / "results/runs").glob("*/meta.json"))
                if json.loads(p.read_text()).get("status") == "done" and p.parent.name != "m0_oracle"]
    m0 = ROOT / "results/runs/m0_oracle"
    return runs + ([m0] if m0.exists() else [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "tablehop-ppo-release"))
    ap.add_argument("--with-h1-checkpoints", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit("--out must be outside the project directory")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for t in TREES:
        if (ROOT / t).exists():
            shutil.copytree(ROOT / t, out / t, ignore=ignore, symlinks=False)
    for f in FILES:
        if (ROOT / f).exists():
            shutil.copy2(ROOT / f, out / f)
    (out / "paper").mkdir()
    for p in (ROOT / "paper").rglob("*"):
        if p.is_file() and p.suffix in PAPER:
            dst = out / "paper" / p.relative_to(ROOT / "paper")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    res = out / "results"
    for t in RESULT_TREES:
        if (ROOT / "results" / t).exists():
            shutil.copytree(ROOT / "results" / t, res / t, ignore=ignore)
    for f in RESULT_FILES + RESULT_DOCS:
        if (ROOT / "results" / f).exists():
            shutil.copy2(ROOT / "results" / f, res / f)
    for ah in ("answer_head_row", "answer_head"):  # metrics + hash only, never weights
        for f in ("dev_metrics.json", "sha256.txt", "evidence_mode.txt"):
            src = ROOT / "results" / ah / f
            if src.exists():
                (res / ah).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, res / ah / f)
    n_ck = 0
    for rd in final_runs():
        dst = res / "runs" / rd.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in RUN_FILES + [p.name for p in rd.glob("metrics_*.json")]:
            if (rd / f).exists():
                shutil.copy2(rd / f, dst / f)
        for p in rd.glob("predictions_*.jsonl"):
            with open(p, "rb") as fi, gzip.open(dst / (p.name + ".gz"), "wb") as fo:
                shutil.copyfileobj(fi, fo)
        if a.with_h1_checkpoints and rd.name.startswith("h1_hoppo") and (rd / "checkpoint.pt").exists():
            shutil.copy2(rd / "checkpoint.pt", dst / "checkpoint.pt")
            n_ck += 1
    (out / ".gitignore").write_text(GITIGNORE)
    (out / "data").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "data/README.md", out / "data/README.md")
    for f in DROP:
        if (out / f).exists():
            (out / f).unlink()
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    big = [(p.stat().st_size, p) for p in out.rglob("*") if p.is_file() and p.stat().st_size > 50e6]
    print(f"[release] {out}  {size / 1e6:.1f} MB  final runs: {len(final_runs())}  h1 checkpoints: {n_ck}")
    for s, p in big:
        print(f"[release] WARNING file > 50 MB (GitHub limit 100 MB): {p} {s / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
