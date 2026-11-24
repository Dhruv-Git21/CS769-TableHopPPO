"""HybridQA loader. Reads only train/dev traced files; the official test file is never opened."""
from __future__ import annotations

import json
import re
from pathlib import Path

from tablehop.data.gold_paths import derive_gold_path
from tablehop.data.schema import Cell, Example, Table

MAX_PASSAGE_CHARS = 1200
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_NUM = re.compile(r"^[-+]?[$€£]?\d[\d,]*(\.\d+)?%?$")
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def infer_dtype(text: str) -> str:
    """text | number | date (heuristic)."""
    s = text.strip().lower()
    if re.fullmatch(r"(1[0-9]{3}|20[0-9]{2})", s):
        return "date"
    if _YEAR.search(s) and (any(m in s for m in _MONTHS) or re.search(r"\d{1,2}[/-]\d{1,2}", s)):
        return "date"
    if _NUM.match(s):
        return "number"
    return "text"


def _paths(raw_dir: str | Path) -> tuple[Path, Path]:
    root = Path(raw_dir)
    return root / "HybridQA" / "released_data", root / "WikiTables-WithLinks"


def load_questions(raw_dir: str | Path, source: str) -> list[dict]:
    """Official train or dev traced questions (with answer-node). source in {"train","dev"}."""
    if source not in ("train", "dev"):
        raise ValueError("only train/dev are allowed; official test is sealed")
    qdir, _ = _paths(raw_dir)
    with open(qdir / f"{source}.traced.json") as f:
        return json.load(f)


def load_raw_table(raw_dir: str | Path, table_id: str) -> dict:
    _, tdir = _paths(raw_dir)
    with open(tdir / "tables_tok" / f"{table_id}.json") as f:
        return json.load(f)


def load_table(raw_dir: str | Path, table_id: str) -> tuple[Table, dict[str, str]]:
    """Build Table and the passages linked from it (truncated to 1,200 chars)."""
    raw = load_raw_table(raw_dir, table_id)
    _, tdir = _paths(raw_dir)
    rq_path = tdir / "request_tok" / f"{table_id}.json"
    req = {}
    if rq_path.exists():
        with open(rq_path) as f:
            req = json.load(f)
    rows, passages = [], {}
    for r, row in enumerate(raw["data"]):
        cells = []
        for c, (text, links) in enumerate(row):
            cells.append(Cell(text, r, c, links[0] if links else None, infer_dtype(text)))
            for lk in links:
                if lk in req and lk not in passages:
                    passages[lk] = req[lk][:MAX_PASSAGE_CHARS]
        rows.append(cells)
    table = Table(
        table_id, raw.get("title", ""), raw.get("section_title", ""),
        [h[0] for h in raw["header"]], rows,
    )
    return table, passages


def build_example(
    raw_dir: str | Path, q: dict, source: str, split: str, domain: str = "",
    cache: dict | None = None,
) -> Example:
    tid = q["table_id"]
    if cache is not None and tid in cache:
        table, passages = cache[tid]
    else:
        table, passages = load_table(raw_dir, tid)
        if cache is not None:
            cache[tid] = (table, passages)
    path, quality, atype = derive_gold_path(table, passages, q["question"], q.get("answer-node"))
    return Example(
        id=f"hqa:{source}:{q['question_id']}", dataset="hybridqa", split=split,
        question=q["question"], table=table, passages=passages,
        answer=[q["answer-text"]], answer_type=atype, gold_path=path,
        gold_path_quality=quality, domain=domain, meta={"source": source},
    )
