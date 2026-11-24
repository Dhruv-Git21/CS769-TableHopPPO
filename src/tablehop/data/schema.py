"""Dataclasses for processed examples. JSON round-trippable."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Cell:
    """One table cell. `link` is the passage id (hyperlink) or None; dtype in text|number|date."""

    text: str
    row: int
    col: int
    link: str | None = None
    dtype: str = "text"


@dataclass
class Table:
    """Table with header (n_cols,) and rows (n_rows, n_cols) of Cell."""

    id: str
    title: str
    section_title: str
    header: list[str]
    rows: list[list[Cell]]


@dataclass(frozen=True)
class EntityRef:
    """Hashable pointer to a cell or a linked passage (passage_id = hyperlink string)."""

    kind: str  # "cell" | "passage"
    table_id: str
    row: int
    col: int
    passage_id: str | None = None


@dataclass
class Example:
    id: str
    dataset: str
    split: str
    question: str
    table: Table
    passages: dict[str, str]
    answer: list[str]
    answer_type: str  # table | passage | none
    gold_path: list[EntityRef] | None = None
    gold_path_quality: str = "none"  # heuristic | exact | none
    domain: str = ""
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> Example:
        t = d["table"]
        table = Table(
            t["id"], t["title"], t["section_title"], list(t["header"]),
            [[Cell(**c) for c in r] for r in t["rows"]],
        )
        gp = d.get("gold_path")
        return Example(
            id=d["id"], dataset=d["dataset"], split=d["split"], question=d["question"],
            table=table, passages=dict(d["passages"]), answer=list(d["answer"]),
            answer_type=d["answer_type"],
            gold_path=None if gp is None else [EntityRef(**e) for e in gp],
            gold_path_quality=d.get("gold_path_quality", "none"),
            domain=d.get("domain", ""), meta=dict(d.get("meta", {})),
        )

    @staticmethod
    def from_json(s: str) -> Example:
        return Example.from_dict(json.loads(s))
