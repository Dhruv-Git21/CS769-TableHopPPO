"""Entity text rendering shared by the encoder and the answer head (specs/07).

cell    -> "{title} | {header[col]} | {row anchor} | {cell text}"   (row anchor = first non-numeric cell of the row)
passage -> "{passage title} : {first PASSAGE_CHARS chars}"
Evidence modes (answer-head input, spec change R3): "entity" renders each selected entity as above;
"row" renders each selected cell as its whole row "{title} | h1: v1 ; h2: v2 ..." (one copy per row, in
first-visit order) and each selected passage as above. Passages are never included unless selected.
"""
from __future__ import annotations

from urllib.parse import unquote

from tablehop.data.schema import EntityRef, Example, Table

PASSAGE_CHARS = 400


def passage_title(passage_id: str) -> str:
    """'/wiki/Harbor_FC' -> 'Harbor FC'."""
    return unquote(passage_id.rsplit("/", 1)[-1]).replace("_", " ")


def row_anchor(table: Table, row: int) -> str:
    """First non-numeric cell of the row (rank/index columns skipped); falls back to the first cell."""
    cells = table.rows[row]
    for c in cells:
        if c.text.strip() and not c.text.replace(".", "").replace(",", "").strip().isdigit():
            return c.text
    return cells[0].text if cells else ""


def render_cell(table: Table, row: int, col: int) -> str:
    cells = table.rows[row]
    header = table.header[col] if col < len(table.header) else ""
    anchor = row_anchor(table, row)
    return f"{table.title} | {header} | {anchor} | {cells[col].text}"


def render_passage(passage_id: str, passages: dict[str, str]) -> str:
    text = passages.get(passage_id, "")[:PASSAGE_CHARS]
    return f"{passage_title(passage_id)} : {text}"


def render_entity(ref: EntityRef, ex: Example) -> str:
    """Text of one entity of an example."""
    if ref.kind == "passage":
        return render_passage(ref.passage_id or "", ex.passages)
    return render_cell(ex.table, ref.row, ref.col)


def render_row(table: Table, row: int) -> str:
    cells = table.rows[row]
    parts = [f"{table.header[c] if c < len(table.header) else ''}: {cell.text}" for c, cell in enumerate(cells)]
    return f"{table.title} | " + " ; ".join(parts)


def render_evidence(refs: list[EntityRef], ex: Example, mode: str = "entity") -> list[str]:
    if mode == "entity":
        return [render_entity(r, ex) for r in refs]
    if mode != "row":
        raise ValueError(f"unknown evidence mode {mode!r}")
    out, seen = [], set()
    for r in refs:
        key = ("p", r.passage_id) if r.kind == "passage" else ("r", r.row)
        if key in seen:
            continue
        seen.add(key)
        out.append(render_passage(r.passage_id or "", ex.passages) if r.kind == "passage"
                   else render_row(ex.table, r.row))
    return out
