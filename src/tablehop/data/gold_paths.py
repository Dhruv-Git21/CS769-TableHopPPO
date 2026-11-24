"""Heuristic gold paths for HybridQA from the weak `answer-node` label (specs/07 Data)."""
from __future__ import annotations

import re

from tablehop.data.schema import EntityRef, Table

STOPWORDS = frozenset(
    ["a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by", "with", "and", "or", "is", "are", "was", "were", "be", "been", "what", "which", "who", "whom", "whose", "when", "where", "how", "did", "does", "do", "has", "have", "had", "that", "this", "these", "those", "it", "its", "as", "after", "before", "than", "then", "there", "their", "his", "her", "he", "she", "they", "year", "many", "much", "name"]
)


def tokens(text: str) -> set[str]:
    """Lowercased alphanumeric tokens without stopwords."""
    return {t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS}


def pick_anchor(table: Table, question: str, row: int, avoid_col: int) -> int | None:
    """Column of the same-row cell (col != avoid_col) with the highest question overlap, else None."""
    q = tokens(question)
    best, best_col = 0, None
    for cell in table.rows[row]:
        if cell.col == avoid_col:
            continue
        ov = len(tokens(cell.text) & q)
        if ov > best:
            best, best_col = ov, cell.col
    return best_col


def derive_gold_path(
    table: Table, passages: dict[str, str], question: str, answer_node: list | None
) -> tuple[list[EntityRef] | None, str, str]:
    """Return (path, quality, answer_type). quality in {"heuristic","none"}."""
    if not answer_node:
        return None, "none", "none"
    node = answer_node[0]
    try:
        r, c = int(node[1][0]), int(node[1][1])
        ntype = node[3]
        cell = table.rows[r][c]
    except (IndexError, TypeError, ValueError):
        return None, "none", "none"
    path: list[EntityRef] = []
    acol = pick_anchor(table, question, r, c)
    if acol is not None:
        path.append(EntityRef("cell", table.id, r, acol))
    path.append(EntityRef("cell", table.id, r, c))
    if ntype == "passage":
        pid = node[2] or cell.link
        if pid is not None and pid in passages:
            path.append(EntityRef("passage", table.id, r, c, pid))
    deduped = [e for i, e in enumerate(path) if i == 0 or e != path[i - 1]]
    return deduped, "heuristic", ntype if ntype in ("table", "passage") else "none"
