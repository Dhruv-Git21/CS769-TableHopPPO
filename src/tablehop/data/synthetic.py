"""Deterministic synthetic multi-hop generator (tests and labelled sanity study ONLY).

Templates (all gold paths are reachable by env rules: same row / same column / link):
  2hop: anchor(Player) -> answer(Team)                        same row
  3hop_row: anchor(Player) -> Team -> City                    same row, same row
  3hop_passage: anchor(Player) -> Team (linked) -> passage    same row, link; answer in passage
"""
from __future__ import annotations

import random

from tablehop.data.schema import Cell, EntityRef, Example, Table

_SYL = ["ka", "ro", "mi", "te", "la", "vo", "su", "ne", "pi", "do", "zan", "bel", "tor", "mar", "quin", "lex", "dar", "fen", "gor", "hal"]
HEADER = ["Player", "Team", "City", "Country"]
TEMPLATES = ("2hop", "3hop_row", "3hop_passage")


def _name(rng: random.Random, used: set[str]) -> str:
    while True:
        n = "".join(rng.choice(_SYL) for _ in range(rng.randint(2, 3))).capitalize()
        if n not in used:
            used.add(n)
            return n


def make_example(i: int, seed: int) -> Example:
    rng = random.Random(seed * 1_000_003 + i)
    tid = f"syn_{seed}_{i}"
    n_rows = rng.randint(5, 8)
    used: set[str] = set()
    rows, passages, facts = [], {}, []
    for r in range(n_rows):
        player, team, city, country = (_name(rng, used) for _ in range(4))
        year = str(rng.randint(1900, 2020))
        pid = f"/wiki/{team}"
        passages[pid] = f"{team} is a club founded in {year}. It is based in {city}."
        rows.append([
            Cell(player, r, 0, None, "text"), Cell(team + " FC", r, 1, pid, "text"),
            Cell(city, r, 2, None, "text"), Cell(country, r, 3, None, "text"),
        ])
        facts.append((player, team + " FC", city, year, pid))
    r = rng.randrange(n_rows)
    player, team, city, year, pid = facts[r]
    kind = TEMPLATES[rng.randrange(len(TEMPLATES))]
    anchor = EntityRef("cell", tid, r, 0)
    team_ref = EntityRef("cell", tid, r, 1)
    if kind == "2hop":
        q, ans, path, atype = f"Which team does {player} play for?", team, [anchor, team_ref], "table"
    elif kind == "3hop_row":
        q = f"In which city is the team of {player} located?"
        ans, atype = city, "table"
        path = [anchor, team_ref, EntityRef("cell", tid, r, 2)]
    else:
        q, ans, atype = f"In what year was the club of {player} founded?", year, "passage"
        path = [anchor, team_ref, EntityRef("passage", tid, r, 1, pid)]
    table = Table(tid, f"Synthetic roster {i}", "Players", list(HEADER), rows)
    return Example(
        id=f"syn:{seed}:{i}", dataset="synthetic", split="synthetic", question=q, table=table,
        passages=passages, answer=[ans], answer_type=atype, gold_path=path,
        gold_path_quality="exact", domain="synthetic", meta={"template": kind},
    )


def generate(n: int, seed: int) -> list[Example]:
    """n examples, deterministic in (n prefix, seed)."""
    return [make_example(i, seed) for i in range(n)]
