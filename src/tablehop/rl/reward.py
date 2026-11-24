"""Reward (specs/02). All weights and switches come from cfg["reward"].

Per entity hop:  r_t = w_hop*on_gold - w_red*redundant - w_step          (STOP itself earns nothing)
Terminal:        r_T = w_ans*F1(pred, gold) - w_schema*[last selected entity schema-inconsistent]
                       + w_stop*[all gold entities visited] (+ w_path*setF1(trace, gold) if path_f1_terminal)
Schema rule: expected dtypes come from question cues (cfg schema_cues): number cues -> {number};
date cues -> {date, number}; no cue -> no constraint. Only the LAST selected entity is checked (spec 02);
passages are never penalised (their dtype is unknown). Examples with gold_path_quality == "none" get no hop term.
Switches: terminal_only zeroes w_hop/w_red/w_schema/w_step; no_redundancy, no_schema, no_step zero one weight.
"""
from __future__ import annotations

from tablehop.data.schema import EntityRef, Example
from tablehop.eval.metrics import f1, gold_set_f1


def expected_dtypes(question: str, cues: dict) -> set[str] | None:
    """Dtypes a final cell may have, from question cues (checked in order: number, then date)."""
    q = question.lower()
    if any(c in q for c in cues["number"]):
        return {"number"}
    if any(c in q for c in cues["date"]):
        return {"date", "number"}
    return None


def schema_ok(ref: EntityRef, ex: Example, cues: dict) -> bool:
    if ref.kind != "cell":
        return True
    exp = expected_dtypes(ex.question, cues)
    return exp is None or ex.table.rows[ref.row][ref.col].dtype in exp


class RewardFn:
    def __init__(self, cfg: dict):
        r = cfg.get("reward", cfg)
        self.w = {k: float(r[k]) for k in ("w_ans", "w_hop", "w_red", "w_schema", "w_step", "w_stop", "w_path")}
        self.path_f1_terminal = bool(r["path_f1_terminal"])
        if r["terminal_only"]:
            for k in ("w_hop", "w_red", "w_schema", "w_step"):
                self.w[k] = 0.0
        for flag, k in (("no_redundancy", "w_red"), ("no_schema", "w_schema"), ("no_step", "w_step")):
            if r[flag]:
                self.w[k] = 0.0

    def hop(self, info, example: Example) -> float:
        """Reward of one hop (HopInfo). STOP / illegal steps return 0."""
        if info.entity is None or not info.legal:
            return 0.0
        w = self.w
        gold_ok = example.gold_path_quality != "none" and example.gold_path
        return (w["w_hop"] * float(bool(info.on_gold and gold_ok))
                - w["w_red"] * float(info.redundant) - w["w_step"])

    def terminal(self, pred: str, example: Example, trace: list) -> float:
        """Terminal reward; `trace` = list of HopInfo of the selected entities."""
        w = self.w
        r = w["w_ans"] * f1(pred, example.answer)
        ents = [h.entity for h in trace if h.entity is not None]
        if trace and not trace[-1].schema_ok:
            r -= w["w_schema"]
        gold = list(example.gold_path or [])
        if gold and example.gold_path_quality != "none":
            if w["w_stop"] and set(gold) <= set(ents):
                r += w["w_stop"]
            if self.path_f1_terminal:
                r += w["w_path"] * gold_set_f1(ents, gold)
        return r
