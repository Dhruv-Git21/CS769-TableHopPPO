"""Answer and path metrics (spec 04). Pure python, no env/model dependency."""
from __future__ import annotations

import collections
import re
import string
from collections.abc import Hashable, Sequence


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, articles and extra whitespace.

    Logic copied from the official HybridQA evaluate_script.py
    (https://github.com/wenhuchen/HybridQA), itself SQuAD-style.
    """
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _golds(golds: str | Sequence[str]) -> list[str]:
    return [golds] if isinstance(golds, str) else list(golds)


def _exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def _f1(gold: str, pred: str) -> float:
    g = normalize_answer(gold).split() if gold else []
    p = normalize_answer(pred).split() if pred else []
    if not g or not p:
        return float(int(g == p))
    same = sum((collections.Counter(g) & collections.Counter(p)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, golds: str | Sequence[str]) -> float:
    """Max EM over gold answers."""
    return float(max(_exact(g, pred) for g in _golds(golds)))


def f1(pred: str, golds: str | Sequence[str]) -> float:
    """Max token F1 over gold answers."""
    return float(max(_f1(g, pred) for g in _golds(golds)))


def path_validity(trace_info: Sequence[dict], gold: Sequence[Hashable]) -> tuple[float, float]:
    """(strict, loose) over hops; trace_info items: {entity, legal, redundant}.

    loose = legal and not redundant; strict = loose and entity in gold path.
    Empty trace gives (0.0, 0.0).
    """
    if not trace_info:
        return 0.0, 0.0
    gold_set = set(gold)
    loose = [bool(t["legal"]) and not t["redundant"] for t in trace_info]
    strict = [ok and t["entity"] in gold_set for ok, t in zip(loose, trace_info)]
    n = len(trace_info)
    return sum(strict) / n, sum(loose) / n


def gold_set_f1(selected: Sequence[Hashable], gold: Sequence[Hashable]) -> float:
    """Set F1 between selected entities and gold path entities."""
    s, g = set(selected), set(gold)
    if not s or not g:
        return float(s == g)
    inter = len(s & g)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(s), inter / len(g)
    return 2 * prec * rec / (prec + rec)


def gold_lcs(selected: Sequence[Hashable], gold: Sequence[Hashable]) -> float:
    """Longest common subsequence length divided by gold length."""
    if not gold:
        return 0.0
    prev = [0] * (len(gold) + 1)
    for a in selected:
        cur = [0]
        for j, b in enumerate(gold):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1] / len(gold)


def hop_efficiency(n_hops: int, gold_len: int, correct: bool) -> float:
    """min_hops / hops_taken; 0 if incorrect; capped at 1.0 (incl. fewer hops than gold)."""
    if not correct:
        return 0.0
    if n_hops <= 0 or n_hops <= gold_len:
        return 1.0
    return gold_len / n_hops


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(predictions: list[dict]) -> dict:
    """Spec-04 metric keys from per-example dicts.

    Each dict: pred, gold (str or list), correct (bool, optional; defaults to EM==1),
    path (list of entities), trace_info (list of dicts), gold_path (list or None/empty).
    Path metrics use only examples with a non-empty gold_path.
    """
    em, f1s, hops = [], [], []
    sv, lv, gf, gl, he = [], [], [], [], []
    n_gp = 0
    for p in predictions:
        e = exact_match(p["pred"], p["gold"])
        em.append(e)
        f1s.append(f1(p["pred"], p["gold"]))
        path = list(p.get("path") or [e_["entity"] for e_ in p.get("trace_info") or []])
        hops.append(len(path))
        gp = p.get("gold_path")
        if not gp:
            continue
        n_gp += 1
        s, lo = path_validity(p.get("trace_info") or [], gp)
        sv.append(s)
        lv.append(lo)
        gf.append(gold_set_f1(path, gp))
        gl.append(gold_lcs(path, gp))
        he.append(hop_efficiency(len(path), len(gp), bool(p.get("correct", e == 1.0))))
    return {
        "em": _mean(em), "f1": _mean(f1s),
        "path_validity_strict": _mean(sv), "path_validity_loose": _mean(lv),
        "gold_f1": _mean(gf), "gold_lcs": _mean(gl), "hop_efficiency": _mean(he),
        "mean_hops": _mean([float(h) for h in hops]),
        "n": len(predictions), "n_with_gold_path": n_gp,
    }
