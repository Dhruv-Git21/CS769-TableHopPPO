"""Hand-computed metric tests; EM/F1 cross-checked against the official HybridQA script."""
import ast
import random
from pathlib import Path

import pytest

from tablehop.eval.metrics import (
    aggregate,
    exact_match,
    f1,
    gold_lcs,
    gold_set_f1,
    hop_efficiency,
    path_validity,
)
from tablehop.eval.stats import bootstrap_ci, gap, paired_bootstrap, seed_summary

OFFICIAL = Path(__file__).parents[1] / "data/raw/hybridqa/HybridQA/evaluate_script.py"


def test_em_f1_basic():
    assert exact_match("The Eiffel Tower!", ["eiffel tower"]) == 1.0
    assert f1("a big red apple", ["red apple"]) == pytest.approx(0.8)
    assert f1("x", ["y", "x"]) == 1.0
    assert exact_match("", [""]) == 1.0 and f1("cat", ["dog"]) == 0.0


def test_validity_strict_loose():
    tr = [
        {"entity": "a", "legal": True, "redundant": False},
        {"entity": "b", "legal": True, "redundant": False},
        {"entity": "a", "legal": True, "redundant": True},
        {"entity": "c", "legal": False, "redundant": False},
    ]
    assert path_validity(tr, ["a", "z"]) == (0.25, 0.5)
    assert path_validity([], ["a"]) == (0.0, 0.0)


def test_gold_metrics():
    assert gold_set_f1(["a", "b"], ["b", "c"]) == 0.5
    assert gold_lcs(["a", "x", "b", "c"], ["a", "b", "c", "d"]) == 0.75
    assert gold_lcs(["c", "b"], ["b", "c"]) == 0.5


def test_hop_efficiency():
    assert hop_efficiency(4, 2, True) == 0.5
    assert hop_efficiency(1, 2, True) == 1.0
    assert hop_efficiency(4, 2, False) == 0.0


def test_aggregate():
    preds = [
        {"pred": "Paris", "gold": ["paris"], "correct": True, "path": ["a", "b"],
         "trace_info": [{"entity": "a", "legal": True, "redundant": False},
                        {"entity": "b", "legal": True, "redundant": False}],
         "gold_path": ["a", "b"]},
        {"pred": "x", "gold": "y", "path": ["q"], "trace_info": [], "gold_path": None},
    ]
    m = aggregate(preds)
    assert m["n"] == 2 and m["n_with_gold_path"] == 1
    assert m["em"] == 0.5 and m["path_validity_strict"] == 1.0 and m["hop_efficiency"] == 1.0
    assert m["mean_hops"] == 1.5


def test_stats():
    vals = [0, 1] * 50
    mean, lo, hi = bootstrap_ci(vals, n=2000)
    assert lo <= mean <= hi
    assert paired_bootstrap(vals, vals, n=2000)["p"] == 1.0
    a = [1.0] * 50 + [0.0] * 50
    b = [0.0] * 40 + [1.0] * 10 + [0.0] * 50
    assert paired_bootstrap(a, b, n=2000)["p"] < 0.05
    assert seed_summary([1, 3]) == (2.0, pytest.approx(2**0.5))
    assert gap(0.8, 0.6) == (pytest.approx(0.2), pytest.approx(0.25))


@pytest.mark.skipif(not OFFICIAL.exists(), reason="official script not present")
def test_matches_official():
    src = OFFICIAL.read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.FunctionDef))
            and getattr(n, "name", "") in
            {"", "normalize_answer", "get_tokens", "compute_exact", "compute_f1"}]
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "official", "exec"), ns)  # noqa: S102 (trusted vendored script)
    rng = random.Random(0)
    words = ["The", "a", "Paris", "new", "York,", "42", "an", "Obama.", "river", "of"]
    for _ in range(20):
        g = " ".join(rng.choices(words, k=rng.randint(0, 4)))
        p = " ".join(rng.choices(words, k=rng.randint(0, 4)))
        assert exact_match(p, [g]) == ns["compute_exact"](g, p)
        assert f1(p, [g]) == pytest.approx(ns["compute_f1"](g, p))
