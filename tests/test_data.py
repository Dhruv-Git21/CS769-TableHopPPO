import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tablehop.data import hybridqa
from tablehop.data.schema import EntityRef, Example
from tablehop.data.splits import assign_splits, audit_splits, choose_heldout
from tablehop.data.synthetic import generate

FIX = Path(__file__).parent / "fixtures" / "hybridqa"
ROOT = Path(__file__).parent.parent


def _ex(qid):
    qs = {q["question_id"]: q for q in hybridqa.load_questions(FIX, "train")}
    return hybridqa.build_example(FIX, qs[qid], "train", "train")


def test_gold_path_table_answer():
    e = _ex("q1")
    assert e.answer_type == "table" and e.gold_path_quality == "heuristic"
    assert e.gold_path == [EntityRef("cell", "Mosques_0", 0, 0), EntityRef("cell", "Mosques_0", 0, 1)]


def test_gold_path_passage_answer():
    e = _ex("q2")
    assert e.answer_type == "passage"
    assert e.gold_path == [
        EntityRef("cell", "Mosques_0", 1, 0), EntityRef("cell", "Mosques_0", 1, 1),
        EntityRef("passage", "Mosques_0", 1, 1, "/wiki/Herat"),
    ]
    assert "/wiki/Herat" in e.passages


def test_gold_path_none_and_roundtrip():
    e = _ex("q3")
    assert e.gold_path is None and e.gold_path_quality == "none"
    e2 = _ex("q2")
    assert Example.from_json(e2.to_json()) == e2


def test_loader_refuses_test():
    with pytest.raises(ValueError):
        hybridqa.load_questions(FIX, "test")


def test_dtype():
    assert hybridqa.infer_dtype("1,200") == "number"
    assert hybridqa.infer_dtype("1999") == "date"
    assert hybridqa.infer_dtype("Kabul") == "text"


def test_synthetic_deterministic():
    a = [e.to_json() for e in generate(20, 3)]
    b = [e.to_json() for e in generate(20, 3)]
    c = [e.to_json() for e in generate(20, 4)]
    assert a == b and a != c


def test_synthetic_paths_reachable():
    kinds = set()
    for e in generate(60, 0):
        kinds.add(e.meta["template"])
        p = e.gold_path
        assert 2 <= len(p) <= 3
        for u, v in itertools.pairwise(p):
            if v.kind == "passage":
                assert e.table.rows[u.row][u.col].link == v.passage_id
                assert v.passage_id in e.passages
            else:
                assert u.row == v.row or u.col == v.col
        last = p[-1]
        txt = (e.passages[last.passage_id] if last.kind == "passage"
               else e.table.rows[last.row][last.col].text)
        assert e.answer[0] in txt
    assert kinds == {"2hop", "3hop_row", "3hop_passage"}


def test_choose_heldout_and_split_disjoint():
    sizes = {0: 100, 1: 5, 2: 6, 3: 4, 4: 50}
    assert choose_heldout(sizes, 100, 3, 0.05) == [1, 2, 3]
    rng = np.random.RandomState(0)
    tabs = [f"t{i}" for i in range(200)]
    cluster_of = {t: int(rng.randint(0, 5)) for t in tabs}
    tq = [{"question_id": f"a{i}", "table_id": tabs[i % 200], "answer-node": []} for i in range(800)]
    dq = [{"question_id": f"b{i}", "table_id": tabs[i % 200], "answer-node": []} for i in range(400)]
    sp = assign_splits(tq, dq, cluster_of, [0], n_train=100, n_dev=50, n_ood_dev=20)
    names = list(sp)
    tabsets = {s: {q["table_id"] for q in sp[s]} for s in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not tabsets[a] & tabsets[b], (a, b)
    assert all(cluster_of[t] == 0 for s in ("ood_test", "ood_dev") for t in tabsets[s])
    assert all(cluster_of[t] != 0 for s in ("train", "dev", "test") for t in tabsets[s])


def _write_toy(d: Path, leak: bool):
    (d / "processed").mkdir(parents=True)
    tmp = _ex("q1")
    for s, tid, src, dom in [("train", "T1", "train", "c1"), ("dev", "T2", "train", "c1"),
                             ("test", "T1" if leak else "T3", "dev", "c1"),
                             ("ood_test", "T4", "dev", "c2"), ("ood_dev", "T5", "train", "c2")]:
        row = json.loads(tmp.to_json())
        row["table"]["id"], row["domain"], row["id"] = tid, dom, f"{s}-1"
        row["meta"] = {"source": src}
        (d / "processed" / f"{s}.jsonl").write_text(json.dumps(row) + "\n")
    (d / "MANIFEST.json").write_text(json.dumps({"heldout_clusters": [2], "official_test_used": False}))


@pytest.mark.parametrize("leak,code", [(False, 0), (True, 1)])
def test_audit_script(tmp_path, leak, code):
    _write_toy(tmp_path, leak)
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_leakage.py"), "--processed",
         str(tmp_path / "processed"), "--manifest", str(tmp_path / "MANIFEST.json")],
        check=False, capture_output=True, text=True)
    assert r.returncode == code, r.stdout
    assert ("FAIL table_id_disjoint" in r.stdout) == leak
    assert "PASS cluster_disjoint" in r.stdout
    assert all(ok for _, ok, _ in audit_splits(tmp_path / "processed", tmp_path / "MANIFEST.json")) != leak
