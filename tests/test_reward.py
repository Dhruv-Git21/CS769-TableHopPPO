import pytest

from tablehop.data.synthetic import generate
from tablehop.env.table_env import HopInfo
from tablehop.rl.reward import RewardFn, expected_dtypes, schema_ok
from tablehop.utils.config import get_config

EX = generate(1, 0)[0]
E0 = EX.gold_path[0]
BAD = HopInfo(E0, False, True, False, True)  # off-gold, redundant, schema-bad
GOOD = HopInfo(E0, True, False, True, True)


def fn(**kw):
    c = get_config("configs/base.yaml")
    c["reward"].update(kw)
    return RewardFn(c)


@pytest.mark.parametrize("kw,info,expected", [
    ({}, GOOD, 0.2 - 0.02),
    ({}, BAD, -0.1 - 0.02),
    ({"terminal_only": True}, GOOD, 0.0),
    ({"terminal_only": True}, BAD, 0.0),
    ({"no_redundancy": True}, BAD, -0.02),
    ({"no_step": True}, GOOD, 0.2),
    ({"no_step": True}, BAD, -0.1),
    ({"no_schema": True}, GOOD, 0.18),
])
def test_hop_terms(kw, info, expected):
    assert fn(**kw).hop(info, EX) == pytest.approx(expected)


def test_stop_and_illegal_hops_zero():
    assert fn().hop(HopInfo(None, False, False, True, True, True), EX) == 0.0
    assert fn().hop(HopInfo(E0, True, False, True, False), EX) == 0.0


def test_no_gold_path_zero_hop_term():
    ex = generate(1, 0)[0]
    ex.gold_path_quality = "none"
    assert fn().hop(GOOD, ex) == pytest.approx(-0.02)


@pytest.mark.parametrize("kw,trace,pred,expected", [
    ({}, [GOOD], EX.answer[0], 1.0),
    ({}, [GOOD], "zzz", 0.0),
    ({}, [BAD], EX.answer[0], 1.0 - 0.05),
    ({"no_schema": True}, [BAD], EX.answer[0], 1.0),
    ({"terminal_only": True}, [BAD], EX.answer[0], 1.0),
    ({"path_f1_terminal": True}, [GOOD], EX.answer[0], 1.0 + 0.2 * (2 * 0.5 * 1 / 1.5 if len(EX.gold_path) == 2 else 0.5)),
])
def test_terminal(kw, trace, pred, expected):
    assert fn(**kw).terminal(pred, EX, trace) == pytest.approx(expected, abs=1e-6)


def test_path_f1_only_when_enabled_and_stop_bonus():
    full = [HopInfo(e, True, False, True, True) for e in EX.gold_path]
    assert fn().terminal("zzz", EX, full) == 0.0
    assert fn(path_f1_terminal=True).terminal("zzz", EX, full) == pytest.approx(0.2)
    assert fn(w_stop=0.3).terminal("zzz", EX, full) == pytest.approx(0.3)
    assert fn(w_stop=0.3).terminal("zzz", EX, full[:1]) == 0.0


def test_schema_rule():
    cues = get_config("configs/base.yaml")["reward"]["schema_cues"]
    assert expected_dtypes("How many goals did he score?", cues) == {"number"}
    assert expected_dtypes("In what year was it founded?", cues) == {"date", "number"}
    assert expected_dtypes("Who won?", cues) is None
    ex = generate(1, 0)[0]
    ex.question = "How many people live there?"
    assert not schema_ok(ex.gold_path[0], ex, cues)  # text cell, number expected
    assert schema_ok(ex.gold_path[-1].__class__("passage", "t", 0, 0, "/wiki/x"), ex, cues)
