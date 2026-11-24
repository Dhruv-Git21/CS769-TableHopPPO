"""Answer head interface/caching/formatting with a stub model (CPU, no downloads), plus evaluate guard."""
import json

import pytest

from tablehop.baselines.sft import build_context
from tablehop.data.synthetic import generate
from tablehop.models.answer_head import AnswerHead, build_input


class StubTok:
    """Tokenizer stub: records inputs; 'tokens' are chars."""
    def __init__(self):
        self.seen = []

    def __call__(self, texts, **kw):
        self.seen.append(list(texts))
        return _Enc(texts)

    def batch_decode(self, gen, skip_special_tokens=True):
        return list(gen)


class _Enc(dict):
    def __init__(self, texts):
        super().__init__(texts=texts)

    def to(self, device):
        return self


class StubModel:
    def __init__(self):
        self.calls = 0
        self.n_inputs = 0

    def parameters(self):
        return iter([])

    def to(self, d):
        return self

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    @property
    def training(self):
        return False

    def generate(self, texts, **kw):
        self.calls += 1
        self.n_inputs += len(texts)
        return [f"ans:{t.split('evidence:')[1].strip()}" for t in texts]


def make():
    m, t = StubModel(), StubTok()
    return AnswerHead(m, t, "cpu", cache_size=3, batch_size=2), m, t


def test_input_format():
    assert build_input("who?", ["a | b", "c"]) == "question: who? evidence: a | b || c"


def test_batched_answer_and_cache():
    h, m, _t = make()
    out = h.answer([("e1", "q", ["x"]), ("e2", "q", ["y"]), ("e1", "q", ["x"])])
    assert out == ["ans:x", "ans:y", "ans:x"]
    assert m.n_inputs == 2 and m.calls == 1  # duplicate deduped within batch
    h.answer([("e1", "q", ["x"])])
    assert m.n_inputs == 2 and h.hits >= 1  # cache hit
    h.answer([("e1", "q", ["x", "z"])])  # different evidence -> miss
    assert m.n_inputs == 3


def test_lru_eviction():
    h, m, _ = make()
    for i in range(4):
        h.answer([(f"e{i}", "q", ["x"])])
    n = m.n_inputs
    h.answer([("e3", "q", ["x"])])
    assert m.n_inputs == n          # recent kept
    h.answer([("e0", "q", ["x"])])
    assert m.n_inputs == n + 1      # oldest evicted


def test_answer_refs_uses_render():
    h, _, t = make()
    ex = generate(1, 0)[0]
    h.answer_refs([(ex, ex.gold_path)])
    sent = t.seen[-1][0]
    assert sent.startswith(f"question: {ex.question} evidence: Synthetic roster 0 | Player |")
    assert sent.count(" || ") == len(ex.gold_path) - 1


def test_sft_context_ranks_matching_row_first():
    ex = generate(1, 0)[0]
    r = ex.gold_path[0].row
    player = ex.table.rows[r][0].text
    ctx = build_context(ex, 50)
    assert ctx.startswith(f"question: {ex.question} context: ")
    assert player in ctx.split(" context: ")[1].split(" || ")[0]


def test_test_guard(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from evaluate import log_test, test_guard
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    log = tmp_path / "TEST_LOG.md"
    test_guard(run, "test", log)  # first time ok
    log_test(run, "test", {"em": 0.1, "f1": 0.2}, log)
    with pytest.raises(SystemExit):
        test_guard(run, "test", log)
    test_guard(run, "ood_test", log)  # other split still allowed
    (run / "metrics_ood_test.json").write_text(json.dumps({}))
    with pytest.raises(SystemExit):
        test_guard(run, "ood_test", log)
