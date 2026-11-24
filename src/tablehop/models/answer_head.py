"""Shared frozen answer head (google/flan-t5-small), specs/07 "Answer head".

Input  : "question: {q} evidence: {e1} || {e2} ..."  (texts from env/render.py, max 512 tokens)
Output : greedy answer string (max 32 new tokens).
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

from tablehop.data.schema import EntityRef, Example
from tablehop.env.render import render_evidence
from tablehop.utils.device import get_device

BASE_MODEL = "google/flan-t5-small"


def build_input(question: str, evidence: Sequence[str]) -> str:
    return f"question: {question} evidence: {' || '.join(evidence)}"


def generate_texts(model, tok, texts: list[str], device: str, batch_size: int = 32,
                   max_len: int = 512, max_new_tokens: int = 32) -> list[str]:
    """Batched greedy generation (length-sorted for speed); returns outputs in input order."""
    import torch

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [""] * len(texts)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len, pad_to_multiple_of=64).to(device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
            for i, g in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = g.strip()
    model.train(was_training)
    if device == "mps":
        torch.mps.empty_cache()
    return out


def sha256_of_dir(path: str | Path) -> str:
    """sha256 of the weights file (model.safetensors, else pytorch_model.bin)."""
    path = Path(path)
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (path / name).exists():
            h = hashlib.sha256()
            with open(path / name, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
    return "missing"


class AnswerHead:
    """Frozen answer generator with an in-memory LRU cache keyed by (example id, evidence texts)."""

    def __init__(self, model, tokenizer, device: str = "cpu", cache_size: int = 100_000,
                 batch_size: int = 32, max_len: int = 512, max_new_tokens: int = 32,
                 sha256: str = "unknown", evidence_mode: str = "entity"):
        self.model, self.tok, self.device = model, tokenizer, device
        self.batch_size, self.max_len, self.max_new_tokens = batch_size, max_len, max_new_tokens
        self.sha256, self.evidence_mode = sha256, evidence_mode
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self.hits = self.misses = 0
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(device).eval()

    @classmethod
    def load(cls, path: str | Path, device: str = "auto", **kw) -> AnswerHead:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        dev = get_device(device)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(path))
        tok = AutoTokenizer.from_pretrained(str(path))
        mf = Path(path) / "evidence_mode.txt"  # written by train_answer_head.py (R3); absent = "entity"
        mode = mf.read_text().strip() if mf.exists() else "entity"
        return cls(model, tok, dev, sha256=sha256_of_dir(path), evidence_mode=mode, **kw)

    def _get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def answer(self, batch: Sequence[tuple[str, str, Sequence[str]]]) -> list[str]:
        """batch of (example_id, question, evidence texts) -> answer strings (cached, batched)."""
        keys = [(ex_id, tuple(ev)) for ex_id, _, ev in batch]
        results: list[str | None] = [None] * len(batch)
        todo: dict = {}
        for i, key in enumerate(keys):
            hit = self._get(key)
            if hit is not None:
                results[i] = hit
                self.hits += 1
            else:
                self.misses += 1
                todo.setdefault(key, i)
        if todo:
            uniq = list(todo.items())
            texts = [build_input(batch[i][1], batch[i][2]) for _, i in uniq]
            outs = generate_texts(self.model, self.tok, texts, self.device, self.batch_size,
                                  self.max_len, self.max_new_tokens)
            for (key, _), o in zip(uniq, outs):
                self._cache[key] = o
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            for i, key in enumerate(keys):
                if results[i] is None:
                    results[i] = self._cache[key]
        return results  # type: ignore[return-value]

    def answer_refs(self, items: Sequence[tuple[Example, Sequence[EntityRef]]]) -> list[str]:
        """Convenience: render EntityRefs of each example with render.py, then answer."""
        return self.answer([(ex.id, ex.question, render_evidence(list(refs), ex, self.evidence_mode))
                            for ex, refs in items])
