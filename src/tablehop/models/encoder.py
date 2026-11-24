"""Frozen sentence encoder with an on-disk embedding cache (specs/07).

Vectors are L2-normalised float16, keyed by sha1 of the text (questions get a "Q:" prefix), so identical
entity texts shared across tables/splits are embedded once. One `<split>.npy` (n, D) + `<split>.keys.json`
per split in `cache_dir`; all existing split files are loaded on construction.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from tablehop.data.schema import Example
from tablehop.env.render import render_cell, render_passage

Encoder = Callable[[list[str]], np.ndarray]


def text_key(text: str, question: bool = False) -> str:
    return hashlib.sha1((("Q:" if question else "") + text).encode()).hexdigest()


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def make_st_encoder(model: str, device: str, batch_size: int = 64, max_len: int = 160, pad_to: int = 32) -> Encoder:
    """Plain transformers encoder, CLS pooling (bge). Sequences are padded to a multiple of `pad_to` so MPS
    sees few distinct shapes (variable shapes made MPS ~25x slower). Returns (n, D) float32."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModel.from_pretrained(model).to(device).eval()

    @torch.no_grad()
    def enc(texts: list[str]) -> np.ndarray:
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), m.config.hidden_size), dtype=np.float32)
        for i in range(0, len(texts), batch_size):
            idx = order[i : i + batch_size]
            b = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len,
                    pad_to_multiple_of=pad_to, return_tensors="pt").to(device)
            out[idx] = m(**b).last_hidden_state[:, 0].float().cpu().numpy()
        return out

    return enc


class EmbeddingCache:
    """embed(texts) -> (n, D) float32 L2-normalised; recomputes nothing already cached."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", cache_dir: str | Path | None = None,
                 device: str = "cpu", encoder: Encoder | None = None, batch_size: int = 256):
        self.model, self.device, self.batch_size = model, device, batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._encoder = encoder
        self.store: dict[str, np.ndarray] = {}
        self.n_computed = 0
        if self.cache_dir and self.cache_dir.exists():
            for kf in sorted(self.cache_dir.glob("*.keys.json")):
                npy = kf.with_name(kf.name.replace(".keys.json", ".npy"))
                if npy.exists():
                    arr = np.load(npy)
                    for k, v in zip(json.loads(kf.read_text()), arr):
                        self.store[k] = v

    def _enc(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            self._encoder = make_st_encoder(self.model, self.device, self.batch_size)
        return self._encoder(texts)

    def embed(self, texts: list[str], question: bool = False) -> np.ndarray:
        keys = [text_key(t, question) for t in texts]
        miss: dict[str, str] = {k: t for k, t in zip(keys, texts) if k not in self.store}
        if miss:
            vecs = _normalize(np.asarray(self._enc(list(miss.values())), dtype=np.float32)).astype(np.float16)
            self.store.update(zip(miss.keys(), vecs))
            self.n_computed += len(miss)
        return np.stack([self.store[k] for k in keys]).astype(np.float32)

    def example_embs(self, ex: Example) -> ExampleEmbs:
        """Embeddings of the question, every cell (R, C, D) and every passage of `ex`."""
        t = ex.table
        q = self.embed([ex.question], question=True)[0]
        texts = [render_cell(t, r, c) for r in range(len(t.rows)) for c in range(len(t.rows[r]))]
        cells = self.embed(texts)
        ncol = len(t.rows[0]) if t.rows else 0
        pids = list(ex.passages)
        pe = self.embed([render_passage(p, ex.passages) for p in pids]) if pids else np.zeros((0, q.shape[0]), np.float32)
        return ExampleEmbs(q, cells.reshape(len(t.rows), ncol, -1), dict(zip(pids, pe)))

    def build_split(self, split: str, examples: Iterable[Example], budget_s: float | None = None,
                    chunk: int = 4096) -> int:
        """Embed everything needed by `examples` and write the split file; returns #unique keys."""
        keys: dict[str, str] = {}
        for ex in examples:
            keys[text_key(ex.question, True)] = ex.question
            for r, row in enumerate(ex.table.rows):
                for c in range(len(row)):
                    s = render_cell(ex.table, r, c)
                    keys[text_key(s)] = s
            for p in ex.passages:
                s = render_passage(p, ex.passages)
                keys[text_key(s)] = s
        miss = {k: s for k, s in keys.items() if k not in self.store}
        t0, items = time.time(), list(miss.items())
        for i in range(0, len(items), chunk):
            if budget_s is not None and time.time() - t0 > budget_s:
                self._save("_partial", list(self.store))  # resumable checkpoint
                raise TimeoutError(f"budget hit; {len(items) - i} texts left, rerun to resume")
            part = items[i : i + chunk]
            vecs = _normalize(np.asarray(self._enc([t for _, t in part]), dtype=np.float32)).astype(np.float16)
            self.store.update(zip((k for k, _ in part), vecs))
            self.n_computed += len(part)
            done = i + len(part)
            print(f"  [{split}] {done}/{len(items)} texts, {done / max(time.time() - t0, 1e-9):.0f}/s", flush=True)
        self._save(split, list(keys))
        (self.cache_dir / "_partial.npy").unlink(missing_ok=True) if self.cache_dir else None
        (self.cache_dir / "_partial.keys.json").unlink(missing_ok=True) if self.cache_dir else None
        return len(keys)

    def _save(self, name: str, ks: list[str]) -> None:
        if self.cache_dir and ks:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / f"{name}.npy", np.stack([self.store[k] for k in ks]))
            (self.cache_dir / f"{name}.keys.json").write_text(json.dumps(ks))


class ExampleEmbs:
    """q (D,), cells (R, C, D), passages {passage_id: (D,)} -- all float32 unit norm."""

    def __init__(self, q: np.ndarray, cells: np.ndarray, passages: dict[str, np.ndarray]):
        self.q, self.cells, self.passages = q, cells, passages
