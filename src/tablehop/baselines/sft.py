"""B1: SFT baseline. flan-t5-small fine-tuned on question + BM25-ranked linearized table rows."""
from __future__ import annotations

import math
import random
import re
import time
from pathlib import Path

import torch

from tablehop.data.load import load_split
from tablehop.data.schema import Example
from tablehop.env.render import passage_title
from tablehop.eval.metrics import aggregate
from tablehop.models.answer_head import generate_texts
from tablehop.utils.device import get_device
from tablehop.utils.logging import CSVLogger, get_logger
from tablehop.utils.runmeta import new_run_dir
from tablehop.utils.seeding import set_seed

log = get_logger("sft")
_TOK = re.compile(r"\w+")


def _toks(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def linearize_rows(ex: Example, passage_chars: int = 200) -> list[str]:
    """Each row: "h1: v1 ; h2: v2 ..." plus first `passage_chars` chars of linked passages in the row."""
    rows = []
    for row in ex.table.rows:
        parts = [f"{ex.table.header[c.col] if c.col < len(ex.table.header) else ''}: {c.text}"
                 for c in row]
        s = " ; ".join(parts)
        for c in row:
            if c.link and c.link in ex.passages:
                s += f" [{passage_title(c.link)}: {ex.passages[c.link][:passage_chars]}]"
        rows.append(s)
    return rows


def build_context(ex: Example, passage_chars: int = 200) -> str:
    """Input text: question + rows ranked by BM25 against the question (truncated later to max_len)."""
    from rank_bm25 import BM25Okapi

    rows = linearize_rows(ex, passage_chars)
    if not rows:
        return f"question: {ex.question} context: "
    bm = BM25Okapi([_toks(r) or ["_"] for r in rows])
    scores = bm.get_scores(_toks(ex.question))
    order = sorted(range(len(rows)), key=lambda i: (-scores[i], i))
    return f"question: {ex.question} context: " + " || ".join(rows[i] for i in order)


def predict(model, tok, exs: list[Example], device: str, scfg: dict) -> list[str]:
    texts = [build_context(e, scfg["passage_chars"]) for e in exs]
    return generate_texts(model, tok, texts, device, scfg["eval_batch_size"], scfg["max_len"],
                          scfg["max_new_tokens"])


def _dev_metrics(model, tok, dev, device, scfg) -> dict:
    sub = dev[: scfg["eval_subset"]]
    preds = predict(model, tok, sub, device, scfg)
    return aggregate([{"pred": p, "gold": e.answer} for p, e in zip(preds, sub)])


def train(cfg: dict, seed: int) -> Path:
    """Run B1 training; returns the run dir (results/runs/<id>/ with checkpoint/ HF dir)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    scfg = dict(cfg["sft"])
    set_seed(seed)
    rng = random.Random(seed)
    device = get_device(cfg.get("device", "auto"))
    rm = new_run_dir("b1_sft", cfg["data"].get("dataset", "hybridqa"), seed, cfg,
                     cfg.get("paths", {}).get("results_dir", "results"))
    run_dir = rm.run_dir
    train_ex = [e for e in load_split(cfg["data"], "train") if e.answer]
    dev = load_split(cfg["data"], "dev")
    tok = AutoTokenizer.from_pretrained(scfg["model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(scfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=scfg["lr"])
    bs = scfg["batch_size"]
    total = math.ceil(scfg["budget_examples"] / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total))
    ctx = [build_context(e, scfg["passage_chars"]) for e in train_ex]
    logger = CSVLogger(run_dir)
    seen, step, t0, losses, next_eval = 0, 0, time.time(), [], scfg["eval_every_examples"]
    order: list[int] = []
    model.train()
    while step < total:
        if len(order) < bs:
            order = list(range(len(train_ex)))
            rng.shuffle(order)
        idx, order = order[:bs], order[bs:]
        opt.zero_grad()
        tot, mb = 0.0, scfg.get("micro_batch", bs)
        for ms in range(0, len(idx), mb):
            sub = idx[ms:ms + mb]
            enc = tok([ctx[i] for i in sub], return_tensors="pt", padding=True,
                      pad_to_multiple_of=64, truncation=True, max_length=scfg["max_len"]).to(device)
            lab = tok([train_ex[i].answer[0] for i in sub], return_tensors="pt", padding=True,
                      truncation=True, max_length=scfg["max_new_tokens"]).input_ids
            lab[lab == tok.pad_token_id] = -100
            loss = model(**enc, labels=lab.to(device)).loss * (len(sub) / len(idx))
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device == "mps" and step % 20 == 0:
            torch.mps.empty_cache()
        sched.step()
        step += 1
        seen += len(idx)
        losses.append(tot)
        if step % 50 == 0:
            log.info("step %d/%d loss %.4f (%.0fs)", step, total, sum(losses[-50:]) / 50,
                     time.time() - t0)
        if seen >= next_eval or step == total:
            next_eval += scfg["eval_every_examples"]
            m = _dev_metrics(model, tok, dev, device, scfg)
            logger.log({"step": step, "examples_seen": seen, "train_loss": sum(losses[-50:]) / len(losses[-50:]),
                        "dev_em_subset": m["em"], "dev_f1_subset": m["f1"],
                        "elapsed_s": time.time() - t0})
            log.info("examples %d dev-subset EM %.3f F1 %.3f", seen, m["em"], m["f1"])
            model.train()
    model.save_pretrained(run_dir / "checkpoint")
    tok.save_pretrained(run_dir / "checkpoint")
    rm.meta["train_time_s"] = time.time() - t0
    rm.meta["examples_seen"] = seen
    rm.finish("done")
    return run_dir
