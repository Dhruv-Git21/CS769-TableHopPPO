"""Train the single frozen answer head (flan-t5-small) on gold-path evidence + distractors.

Usage: python scripts/train_answer_head.py [--config configs/answer_head.yaml] [--override k=v ...]
Outputs results/answer_head/{model files, sha256.txt, train_log.csv, dev_metrics.json}.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from tablehop.data.load import load_split
from tablehop.data.schema import EntityRef, Example
from tablehop.env.render import render_evidence
from tablehop.eval.metrics import aggregate
from tablehop.models.answer_head import build_input, generate_texts, sha256_of_dir
from tablehop.utils.config import get_config
from tablehop.utils.device import get_device
from tablehop.utils.logging import CSVLogger, get_logger
from tablehop.utils.seeding import set_seed

log = get_logger("answer_head")


def all_entities(ex: Example) -> list[EntityRef]:
    """Every cell and linked passage of the example's table (distractor pool)."""
    out = [EntityRef("cell", ex.table.id, r, c) for r, row in enumerate(ex.table.rows)
           for c in range(len(row))]
    out += [EntityRef("passage", ex.table.id, -1, -1, pid) for pid in ex.passages]
    return out


def train_evidence(ex: Example, rng: random.Random, p: float, kmax: int, mode: str = "entity") -> list[str]:
    """Gold path entities, plus 1..kmax random distractors with prob. p, shuffled."""
    refs = list(ex.gold_path or [])
    if rng.random() < p:
        gold = set(refs)
        pool = [e for e in all_entities(ex) if e not in gold]
        refs += rng.sample(pool, min(len(pool), rng.randint(1, kmax)))
    rng.shuffle(refs)
    return render_evidence(refs, ex, mode)


def oracle_texts(exs: list[Example], mode: str = "entity") -> list[str]:
    return [build_input(ex.question, render_evidence(list(ex.gold_path or []), ex, mode)) for ex in exs]


def eval_oracle(model, tok, exs: list[Example], device: str, cfg) -> dict:
    preds = generate_texts(model, tok, oracle_texts(exs, getattr(cfg, "evidence", "entity")), device, cfg.eval_batch_size,
                           cfg.max_len, cfg.max_new_tokens)
    recs = [{"pred": p, "gold": ex.answer} for p, ex in zip(preds, exs)]
    allm = aggregate(recs)
    sub = [r for r, ex in zip(recs, exs) if ex.gold_path]
    subm = aggregate(sub)
    return {"em": allm["em"], "f1": allm["f1"], "n": allm["n"],
            "em_gold_path_subset": subm["em"], "f1_gold_path_subset": subm["f1"],
            "n_gold_path_subset": subm["n"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/answer_head.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = get_config(args.config, args.override)
    ah = cfg.answer_head
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    set_seed(ah.seed)
    rng = random.Random(ah.seed)
    device = get_device(cfg.get("device", "auto"))
    out = Path(ah.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = [e for e in load_split(cfg.data, "train") if e.gold_path]
    dev = load_split(cfg.data, "dev")
    log.info("train=%d (with gold path) dev=%d device=%s", len(train), len(dev), device)

    tok = AutoTokenizer.from_pretrained(ah.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(ah.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ah.lr)
    logger = CSVLogger(out)
    best_f1, bad, t0, step = -1.0, 0, time.time(), 0
    best_dir = out / "_best"
    for epoch in range(1, ah.epochs + 1):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses = []
        for s in range(0, len(order), ah.batch_size):
            batch = [train[i] for i in order[s:s + ah.batch_size]]
            opt.zero_grad()
            tot = 0.0
            mb = ah.micro_batch
            for ms in range(0, len(batch), mb):
                mbatch = batch[ms:ms + mb]
                src = [build_input(e.question, train_evidence(e, rng, ah.distractor_p, ah.distractor_max, getattr(ah, "evidence", "entity")))
                       for e in mbatch]
                enc = tok(src, return_tensors="pt", padding=True, pad_to_multiple_of=64,
                          truncation=True, max_length=ah.max_len).to(device)
                lab = tok([e.answer[0] for e in mbatch], return_tensors="pt", padding=True,
                          truncation=True, max_length=ah.max_new_tokens).input_ids
                lab[lab == tok.pad_token_id] = -100
                loss = model(**enc, labels=lab.to(device)).loss * (len(mbatch) / len(batch))
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if device == "mps" and step % 20 == 0:
                torch.mps.empty_cache()
            step += 1
            losses.append(tot)
            if step % 50 == 0:
                log.info("epoch %d step %d loss %.4f (%.0fs)", epoch, step,
                         sum(losses[-50:]) / 50, time.time() - t0)
        m = eval_oracle(model, tok, dev, device, ah)
        logger.log({"epoch": epoch, "step": step, "train_loss": sum(losses) / len(losses),
                    "dev_em": m["em"], "dev_f1": m["f1"], "elapsed_s": time.time() - t0})
        log.info("epoch %d dev EM %.4f F1 %.4f", epoch, m["em"], m["f1"])
        if m["f1"] > best_f1:
            best_f1, bad = m["f1"], 0
            model.save_pretrained(best_dir)
        else:
            bad += 1
            if bad > ah.patience:
                log.info("early stop")
                break
    train_time = time.time() - t0
    # final = best-dev-F1 epoch
    model = AutoModelForSeq2SeqLM.from_pretrained(str(best_dir)).to(device)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    for f in best_dir.iterdir():
        f.unlink()
    best_dir.rmdir()
    (out / "evidence_mode.txt").write_text(getattr(ah, "evidence", "entity") + "\n")
    sha = sha256_of_dir(out)
    (out / "sha256.txt").write_text(sha + "\n")
    m = eval_oracle(model, tok, dev, device, ah)
    m.update({"split": "dev", "method": "m0_oracle", "sha256": sha, "train_time_s": train_time,
              "device": device, "note": "answer head on gold path entities, no distractors"})
    (out / "dev_metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
