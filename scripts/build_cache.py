"""Embed questions + all entity texts for every split into data/cache/ and report hop-0 recall@K.

usage: python scripts/build_cache.py [--config configs/base.yaml] [--splits train dev ...] [--device mps]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from tablehop.data.schema import Example
from tablehop.env.candidates import CandidateGen, first_gold_in_hop0
from tablehop.models.encoder import EmbeddingCache
from tablehop.utils.config import get_config
from tablehop.utils.device import get_device

SPLITS = ["train", "dev", "test", "ood_dev", "ood_test"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--budget", type=float, default=None, help="seconds per split before checkpoint+exit(3); rerun resumes")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N examples per split")
    a = ap.parse_args()
    cfg = get_config(a.config)
    device = get_device(a.device or cfg["device"])
    cache_dir = Path(cfg["paths"]["cache_dir"])
    proc = Path(cfg["paths"]["data_dir"]) / "processed"
    cache = EmbeddingCache(cfg["env"]["encoder"], cache_dir, device, batch_size=int(cfg["env"]["embed_batch"]))
    rows = []
    for sp in a.splits:
        f = proc / f"{sp}.jsonl"
        if not f.exists():
            print(f"[skip] {f} missing")
            continue
        exs = [Example.from_json(line) for line in f.read_text().splitlines() if line.strip()]
        exs = exs[: a.limit] if a.limit else exs
        t0 = time.time()
        try:
            n_keys = cache.build_split(sp, exs, budget_s=a.budget)
        except TimeoutError as e:
            print(f"{sp}: {e}")
            raise SystemExit(3) from e
        dt = time.time() - t0
        hits = []
        for ex in exs:
            h = first_gold_in_hop0(CandidateGen(ex, cache.example_embs(ex), cfg["env"]))
            if h is not None:
                hits.append(h)
        rec = sum(hits) / max(len(hits), 1)
        print(f"{sp}: examples={len(exs)} unique_texts={n_keys} embed_time={dt:.1f}s "
              f"recall@{cfg['env']['K']}={rec:.4f} (n_with_gold={len(hits)})")
        rows.append({"split": sp, "n_examples": len(exs), "n_with_gold": len(hits),
                     f"recall_at_{cfg['env']['K']}": round(rec, 4), "embed_seconds": round(dt, 1)})
    out = Path(cfg["paths"]["results_dir"]) / "tables" / "candidate_recall.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {out}; newly embedded texts: {cache.n_computed}")


if __name__ == "__main__":
    main()
