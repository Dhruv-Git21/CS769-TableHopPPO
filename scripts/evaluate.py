"""Evaluate a run: python scripts/evaluate.py --run results/runs/<id> --split dev|test|ood_test|ood_dev
or the M0 oracle: python scripts/evaluate.py --method m0 --split dev [--config configs/m0.yaml].

Writes metrics_<split>.json and predictions_<split>.jsonl in the run dir. test/ood_test evaluation
appends to results/TEST_LOG.md and is refused a second time for the same run and split.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tablehop.data.load import load_split
from tablehop.eval.metrics import aggregate
from tablehop.models.answer_head import AnswerHead
from tablehop.utils.config import get_config
from tablehop.utils.runmeta import env_lock_sha256

TEST_SPLITS = ("test", "ood_test")
PATH_KEYS = ("path_validity_strict", "path_validity_loose", "gold_f1", "gold_lcs", "hop_efficiency")


def test_guard(run_dir: Path, split: str, log_path: Path) -> None:
    """Exit if this run already evaluated on this test split (metrics file or TEST_LOG entry)."""
    tag = f"| {run_dir.name} | {split} |"
    if (run_dir / f"metrics_{split}.json").exists() or (
            log_path.exists() and tag in log_path.read_text()):
        raise SystemExit(f"REFUSED: {run_dir.name} was already evaluated on {split} (see {log_path})")


def log_test(run_dir: Path, split: str, m: dict, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("# TEST_LOG\n\n| time | run | split | EM | F1 |\n|---|---|---|---|---|\n")
    with open(log_path, "a") as f:
        f.write(f"| {time.strftime('%Y-%m-%dT%H:%M:%S')} | {run_dir.name} | {split} | "
                f"{m['em']:.4f} | {m['f1']:.4f} |\n")


def predict_m0(cfg: dict, exs) -> tuple[list[dict], dict]:
    head = AnswerHead.load(cfg["answer_head"]["path"], cfg.get("device", "auto"))
    preds = head.answer_refs([(e, e.gold_path or []) for e in exs])
    recs = []
    for e, p in zip(exs, preds):
        gp = list(e.gold_path or [])
        recs.append({"id": e.id, "question": e.question, "pred": p, "gold": e.answer,
                     "path": gp, "gold_path": gp,
                     "trace_info": [{"entity": g, "legal": True, "redundant": False} for g in gp]})
    return recs, {"answer_head_sha256": head.sha256}


def predict_b1(run_dir: Path, cfg: dict, exs) -> tuple[list[dict], dict]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from tablehop.baselines.sft import predict
    from tablehop.utils.device import get_device

    device = get_device(cfg.get("device", "auto"))
    tok = AutoTokenizer.from_pretrained(str(run_dir / "checkpoint"))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(run_dir / "checkpoint")).to(device).eval()
    preds = predict(model, tok, exs, device, cfg["sft"])
    recs = [{"id": e.id, "question": e.question, "pred": p, "gold": e.answer, "path": [],
             "gold_path": None, "trace_info": []} for e, p in zip(exs, preds)]
    return recs, {}


def predict_b3(run_dir: Path, cfg: dict, exs) -> tuple[list[dict], dict]:
    from tablehop.baselines.soft_region import make_cache, predict
    from tablehop.utils.device import get_device

    head = AnswerHead.load(cfg["answer_head"]["path"], cfg.get("device", "auto"))
    cache = make_cache(cfg, get_device(cfg.get("device", "auto")))
    return predict(run_dir, cfg, exs, head, cache), {"answer_head_sha256": head.sha256}


POLICY_METHODS = ("h1_hoppo", "a1_terminal", "a2_noredund", "a3_noschema", "a4_unrestricted", "b2a_grpo",
                  "b2b_grpo_path")


def predict_policy(run_dir: Path, cfg: dict, exs) -> tuple[list[dict], dict]:
    """Greedy decoding of a PPO/GRPO policy; extra = per-hop validity by hop index + mean reward components."""
    import torch

    from tablehop.rl.reward import RewardFn
    from tablehop.rl.rollout import EnvPool
    from tablehop.rl.runner import build_cache_head, evaluate_policy, make_policy

    cache, head = build_cache_head(cfg)
    ck = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    pol = make_policy(cfg, ck["d_emb"])
    pol.load_state_dict(ck["policy"])
    recs = evaluate_policy(pol, exs, EnvPool(cache, cfg, int(cfg["rl"]["env_pool"])), head, RewardFn(cfg))
    hv: dict[int, list[float]] = {}
    for r in recs:
        gold = set(r["gold_path"] or [])
        if not gold:
            continue
        for k, t in enumerate(r["trace_info"]):
            hv.setdefault(k, []).append(float(t["legal"] and not t["redundant"] and t["entity"] in gold))
    comps = {k: sum(r["comps"][k] for r in recs) / max(1, len(recs)) for k in ("hop", "red", "schema", "step", "terminal")}
    return recs, {"answer_head_sha256": getattr(head, "sha256", "unknown"),
                  "hop_validity_by_index": {str(k): sum(v) / len(v) for k, v in sorted(hv.items())},
                  "hop_count_by_index": {str(k): len(v) for k, v in sorted(hv.items())},
                  "reward_components_mean": comps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--method", choices=["m0"])
    ap.add_argument("--config")
    ap.add_argument("--split", required=True, choices=["dev", "test", "ood_dev", "ood_test"])
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()
    log_path = Path(args.results_dir) / "TEST_LOG.md"

    if args.method == "m0":
        cfg = dict(get_config(args.config or "configs/m0.yaml"))
        run_dir = Path(args.results_dir) / "runs" / "m0_oracle"
        run_dir.mkdir(parents=True, exist_ok=True)
        method = "m0"
    else:
        if not args.run:
            raise SystemExit("need --run or --method m0")
        run_dir = Path(args.run)
        meta = json.loads((run_dir / "meta.json").read_text())
        cfg, method = meta["config"], meta["method"]
    if args.split in TEST_SPLITS:
        hold = Path(args.results_dir) / "study" / "HOLD_TEST"
        if hold.exists():  # design decisions still open: test stays sealed (remove the file to release)
            raise SystemExit(f"HELD: {hold} exists; test splits are sealed until it is removed")
        test_guard(run_dir, args.split, log_path)

    exs = load_split(cfg["data"], args.split)
    t0 = time.time()
    if method == "m0":
        recs, extra = predict_m0(cfg, exs)
    elif method == "b3_soft":
        recs, extra = predict_b3(run_dir, cfg, exs)
    elif method in POLICY_METHODS:
        recs, extra = predict_policy(run_dir, cfg, exs)
    else:
        recs, extra = predict_b1(run_dir, cfg, exs)
    m = aggregate(recs)
    if method == "b1_sft":  # B1 has no hops: path metrics not applicable
        for k in (*PATH_KEYS, "mean_hops", "n_with_gold_path"):
            m[k] = None
    m.update({"run_id": run_dir.name, "method": method, "split": args.split,
              "eval_time_s": time.time() - t0, "env_lock_sha256": env_lock_sha256(), **extra})
    (run_dir / f"metrics_{args.split}.json").write_text(json.dumps(m, indent=2))
    with open(run_dir / f"predictions_{args.split}.jsonl", "w") as f:
        for r in recs:
            row = {"id": r["id"], "question": r["question"], "pred": r["pred"], "gold": r["gold"],
                   "path": [vars(p) if hasattr(p, "__dict__") else p for p in r["path"]],
                   "gold_path": [vars(p) if hasattr(p, "__dict__") else p for p in (r.get("gold_path") or [])],
                   "trace_info": [{"legal": t["legal"], "redundant": t["redundant"]} for t in r.get("trace_info", [])]}
            if "hop_rewards" in r:  # reward-component traces for F4/F10
                row.update(hop_rewards=r["hop_rewards"], terminal=r["terminal"], comps=r["comps"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.split in TEST_SPLITS:
        log_test(run_dir, args.split, m, log_path)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
