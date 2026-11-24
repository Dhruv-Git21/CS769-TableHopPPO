"""Dispatcher: python scripts/train.py --config configs/<m>.yaml --seed S [--override k=v ...].

Methods register in REGISTRY as "method": "module:function"; function(cfg: dict, seed: int) -> run_dir.
"""
from __future__ import annotations

import argparse
import importlib

from tablehop.utils.config import get_config

REGISTRY: dict[str, str] = {
    "b1_sft": "tablehop.baselines.sft:train",
    "b3_soft": "tablehop.baselines.soft_region:train",
    **{m: "tablehop.rl.ppo:train" for m in ("h1_hoppo", "a1_terminal", "a2_noredund", "a3_noschema",
                                            "a4_unrestricted")},
    **{m: "tablehop.rl.grpo:train" for m in ("b2a_grpo", "b2b_grpo_path")},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="run dir to resume (PPO/GRPO methods)")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = dict(get_config(args.config, args.override))
    if args.resume:
        cfg["resume"] = args.resume
    method = cfg.get("method")
    if method not in REGISTRY:
        raise SystemExit(f"unknown method {method!r}; registered: {sorted(REGISTRY)}")
    mod, fn = REGISTRY[method].split(":")
    run_dir = getattr(importlib.import_module(mod), fn)(cfg, args.seed)
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
