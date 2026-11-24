#!/usr/bin/env bash
# 2-minute check that everything is wired: unit tests (synthetic data, CPU) + tiny training runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/CS769-work/CS769-tablehop/bin/python}
$PY -m pytest -q tests
RL="paths.results_dir=results/smoke rl.budget_trajectories=1280 rl.eval_every=5 rl.eval_subset=50"
$PY scripts/train.py --config configs/smoke_h1.yaml --seed 0 --override $RL
$PY scripts/train.py --config configs/smoke_b2a.yaml --seed 0 --override $RL
$PY scripts/train.py --config configs/smoke_b3.yaml --seed 0 --override paths.results_dir=results/smoke
echo "[smoke] ok"
