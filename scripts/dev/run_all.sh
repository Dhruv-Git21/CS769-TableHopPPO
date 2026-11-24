#!/usr/bin/env bash
# Full pipeline, strictly sequential (one heavy job at a time) and resumable: a (config, seed) pair with a
# finished run (meta status=done) is skipped; an evaluation whose metrics file exists is skipped.
# usage: bash scripts/run_all.sh [SEEDS="0 1 2"] ; env: PY, CONFIGS (override method list), SKIP_TEST=1
set -u
PY=${PY:-$HOME/CS769-work/CS769-tablehop/bin/python}
SEEDS=${SEEDS:-"0 1 2"}
CONFIGS=${CONFIGS:-"b1_sft b2a_grpo b2b_grpo_path b3_soft h1_hoppo a1_terminal a2_noredund a3_noschema a4_unrestricted"}
cd "$(dirname "$0")/.."
RUNS=results/runs

step() { echo "[run_all] $* $(date)"; }

# find_run CONFIG SEED -> prints the run dir of a finished run for this config file + seed, if any
find_run() {
  $PY - "$1" "$2" <<'PYEOF'
import json, sys, glob
cfg, seed = sys.argv[1], int(sys.argv[2])
from tablehop.utils.config import get_config
want = dict(get_config(f"configs/{cfg}.yaml"))
for mp in sorted(glob.glob("results/runs/*/meta.json")):
    m = json.load(open(mp))
    c = m.get("config", {})
    if m.get("status") == "done" and m.get("seed") == seed and m.get("method") == want.get("method") \
            and c.get("exp") == want.get("exp") and c.get("reward") == want.get("reward"):
        print(mp.rsplit("/", 1)[0]); break
PYEOF
}

step "data"
if [ ! -f data/processed/train.jsonl ]; then $PY scripts/prepare_data.py || exit 1; fi
$PY scripts/audit_leakage.py || { echo "leakage audit failed"; exit 1; }
step "embedding cache"
until $PY scripts/build_cache.py --device mps --budget 100000; do rc=$?; [ $rc -eq 3 ] || { echo "cache failed"; exit 1; }; done
step "answer head"
[ -f results/answer_head/sha256.txt ] || $PY scripts/train_answer_head.py || exit 1
[ -f $RUNS/m0_oracle/metrics_dev.json ] || $PY scripts/evaluate.py --method m0 --config configs/m0.yaml --split dev
step "HP search (optional hook)"
[ -f scripts/hp_search.sh ] && bash scripts/hp_search.sh

for cfg in $CONFIGS; do
  if [ ! -f configs/$cfg.yaml ]; then echo "[run_all] WARNING: configs/$cfg.yaml missing, skipped"; continue; fi
  for seed in $SEEDS; do
    run=$(find_run $cfg $seed)
    if [ -z "$run" ]; then
      step "train $cfg seed $seed"
      $PY scripts/train.py --config configs/$cfg.yaml --seed $seed || { echo "[run_all] $cfg s$seed FAILED"; continue; }
      run=$(find_run $cfg $seed)
    fi
    [ -z "$run" ] && continue
    splits="dev ood_dev"; [ "${SKIP_TEST:-0}" = 1 ] || splits="$splits test ood_test"
    for sp in $splits; do
      [ -f $run/metrics_$sp.json ] && continue
      step "eval $run $sp"
      $PY scripts/evaluate.py --run "$run" --split $sp || echo "[run_all] eval $sp failed"
    done
  done
done
step "summary + figures"
$PY scripts/summarize.py
$PY scripts/make_figures.py
echo "[run_all] done $(date)"
