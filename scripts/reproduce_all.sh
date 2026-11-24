#!/usr/bin/env bash
# Full study, strictly sequential and resumable (re-run after an interruption; finished steps are skipped):
#   data -> leakage audit -> embedding cache -> frozen answer head -> tuning (3 trials/method, dev)
#   -> final runs x seeds {0,1,2} -> dev/test/OOD evaluation (test once per run) -> tables -> figures.
# Measured on an Apple M4 laptop (16 GB, CPU+MPS): about 12-15 hours end to end.
# usage: bash scripts/reproduce_all.sh      env: PY=<python of the CS769-tablehop env>
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/CS769-work/CS769-tablehop/bin/python}
step() { echo; echo "==== [reproduce] $* ($(date '+%F %T'))"; }

step "1/7 data"
[ -d data/raw/hybridqa/HybridQA ] || bash scripts/download_data.sh
[ -f data/processed/train.jsonl ] || $PY scripts/prepare_data.py
$PY scripts/audit_leakage.py

step "2/7 frozen encoder embeddings (resumable in chunks)"
until $PY scripts/build_cache.py --budget 900; do rc=$?; [ $rc -eq 3 ] || { echo "cache failed ($rc)"; exit 1; }; done

step "3/7 shared frozen answer head (row evidence, spec change R3)"
[ -f results/answer_head_row/sha256.txt ] || $PY scripts/train_answer_head.py --config configs/answer_head_row.yaml

step "4-6/7 tuning -> final runs x 3 seeds -> evaluation (scripts/run_study.py, state in results/study/state.json)"
$PY scripts/run_study.py --phase all

step "7/7 tables + figures"
$PY scripts/summarize.py
$PY scripts/make_figures.py
echo "[reproduce] done. Tables: results/tables/  Figures: results/figures/  Test log: results/TEST_LOG.md"
