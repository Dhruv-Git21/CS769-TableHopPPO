#!/usr/bin/env bash
# Query -> answer + evidence with the trained Hop-PPO model (display only; never used for tuning or test scores).
#
#   bash scripts/demo.sh                    # showcase: curated dev questions, H1 vs B1 (SFT) vs oracle path
#   bash scripts/demo.sh ask [EXAMPLE_ID]   # interactive: type your own questions about one dev table
#   bash scripts/demo.sh custom TABLE.csv "your question" [PASSAGES.json]   # your own table
#
# env: PY (python of the CS769-tablehop env), RUN (H1 run dir; default = newest finished h1_hoppo run),
#      SAVE (output dir for trace.json + figure; default results/demo)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/CS769-work/CS769-tablehop/bin/python}
SAVE=${SAVE:-results/demo}
RUNARG=()
[ -n "${RUN:-}" ] && RUNARG=(--run "$RUN")
B1=$(ls -td results/runs/b1_sft_hybridqa_s0_* 2>/dev/null | head -1 || true)
CMP=()
[ -n "$B1" ] && CMP=(--compare "$B1")
# curated dev questions (multi-hop: table cell -> linked passage); chosen from dev predictions, never from test
SHOWCASE=(hqa:train:029c209ea2dfc169 hqa:train:0742e9b018c6bdc5 hqa:train:01a3c53762efb8ef)

quiet() { grep -v -E "Redirects are currently|tied weights|Loading weights" || true; }

case "${1:-showcase}" in
  showcase)
    for id in "${SHOWCASE[@]}"; do
      $PY scripts/demo.py "${RUNARG[@]}" --split dev --id "$id" "${CMP[@]}" --m0 --save "$SAVE" 2>&1 | quiet
      echo; echo "----------------------------------------------------------------------"
    done ;;
  ask)
    $PY scripts/demo.py "${RUNARG[@]}" --split dev --ask "${2:-${SHOWCASE[0]}}" 2>&1 | quiet ;;
  custom)
    [ $# -ge 3 ] || { echo "usage: bash scripts/demo.sh custom TABLE.csv \"question\" [PASSAGES.json]"; exit 1; }
    P=(); [ $# -ge 4 ] && P=(--passages "$4")
    $PY scripts/demo.py "${RUNARG[@]}" --table "$2" --question "$3" "${P[@]}" --save "$SAVE" 2>&1 | quiet ;;
  *) sed -n 2,10p "$0"; exit 1 ;;
esac
