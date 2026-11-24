#!/usr/bin/env bash
# Create the project environment (venv "CS769-tablehop", Python 3.12) from the pinned lock file.
# usage: bash scripts/setup.sh [ENV_DIR]     (default: $HOME/CS769-work/CS769-tablehop)
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${1:-${ENV:-$HOME/CS769-work/CS769-tablehop}}
PYBIN=${PYBIN:-$(command -v python3.12 || command -v python3)}
"$PYBIN" -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10"'
echo "[setup] creating $ENV with $PYBIN"
"$PYBIN" -m venv --prompt CS769-tablehop "$ENV"
"$ENV/bin/pip" install -U pip
"$ENV/bin/pip" install -r requirements-lock.txt
"$ENV/bin/pip" install -e .
"$ENV/bin/python" -c "import torch, transformers; print('[setup] torch', torch.__version__, '| transformers', transformers.__version__, '| mps', torch.backends.mps.is_available(), '| cuda', torch.cuda.is_available())"
echo "[setup] done. Activate with: source $ENV/bin/activate   (or pass PY=$ENV/bin/python to make)"
