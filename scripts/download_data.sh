#!/usr/bin/env bash
# Download HybridQA (questions) and WikiTables-WithLinks (tables + passages) from the official GitHub repos,
# pinned to the commits used in this project (docs/lit/DATASETS.md). ~880 MB. The official test file is never read.
# usage: bash scripts/download_data.sh [RAW_DIR]   (default data/raw/hybridqa)
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=${1:-data/raw/hybridqa}
mkdir -p "$RAW"
fetch() {  # repo dir commit
  if [ -d "$RAW/$2/.git" ]; then echo "[data] $2 present"; return; fi
  git clone --filter=blob:none "https://github.com/wenhuchen/$1.git" "$RAW/$2"
  git -C "$RAW/$2" checkout -q "$3"
}
fetch HybridQA HybridQA db22fda8c5951438fade3c69d75b350335ba93b3
fetch WikiTables-WithLinks WikiTables-WithLinks dc066e1a6d5281511d8b73a6107d5ad2824cc2b2
echo "[data] done: $RAW (licenses: HybridQA MIT; tables/passages crawled from Wikipedia)"
