# data/

Raw and processed data are not committed. Build them with:

```bash
bash scripts/download_data.sh      # data/raw/hybridqa (official GitHub repos, pinned commits)
python scripts/prepare_data.py     # data/processed/{train,dev,test,ood_dev,ood_test}.jsonl
python scripts/audit_leakage.py    # must pass
python scripts/build_cache.py      # data/cache/ frozen-encoder embeddings (~1.5 h)
```

`splits/MANIFEST.json` records counts, sha256 of each split file, the held-out cluster ids and the seed. Splits are deterministic (seed 0).
