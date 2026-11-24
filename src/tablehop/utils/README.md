# tablehop.utils

| File | What it does |
|---|---|
| `config.py` | YAML loading with `base:` inheritance and `key.sub=value` overrides |
| `runmeta.py` | Creates `results/runs/<run_id>/meta.json` with config, seed, git hash, device, versions and `env_lock_sha256` |
| `seeding.py` | Seeds python, numpy and torch |
| `device.py` | `auto` resolves to mps, then cuda, then cpu |
| `logging.py` | Console logger and `CSVLogger` for `train_log.csv` |
