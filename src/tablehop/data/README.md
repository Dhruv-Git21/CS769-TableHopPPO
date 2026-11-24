# tablehop.data

| File | What it does |
|---|---|
| `schema.py` | Dataclasses `Cell`, `Table`, `EntityRef` (a cell or a linked passage) and `Example`, with JSON (de)serialization |
| `hybridqa.py` | Reads the raw HybridQA questions and WikiTables-WithLinks tables/passages from `data/raw/hybridqa`. Never opens the official test file. |
| `gold_paths.py` | Heuristic gold path from HybridQA's `answer-node`: an anchor cell, then the answer cell or the link cell and its passage |
| `splits.py` | k-means (k=20) over table embeddings; 3 held-out clusters become OOD. Writes `data/processed/*.jsonl` and `data/splits/MANIFEST.json` |
| `synthetic.py` | Small deterministic 2-3 hop tables with exact gold paths. Used only for tests and the labelled sanity study. |
| `load.py` | `load_split(cfg, split)` returns `Example` objects for any split, real or synthetic |

The data is built by `python scripts/prepare_data.py` and checked by `python scripts/audit_leakage.py`, which fails if any table appears in two splits.
