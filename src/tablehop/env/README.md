# tablehop.env

| File | What it does |
|---|---|
| `render.py` | Turns entities into text for the encoder and the answer head. `render_evidence(mode="row")` shows a selected cell as its whole row (spec change R3). |
| `candidates.py` | The action set. At hop 0: the top-32 cells by 0.5*BM25 + 0.5*cosine. Later hops: same row, the top-8 cells of the same column, the linked passage, and STOP. Also computes per-candidate features [cosine, BM25, question overlap, header overlap] (R2). Relation ids: start 0, same_row 1, same_col 2, link 3, stop 4. |
| `table_env.py` | `TableEnv.reset()/step(a)`. At most 4 hops; STOP is illegal at hop 0. Returns `State` (embeddings, mask, features) and `HopInfo` (on_gold, redundant, legal). A revisit never earns the hop reward again. |

Unit tests: `tests/test_env.py`.
