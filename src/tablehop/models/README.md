# tablehop.models

| File | What it does |
|---|---|
| `encoder.py` | `EmbeddingCache`: frozen `BAAI/bge-small-en-v1.5` embeddings (L2-normalized). Cached to `data/cache/` by `scripts/build_cache.py`, so nothing is recomputed. |
| `policy.py` | `Policy`: a GRU over the visited entities. Scores each candidate as bilinear(state, candidate) + a relation term + beta*cosine + a feature term, then takes a masked softmax. Has a value head for PPO. About 0.3M parameters; runs on CPU. |
| `answer_head.py` | `AnswerHead`: `google/flan-t5-small`, fine-tuned once by `scripts/train_answer_head.py`, then frozen and shared by every path method. Its sha256 is recorded in every run. It reads `evidence_mode.txt` (row/entity) from its folder. |
