# tests/

Fast unit tests (CPU, synthetic data or tiny fixtures; about 30 s in total): `make test` or `python -m pytest -q tests`.

| File | Covers |
|---|---|
| `test_env.py` | Masks, STOP rules, forced stop, gold path reachable, order-consistent and no-revisit hop reward, row evidence |
| `test_reward.py` | Every reward term and ablation switch |
| `test_ppo.py`, `test_grpo.py` | GAE, clipped loss, group advantages, learning on a toy task |
| `test_metrics.py` | EM/F1/path metrics on hand-computed cases |
| `test_data.py` | Schema round-trip, gold-path heuristic, split disjointness |
| `test_answer_head.py`, `test_b3.py`, `test_demo.py`, `test_summarize.py`, `test_utils.py` | Interfaces and scripts |
| `fixtures/` | A few HybridQA-format examples |
