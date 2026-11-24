# results/

Everything here is written by scripts. Numbers in the paper, report and slides come only from these files.

| Path | Written by | Contents |
|---|---|---|
| `runs/<run_id>/` | `train.py`, `evaluate.py` | `meta.json` (config, seed, git hash, env hash), `train_log.csv`, `metrics_<split>.json`, `predictions_<split>.jsonl[.gz]`, checkpoint (only the H1 checkpoints are in the release) |
| `runs/m0_oracle/` | `evaluate.py --method m0` | Oracle upper bound |
| `study/state.json` | `run_study.py` | Tuning trials, the chosen setting per method, and the run dir of each final (method, seed) |
| `tables/` | `summarize.py` | `main`, `ablations`, `ood`, `paired` as `.csv` and `.tex`; `macros.tex` for the paper |
| `figures/` | `make_figures.py` | F2 main, F3 learning curves, F4 per-hop validity, F5 ablations, F7 hop counts, F9 error taxonomy, F10 reward components |
| `TEST_LOG.md` | `evaluate.py` | One line per test/OOD-test evaluation (each run is evaluated once) |
| `data_report.md` | `prepare_data.py` | Split sizes, gold-path coverage, table statistics |
| `answer_head*/` | `train_answer_head.py` | Dev metrics and sha256 of the shared answer head (weights are not committed) |
| `demo/` | `demo.py`, `pick_showcase.py` | Saved traces, showcase list, portal screenshot |
| `sanity/` | | Labelled synthetic sanity run (E6); not a benchmark result |
