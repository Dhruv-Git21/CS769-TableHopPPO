# scripts/

Run every script from the repository root with the project environment's python (`PY=~/CS769-work/CS769-tablehop/bin/python`, or activate the venv). The Makefile wraps the common ones (`make help`).

## Shell drivers (start here)
| Script | Purpose |
|---|---|
| `setup.sh` | Create the `CS769-tablehop` venv from `requirements-lock.txt` |
| `download_data.sh` | Fetch HybridQA + WikiTables-WithLinks at pinned commits |
| `smoke.sh` | 76 unit tests + tiny training runs (~1-2 min) |
| `reproduce_all.sh` | The full study, sequential and resumable (~12-15 h on a laptop) |
| `demo.sh` | Terminal demo: `showcase`, `ask [ID]`, `custom TABLE.csv "question"` |

## Pipeline steps (in order)
| Script | Purpose |
|---|---|
| `prepare_data.py` | Build `data/processed/*.jsonl` + `data/splits/MANIFEST.json` + `results/data_report.md` |
| `audit_leakage.py` | Fail if any table appears in two splits, or if official test is used |
| `build_cache.py` | Frozen encoder embeddings for every split; checkpoints and exits with code 3 every `--budget` seconds (rerun to resume) |
| `train_answer_head.py` | Train the shared answer head once (`--config configs/answer_head_row.yaml`) |
| `train.py` | Train one method: `--config configs/<method>.yaml --seed S [--override k=v ...]` |
| `evaluate.py` | `--run <dir> --split dev\|test\|ood_dev\|ood_test`, or `--method m0`. Test splits only once per run (logged in `results/TEST_LOG.md`). |
| `run_study.py` | Tuning (3 trials per method, dev) -> final runs x 3 seeds -> evaluation. State is kept in `results/study/state.json`. |
| `summarize.py` | Console comparison + `results/tables/*.csv\|tex` |
| `make_figures.py` | `results/figures/F*.png\|pdf` |

## Demo and deliverables
| Script | Purpose |
|---|---|
| `demo.py` | Hop-by-hop trace of one question (modes: `--id`, `--ask`, `--table`) |
| `webapp.py` + `webapp.html` | Web portal at http://127.0.0.1:8769 (`make webapp`) |
| `pick_showcase.py` | Choose demo questions from dev predictions |
| `make_case_studies.py` | Path figures for case studies |
| `make_report.py` | `deliverables/WORK_REPORT.docx/.pdf` from `docs/report/WORK_REPORT.md` |
| `slides/build_final_deck.py` | `deliverables/final_presentation.pptx` in the team template |
| `export_release.py` | Clean GitHub/submission copy (`make release`) |
| `probe_env.sh`, `verify_setup.sh` | Environment checks |
| `dev/` | One-off drivers kept for the record (not needed to reproduce) |
