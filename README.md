# TableHop-PPO

**Learning multi-hop table reasoning with reinforcement learning: PPO-based entity selection with hop-level rewards.**

Multi-hop table QA (for example HybridQA) needs a chain of lookups: find the right cell, move along its row, follow a hyperlink to a passage, and read the answer. TableHop-PPO treats each hop as a discrete action: choose the next table cell or linked passage, or STOP. A small policy is trained with PPO using a *hop-level* (dense) reward. A shared, frozen answer model then reads the selected evidence and writes the answer.

The project asks: **does hop-level credit assignment beat (a) supervised fine-tuning that reads the whole table and (b) trace-level RL (GRPO) that is rewarded only at the end?** Everything runs on a laptop (Apple M4, 16 GB, CPU + MPS, no GPU). All numbers in the paper, report and slides are produced by the scripts in this repository from `results/`.

> Course project, CS769. Team: Dhruv Goyal, Shaurya Goyal.

---

## Contents
1. [What is in the box](#1-what-is-in-the-box)
2. [Quick start (10 minutes, no data download)](#2-quick-start)
3. [Run an example query (demo)](#3-run-an-example-query-demo)
4. [Full reproduction](#4-full-reproduction)
5. [Method in one page](#5-method-in-one-page)
6. [Baselines, ablations, metrics](#6-baselines-ablations-and-metrics)
7. [Results](#7-results)
8. [Repository layout](#8-repository-layout)
9. [Configuration](#9-configuration)
10. [Extending: new baseline or dataset](#10-extending)
11. [Troubleshooting](#11-troubleshooting)
12. [Limitations](#12-limitations)
13. [Licenses and citation](#13-licenses-and-citation)

---

## 1. What is in the box

| Path | What it is |
|---|---|
| `src/tablehop/` | Python package: data, environment, reward, policy, PPO/GRPO, baselines, metrics |
| `scripts/` | Entry points: `train.py`, `evaluate.py`, `run_study.py`, `summarize.py`, `make_figures.py`, `demo.py`, plus shell drivers |
| `configs/` | One YAML per method, inheriting from `base.yaml` / `rl_base.yaml` |
| `tests/` | 76 fast unit tests (CPU, synthetic data, ~30 s) |
| `results/` | Tables (`tables/`), figures (`figures/`), per-run metrics (`runs/`), tuning state (`study/`), `TEST_LOG.md` |
| `paper/` | ACL-format LaTeX source of the paper |
| `deliverables/` | `paper.pdf`, `WORK_REPORT.docx/.pdf`, `final_presentation.pptx`, `FINAL_SUMMARY.md` |
| `examples/` | A tiny custom table + passages for the demo |
| `docs/` | Proposal, literature notes with verified references, dataset notes, design decisions |

## 2. Quick start

Requirements: macOS or Linux, Python 3.10+ (3.12 used), ~2 GB disk for the environment. A GPU is not needed.

```bash
# 1. environment (venv named CS769-tablehop, pinned versions from requirements-lock.txt)
bash scripts/setup.sh                      # default location: ~/CS769-work/CS769-tablehop
source ~/CS769-work/CS769-tablehop/bin/activate

# 2. sanity check: 76 unit tests + three tiny training runs on synthetic data (~1-2 min on CPU)
make smoke
```

`make help` lists every target. All targets use the env's python automatically; override with `make PY=/path/to/python ...`.

## 3. Run an example query (demo)

The demo shows the model's reasoning step by step for one question:
1. the table;
2. each hop the policy takes, with its probability and the runner-up candidates;
3. the evidence the answer model actually reads;
4. the final answer, and the gold answer when known.

**Web portal (recommended for presentations):**

```bash
make webapp        # then open http://127.0.0.1:8769
```

The portal puts everything on one page:
- **Pick a question:** curated dev questions in three tabs, "Hop-PPO wins", "Both right" and "Failure cases". `scripts/pick_showcase.py` selects them from the dev predictions.
- **Ask your own:** type a question about the same table, or click a suggested follow-up from `examples/demo_questions.json`.
- **Answer:** the answer with its exact-match badge, plus the table with the policy's hops numbered on it.
- **Reasoning path:** each hop with its probability bar and the runner-up candidates.
- **Evidence and comparison:** the exact evidence the answer model read, and a comparison with SFT (B1) and the gold-path oracle (M0).

Deep links work, for example `http://127.0.0.1:8769/?id=hqa:train:029c209ea2dfc169`. It uses only the Python standard library, serves on localhost only, and loads the model once (about 0.5-3 s per question on a laptop). `docs/DEMO_GUIDE.md` has a ready-made 5-minute presentation script with example queries.

**Terminal:**

```bash
make demo          # curated dev questions: Hop-PPO vs SFT (B1) vs the oracle path (M0)
make demo-ask      # interactive: type your own questions about a dev table
bash scripts/demo.sh custom examples/olympics.csv "Who coached the athlete who ran the 100 m in 9.63 ?" examples/olympics_passages.json
```

Example output (dev question, Hop-PPO seed 0):

```
Question: Who was the coach of the gymnast who finished 8th at the 2001 World Artistic
          Gymnastics Championships floor exercise ?
Hop 0: ... | Gymnast | Alexander Beresh ( UKR )          [start]  p=0.786
Hop 1: Alexander Beresh : ... Coached by former Soviet World Champion Igor Korobchinsky ...  [link]  p=1.000
Hop 2: STOP
Evidence read by the answer model:
  [1] 2001 World Artistic Gymnastics Championships | Rank: 8 ; Gymnast: Alexander Beresh ( UKR ) ; Total: 7.862
  [2] Alexander Beresh : Oleksandr Beresch ... Coached by former Soviet World Champion Igor Korobchinsky ...
ANSWER: 'Igor Korobchinsky'   (gold: Igor Korobchinsky  EM 1 F1 1.00)
  b1_sft     'Kevin Harmse'       EM 0
  m0_oracle  'Igor Korobchinsky'  EM 1
```

`--save DIR` also writes `trace.json` and a figure of the path drawn on the table. The demo is display-only: it never writes to `results/TEST_LOG.md` and is never used for model selection.

The model is small, so treat the demo honestly. It often succeeds on HybridQA-style questions. On new tables or phrasings it often picks the wrong row: ordinal words such as "4th" do not match the cell "4" lexically. Section 12 has details.

## 4. Full reproduction

```bash
make data              # download HybridQA + WikiTables-WithLinks (pinned commits, ~880 MB), build splits, leakage audit
make reproduce-all     # embedding cache -> answer head -> tuning -> 3 seeds -> dev/test/OOD eval -> tables -> figures
make paper report slides
```

`make reproduce-all` runs `scripts/reproduce_all.sh`. It is **strictly sequential and resumable**: re-run it after an interruption and finished steps are skipped. Measured wall time on an Apple M4 MacBook Air (16 GB) is about 12-15 hours:

| Step | Script | Time (M4 laptop) |
|---|---|---|
| Splits (3,000 train / 1,000 dev / 2,045 test / 569 OOD test / 500 OOD dev) | `prepare_data.py` | ~5 min |
| Frozen encoder embeddings (bge-small, ~1M texts) | `build_cache.py` | ~1.5 h |
| Shared answer head (flan-t5-small, 3 epochs) | `train_answer_head.py` | ~10 min |
| One RL run (48k trajectories) | `train.py` | ~12-20 min |
| One SFT run (48k examples) | `train.py` | ~75 min |
| Tuning (3 trials x 5 methods) + final runs (9 methods x 3 seeds) + evaluation | `run_study.py` | ~10 h |

Individual commands:

```bash
python scripts/train.py    --config configs/h1_hoppo.yaml --seed 0 [--override ppo.lr_head=1e-3 ...]
python scripts/evaluate.py --run results/runs/<run_id> --split dev        # dev | test | ood_dev | ood_test
python scripts/evaluate.py --method m0 --split dev                         # oracle path upper bound
python scripts/summarize.py      # console comparison table + results/tables/*.csv|tex
python scripts/make_figures.py   # results/figures/*.png|pdf
```

**Test-set discipline.** Each final run is evaluated on `test` and `ood_test` exactly once. `evaluate.py` refuses a second evaluation and logs every test evaluation in `results/TEST_LOG.md`. All tuning and design decisions used dev only. The official HybridQA test file is never opened: our test split is drawn from official dev questions on tables unused for training (see `results/data_report.md`).

**One environment for every run.** Every run records the sha256 of `requirements-lock.txt` in its `meta.json`. `summarize.py` refuses to compare runs made with different environments.

## 5. Method in one page

**Problem as a POMDP.** State: the question plus the entities visited so far. Action: one candidate entity, or STOP. Transition: move to that entity and expose its neighbours. Episodes last at most 4 hops.

**Candidates** (`src/tablehop/env/candidates.py`):
- Hop 0: the top-32 cells by 0.5 · BM25 + 0.5 · cosine(question, cell).
- Later hops: cells in the same row, the top-8 cells in the same column, the cell's linked Wikipedia passage, and STOP. STOP is illegal at hop 0.

**Policy** (`src/tablehop/models/policy.py`, ~0.3M parameters, CPU). A GRU over the embeddings of visited entities. Each candidate is scored by:
- a bilinear match with the policy state;
- a relation embedding (same row / same column / link / stop);
- a learned cosine term;
- four lexical features: cosine, BM25, question-token overlap and header overlap.

The text encoder `BAAI/bge-small-en-v1.5` is frozen.

**Hybrid reward** (`src/tablehop/rl/reward.py`):

| term | value | when |
|---|---|---|
| hop | +0.2 | the hop lands on the next gold-path entity (order-consistent; revisits never pay) |
| redundancy | -0.1 | revisiting an entity |
| step | -0.02 | every hop |
| schema | -0.05 | at the end, if the last cell's type does not match the question cue (number/date) |
| answer | +1.0 · F1 | at the end, F1 of the answer model's output |

**Training.** PPO with GAE (λ=0.95, γ=1), clip 0.2, 4 epochs, entropy 0.01 decayed to 0, and a budget of 48,000 trajectories per run.

**Answer head** (`src/tablehop/models/answer_head.py`). `google/flan-t5-small` is fine-tuned once on gold evidence, then frozen and shared by every path-based method. Each selected cell is shown to it as its whole row, and each selected passage as text.

## 6. Baselines, ablations and metrics

| ID | Method | What it controls for |
|---|---|---|
| M0 | Oracle: answer head on the gold path | upper bound of every path method |
| B1 | SFT: flan-t5-small reads the BM25-ranked table + passages | no RL, full context |
| B2a | GRPO, answer F1 only (trace-level reward) | credit assignment at trace level |
| B2b | GRPO + 0.2 · path F1 at the end | same supervision as H1, but not per hop |
| B3 | Soft region attention over candidates, answer-only signal | no sequential decisions |
| **H1** | **Hop-PPO with the hybrid reward** | the proposed method |
| A1 | H1 with terminal reward only | does the dense reward matter? |
| A2 / A3 | H1 without redundancy / schema terms | reward components |
| A4 | H1 with unrestricted (whole-table) actions | structured action space |

The comparison is kept fair in four ways:
- The same budget unit (48k trajectories or examples) and the same tuning budget (3 trials per method on dev) for every method.
- The same encoder and frozen answer head for every path method.
- 3 seeds per method.
- 95% bootstrap CIs and paired bootstrap tests against H1.

Metrics: exact match (EM), token F1, strict and loose path validity, gold-path F1, hop efficiency, mean hops, and the ID-to-OOD gap. OOD means held-out table clusters, found by k-means over table embeddings.

## 7. Results

<!-- RESULTS:BEGIN (generated from results/tables/main.csv; do not edit by hand) -->
See `results/tables/main.csv` and `results/figures/`. This section is filled automatically after the final runs.
<!-- RESULTS:END -->

## 8. Repository layout

Every directory has its own `README.md` explaining its files: `src/tablehop/*/`, `scripts/`, `configs/`, `tests/`, `results/`, `docs/`, `paper/`, `data/`, `examples/`, `deliverables/`.

```
src/tablehop/
  data/       schema.py (Cell/Table/Example), hybridqa.py (loader), gold_paths.py, splits.py (ID/OOD), synthetic.py
  env/        render.py (entity -> text), candidates.py (action sets + features), table_env.py (POMDP)
  models/     encoder.py (frozen bge-small + cache), policy.py (GRU policy + value), answer_head.py (frozen flan-t5)
  rl/         reward.py, rollout.py, ppo.py, grpo.py, runner.py (training loop shared by H1/A*/B2*)
  baselines/  sft.py (B1), soft_region.py (B3)
  eval/       metrics.py, stats.py (bootstrap), trace_viz.py (demo figure)
  utils/      config.py (YAML inheritance + --override), runmeta.py (meta.json, env hash), seeding, logging
scripts/      setup.sh, download_data.sh, smoke.sh, reproduce_all.sh, demo.sh, export_release.py,
              prepare_data.py, audit_leakage.py, build_cache.py, train_answer_head.py, train.py, evaluate.py,
              run_study.py, summarize.py, make_figures.py, make_case_studies.py, make_report.py, slides/, dev/
configs/      base.yaml, rl_base.yaml, one file per method, smoke_*.yaml (synthetic)
tests/        unit tests for env, reward, PPO/GRPO, metrics, data splits, answer head, B3, demo
results/      tables/, figures/, runs/<run_id>/{meta.json, metrics_<split>.json, train_log.csv, predictions}, study/, TEST_LOG.md
```

## 9. Configuration

Configs inherit with `base: <file>`. Any key can be overridden from the command line:

```bash
python scripts/train.py --config configs/h1_hoppo.yaml --seed 1 --override reward.w_hop=0.1 env.max_hops=3
```

| Key | Default | Meaning |
|---|---|---|
| `env.K`, `env.col_topk`, `env.max_hops` | 32, 8, 4 | candidate set sizes, episode length |
| `reward.w_hop/w_red/w_step/w_schema/w_ans` | 0.2/0.1/0.02/0.05/1.0 | hybrid reward weights |
| `reward.terminal_only`, `no_redundancy`, `no_schema` | false | ablation switches A1-A3 |
| `env.unrestricted` | false | ablation A4 |
| `ppo.lr_head`, `ppo.entropy_coef` | 3e-4, 0.01 | tuned on dev (3 trials) |
| `rl.budget_trajectories` | 48000 | matched training budget |
| `answer_head.path` | results/answer_head_row | shared frozen answer model |

## 10. Extending

- **New baseline:** add `src/tablehop/baselines/<name>.py` with `train(cfg, seed) -> run_dir`. Register it in the `scripts/train.py` registry and add a predictor in `scripts/evaluate.py`. Add `configs/<name>.yaml` with `method: <name>`, and a trial grid in `scripts/run_study.py`, so it gets the same tuning budget as every other method.
- **New dataset:** convert it to `Example`/`Table` (`src/tablehop/data/schema.py`) and write `data/processed/{split}.jsonl`. The environment, reward and metrics need nothing else. Gold paths are optional; they are only needed for hop rewards and path metrics.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| Laptop very slow, heavy swapping | Run one heavy job at a time (`run_study.py` is sequential by default). Keep data and runs outside iCloud-synced folders. |
| `build_cache.py` exits with code 3 | Normal: it checkpoints every `--budget` seconds; the loop in `reproduce_all.sh` resumes it. |
| `REFUSED: ... already evaluated on test` | By design: test is evaluated once per run. |
| `env_lock_sha256 differs` from `summarize.py` | Runs come from different environments. Rebuild with `scripts/setup.sh` and re-run. |
| Demo says "no finished h1_hoppo run" | Train one (`scripts/train.py --config configs/h1_hoppo.yaml --seed 0`) or pass `RUN=<run dir>`. |
| MPS errors | Set `device: cpu` in the config, or `--override device=cpu` (slower). |

## 12. Limitations

- **Small models.** A frozen 33M-parameter encoder, a 0.3M-parameter policy and a 77M-parameter answer model, all trained on 3,000 questions. Absolute scores are far below published HybridQA systems and are not comparable to them.
- **Noisy gold paths.** HybridQA gives no gold hop paths, so they are reconstructed heuristically: 98% of questions get one, and none are verified by hand. The hop reward is therefore only as good as this heuristic.
- **One benchmark** (HybridQA), with OOD measured on held-out table clusters inside it.
- **Lexical first hop.** The first hop depends heavily on lexical overlap with the question. On new phrasings or tables the policy often picks the wrong row.

The paper and work report discuss which hypotheses were and were not supported.

## 13. Licenses and citation

| Item | License |
|---|---|
| HybridQA | MIT; tables and passages are crawled from Wikipedia |
| `BAAI/bge-small-en-v1.5` | MIT |
| `google/flan-t5-small` | Apache-2.0 |

The repository license for this code is chosen by the team: `[TEAM TO CONFIRM: license for this code, e.g. MIT]`.

```bibtex
@misc{tablehop2026,
  title  = {Learning Multi-Hop Table Reasoning through RL: PPO-Based Entity Selection for Decision Making},
  author = {Goyal, Dhruv and Goyal, Shaurya},
  year   = {2026},
  note   = {CS769 course project. [TEAM TO CONFIRM: full author names, venue]}
}
```
