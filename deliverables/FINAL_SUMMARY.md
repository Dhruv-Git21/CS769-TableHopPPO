# TableHop-PPO: final summary

Team: Dhruv Goyal, Shaurya Goyal (CS769 course project). Build date: 2026-11-24.

## Deliverables (this folder)
| File | What it is | Rebuild with |
|---|---|---|
| `paper.pdf` | ACL-format paper, camera-ready style (authors shown) | `make paper` |
| `paper_anonymous.pdf` | Same paper, anonymous review build with line numbers | `make paper` |
| `final_presentation.pptx` | 21 slides in the team template (15 presented + references, thank-you, 4 backup); every result number read from `results/tables` | `make slides` |
| `TALK_SCRIPT.pdf` / `.docx` / `.md` | What to say on each slide, presenter, timing (~14.8 min) | `make slides` (+ pandoc for docx/pdf) |
| `WORK_REPORT.pdf` / `.docx` | Course work report (16 pages) | `make report` |

Working code: the clean, GitHub-ready directory `../tablehop-ppo-submission/` (README in every folder; `make help`).

## What ran
- Data: HybridQA. Splits: 3,000 train, 1,000 dev, 2,045 test (official-dev questions on unseen tables), 500/569 OOD dev/test (3 held-out table clusters). The official test set was never opened.
- Methods: M0 oracle path; B1 full-table SFT; B2a answer-only GRPO; B2b GRPO with an end-of-trace path bonus; B3 soft selection; H1 Hop-PPO (ours); ablations A1-A4.
- Protocol: 48k training episodes or examples per run, 3 tuning trials per method on dev, 3 seeds, test evaluated once per run, 95% bootstrap CIs and paired tests.
- Compute: one Apple M4 laptop (16 GB, CPU + MPS, no GPU); frozen bge-small encoder and flan-t5-small answer head.

## Status of results in this build
The tables, figures, paper, report and slides were built from the results available on 2026-11-24: the **development split with seed 0** for each method. The 3-seed final runs and the test/OOD evaluation are running unattended (`scripts/run_study.py`). When they finish, rebuild with `make tables figures paper slides report release`. The prose uses generated macros, so numbers and the split name update automatically; then re-check the claim directions (see `STATUS.md` checklist).

## Hypotheses (preliminary, dev, seed 0)
| ID | Claim | Status |
|---|---|---|
| H1 | Hop-level credit beats trace-level RL | inconclusive (vs B2a, CI includes 0) |
| H1b | Gains are not just extra supervision (vs B2b) | supported in direction |
| H1c | The dense reward matters (vs A1) | not supported (A1 is more accurate) |
| H2 | Paths are valid and auditable | partially supported |
| H3 | Smaller ID-to-OOD gap | not yet assessed |

## What was cut
MultiHiertt, TAT-QA and FinQA; a dual-agent baseline; a text-generation GRPO baseline with a large model; potential-based shaping; 5 seeds; GPU-scale models. Tuning used 3 trials per method instead of 6 (design change R4).

## Design changes (all decided on dev, before any test evaluation)
R2 candidate features; R3 row evidence for the answer head; R4 tuning budget; a fix for a revisit reward hack. See `docs/DESIGN_NOTES.md`.

## Open items for the team
- Paper: affiliation and email (`[TEAM TO CONFIRM]`). Code link added (final build only; hidden in the anonymous build).
- Slides: department and institute on the title slide.
- Work report: contributions recorded as an equal split (section 13).
- Repository license (README section 13).
- Demo: after the final runs, run `python scripts/pick_showcase.py --run <final H1 run>` and re-check the example queries in `docs/DEMO_GUIDE.md`.
