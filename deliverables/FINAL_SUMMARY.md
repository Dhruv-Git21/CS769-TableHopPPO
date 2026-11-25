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

## Status of results
Final: all 9 trained methods x 3 seeds, evaluated once on test and OOD test (results/TEST_LOG.md), 2026-11-26.

## Hypotheses (final, test split, 3 seeds)
| ID | Claim | Status |
|---|---|---|
| H1 | Hop-level credit beats trace-level RL | not supported (vs B2a: -2.0 [-2.8, -1.2], seeds disagree) |
| H1b | Gains are not just extra supervision (vs B2b) | supported (+2.6 [+1.6, +3.6], all seeds) |
| H1c | The dense reward matters (vs A1) | not supported |
| H2 | Paths are valid and auditable | partially supported (highest validity among main methods, 14.0%) |
| H3 | Smaller ID-to-OOD gap | inconclusive (5.8% vs 14.6% relative for SFT; not tested) |

Headline (test EM): oracle 57.2, SFT 27.0, answer-only GRPO 11.5, soft selection 11.3, Hop-PPO 9.5, answer+path GRPO 6.9.

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
