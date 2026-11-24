# Claims ledger (owned by Opus; writer may only read)

Every improvement claim in the paper/report/slides must have a row here with status and evidence. Opus fills `status` after Gate G5; `writer` uses only the wording allowed by status.

| ID | Claim | Experiment | Metric | Evidence (run ids / table) | Seeds | Effect (95% CI) | Status | Allowed wording |
|---|---|---|---|---|---|---|---|---|
| H1 | Hop-level credit reduces error propagation vs trace-level rewards | E1, E4 | EM vs B2a/B2b, per-hop validity | results/tables/paired.csv (h1 - b2a, h1 - b2b) | 1 (prelim) | vs B2a +1.3 [-0.9, +3.5] dev | inconclusive | "not reliably different from answer-only GRPO" |
| H1b | Gains are not just extra supervision (vs B2b) | E1 | EM, path validity | paired.csv (h1 - b2b) | 1 (prelim) | +4.0 [+2.0, +6.1] dev | supported in direction (needs 3 seeds) | "beats trace-level RL given the same path signal at the end" |
| H1c | Dense reward matters, not only discrete actions (vs A1) | E2 | EM, validity, hop efficiency | ablations.csv, paired.csv (h1 - a1) | 1 (prelim) | -8.9 [-11.3, -6.5] dev | not supported | "the terminal-only variant is more accurate" |
| H2 | Paths are auditable and valid | E1, E5 | strict validity | main.csv | 1 (prelim) | H1 17.2% vs 7-9% for other path methods | partially supported | "most valid paths among learned methods, but low in absolute terms" |
| H3 | Smaller ID->OOD gap than SFT and trace-level RL | E3 | gap | ood.csv | 0 | pending | inconclusive (not yet evaluated) | "not assessed in this version" |
| R1 | Baselines reproduced within ~2-3 points of published | E1 | EM/F1 | STATUS.md R1 | n/a | n/a | not applicable (context only) | "published systems use far larger models; not comparable" |

Status values: supported (CI excludes 0 and >=3 seeds agree in sign) / partially supported / not supported (CI includes 0 or sign reverses) / inconclusive (too few seeds or budget).
Allowed wording examples: supported -> "improves ... (+x EM, 95% CI [a,b])"; not supported -> "we did not observe an improvement"; inconclusive -> "results are suggestive but underpowered".

Statuses above are PRELIMINARY (dev split, seed 0, 2026-11-24). Re-derive from results/tables after the final 3-seed test runs; the paper's Table (tab:claims) must match.
