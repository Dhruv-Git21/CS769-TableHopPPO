# Claims ledger (owned by Opus; writer may only read)

Every improvement claim in the paper/report/slides must have a row here with status and evidence. Opus fills `status` after Gate G5; `writer` uses only the wording allowed by status.

| ID | Claim | Experiment | Metric | Evidence (run ids / table) | Seeds | Effect (95% CI) | Status | Allowed wording |
|---|---|---|---|---|---|---|---|---|
| H1 | Hop-level credit reduces error propagation vs trace-level rewards | E1, E4 | EM vs B2a | paired.csv test (h1 - b2a) | 3 | -2.0 [-2.8, -1.2]; per seed -1.2, -5.8, +1.0 | not supported | "did not beat answer-only trace-level RL" |
| H1b | Gains are not just extra supervision (vs B2b) | E1 | EM, path validity | paired.csv test (h1 - b2b) | 3 | +2.6 [+1.6, +3.6]; per seed +2.2, +1.7, +3.9 | supported | "beats trace-level RL given the same path signal at the end" |
| H1c | Dense reward matters, not only discrete actions (vs A1) | E2 | EM | paired.csv test (h1 - a1) | 3 | -2.3 [-3.2, -1.5]; per seed -9.7, -0.1, +2.8 | not supported | "the dense reward did not improve accuracy" |
| H2 | Paths are auditable and valid | E1, E5 | strict validity | main.csv test | 3 | H1 14.0% vs 6.4-10.5% for main baselines | partially supported | "most valid paths among main methods, low in absolute terms" |
| H3 | Smaller ID->OOD gap than SFT and trace-level RL | E3 | gap | ood.csv | 3 | H1 0.6 pts (5.8%) vs B1 3.9 (14.6%), B2a 1.7 (14.9%) | inconclusive | "smaller relative gap, not tested for significance" |
| R1 | Baselines reproduced within ~2-3 points of published | E1 | EM/F1 | STATUS.md R1 | n/a | n/a | not applicable (context only) | "published systems use far larger models; not comparable" |

Status values: supported (CI excludes 0 and >=3 seeds agree in sign) / partially supported / not supported (CI includes 0 or sign reverses) / inconclusive (too few seeds or budget).
Allowed wording examples: supported -> "improves ... (+x EM, 95% CI [a,b])"; not supported -> "we did not observe an improvement"; inconclusive -> "results are suggestive but underpowered".

Statuses above are FINAL: test split, 3 seeds, 2026-11-26 (claim rule: 95% CI excludes 0 and all seeds agree in sign).
