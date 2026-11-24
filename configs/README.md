# configs/

YAML configs with inheritance (`base: <file>`). Any key can be overridden on the command line with `--override a.b=value`.

| File | Method |
|---|---|
| `base.yaml` | Shared defaults: environment (K=32, col_topk=8, max_hops=4), reward weights, PPO hyperparameters |
| `rl_base.yaml` | RL budget (48,000 trajectories), evaluation cadence, GRPO group size, shared answer head path |
| `h1_hoppo.yaml` | **H1** Hop-PPO with the hybrid reward |
| `a1_terminal.yaml` ... `a4_unrestricted.yaml` | Ablations: terminal-only, no redundancy, no schema, unrestricted actions |
| `b2a_grpo.yaml`, `b2b_grpo_path.yaml` | Trace-level GRPO; B2b adds path F1 at the end |
| `b1_sft.yaml`, `b3_soft.yaml` | Supervised fine-tuning; soft-region attention |
| `m0.yaml` | Oracle: answer head on the gold path |
| `answer_head.yaml`, `answer_head_row.yaml` | Shared answer head (row evidence is the one used) |
| `smoke_*.yaml` | Synthetic-data configs for tests and `make smoke` |

The tuning grid (3 trials per method) is defined in `scripts/run_study.py`.
