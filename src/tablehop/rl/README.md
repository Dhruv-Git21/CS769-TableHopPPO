# tablehop.rl

| File | What it does |
|---|---|
| `reward.py` | `RewardFn`. Per hop: `+w_hop` if on the gold path (in order), `-w_red` for a revisit, `-w_step`. At the end: `w_ans*F1 - w_schema*[type mismatch]`, plus `w_path*pathF1` for B2b only. Switches `terminal_only`, `no_redundancy`, `no_schema` give ablations A1-A3. |
| `rollout.py` | Batched episodes over many environments; calls the answer head once per finished episode |
| `ppo.py` | GAE, clipped PPO loss, value loss, entropy bonus, update loop (used by H1 and A1-A4) |
| `grpo.py` | Group-normalized trajectory advantages, G=8 samples per question (B2a, B2b) |
| `runner.py` | The training loop shared by all RL methods. Logs `train_log.csv` (reward components and reward-hacking signals) and evaluates on 300 dev questions every 50 updates. |

Train with: `python scripts/train.py --config configs/h1_hoppo.yaml --seed 0`.
