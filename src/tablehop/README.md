# tablehop package

| Subpackage | Role | Key entry points |
|---|---|---|
| `data/` | Load HybridQA, build gold paths, make the ID/OOD splits, synthetic sanity tasks | `load.load_split`, `schema.Example` |
| `env/` | The decision problem: candidates, features, rendering of entities to text, environment steps | `table_env.TableEnv`, `candidates.CandidateGen` |
| `models/` | Frozen text encoder with cache, GRU policy + value head, frozen answer head | `policy.Policy`, `answer_head.AnswerHead`, `encoder.EmbeddingCache` |
| `rl/` | Reward, batched rollouts, PPO, GRPO, the shared training loop | `runner.run`, `reward.RewardFn` |
| `baselines/` | B1 supervised fine-tuning, B3 soft-region attention | `sft.train`, `soft_region.train` |
| `eval/` | Metrics, bootstrap statistics, hop-by-hop trace for the demo | `metrics.aggregate`, `stats.paired_bootstrap` |
| `utils/` | YAML configs with inheritance and `--override`, run metadata, seeding, device, logging | `config.get_config`, `runmeta.new_run_dir` |

Data flow for one question:

```
Example (data) -> CandidateGen + TableEnv (env) -> Policy picks hops (models)
  -> RewardFn per hop + at the end (rl) -> AnswerHead reads selected evidence (models)
  -> metrics (eval)
```
