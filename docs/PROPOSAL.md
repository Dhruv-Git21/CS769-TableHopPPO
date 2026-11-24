# Research proposal (condensed from the team's proposal doc)

Source doc: https://claude.ai/artifact/8H7Hc2JiDNJJRPPLz9FStT (dated Oct 11, 2026). This file is the working copy for agents.

**Title.** Learning Multi-Hop Table Reasoning through RL: PPO-Based Entity Selection for Decision Making.
**Team.** Dhruv Goyal, Shaurya. **Target venue.** ACL 2027 (main), deadline expected mid-Feb 2027 (estimate, verify).

## Motivation
Multi-hop table reasoning is a sequential selection problem. A question needs path-dependent choices about which entity (row, cell, linked value) to visit next, and the reward is known only when the final answer is checked. Existing table-RL methods reward the whole trace, so no individual hop receives credit. Dhruv has prior experience framing sequential selection as MDP/POMDP (causal RL with Bayesian posterior belief states; counterfactual policy ranking in MARL under sparse delayed reward; agentic RAG with self-critique).

## Related work and gap (anchor papers, all to be re-verified by lit-researcher)
| Paper | Venue | Relevance |
|---|---|---|
| Yang, Chen, Cohan, Zhao. Table-R1: Inference-Time Scaling for Table Reasoning Tasks | EMNLP 2025, pp. 20605-20624 (https://aclanthology.org/2025.emnlp-main.1040/) | RL with verifiable rewards (GRPO) beats distillation-based SFT on multi-step and out-of-domain table tasks. Rewards the whole trace. |
| Bai, Chen, Xiao. Multi-hop temporal knowledge graph reasoning with multi-agent reinforcement learning (MA-TPath) | Applied Soft Computing, 2024 | Separate entity-selection agent and relation agent build paths hop by hop; reward over entity choice, path length, accuracy. Graphs only. |
| Liu, Fu, Fu, Wu, Bai. Multi-hop reasoning over sparse knowledge graphs with deep reinforcement learning (MASKGR) | Expert Systems with Applications, 2025 | Relation agent and entity agent collaborate. Graphs only. |

Gap: no published work does hop-level, PPO-optimized entity selection on tables. Tables have rows/columns/cells rather than triples, so graph agents cannot be reused unchanged.

## Research question
Can multi-hop table reasoning be formulated as a sequential entity-selection MDP, where each hop's choice of row, cell or entity is a discrete, credit-assignable PPO action, and does this improve answer accuracy, interpretability and out-of-domain generalization over (a) trace-level rewards without per-hop credit (GRPO-style table RL) and (b) soft attention / region selection?

## Hypothesis
A separate PPO entity-selection policy, rewarded for intermediate hop correctness as well as final-answer correctness, will:
1. reduce error propagation across hops vs end-to-end trace rewards;
2. produce auditable reasoning paths (each hop is an explicit, logged action);
3. generalize better to unseen table schemas (learns a transferable traversal strategy).

## Contributions
1. Methodological: multi-hop table reasoning as a POMDP over table entities. State = partial reasoning trace + current sub-table view; action = next entity/cell; PPO optimizes the policy.
2. Reward design: hybrid reward = sparse final-answer correctness + dense reward for valid entity transitions, penalty for redundant hops, schema-consistency term.
3. Empirical: controlled comparison of hop-level entity-selection RL vs SFT, trace-level RL and region-based selection on multi-hop table QA, measuring accuracy, path validity, in-domain to OOD gap.

## Technical approach
State: question + partial reasoning trace (entities and cells visited) + candidate sub-table reachable from the last selected entity; tables linearized and encoded by a table-aware encoder or fine-tuned LLM backbone.
Action space: at each hop the next cell/row/linked value, restricted to entities reachable by shared row, column, or explicit cross-reference. STOP ends traversal.
Reward: terminal reward for answer correctness; intermediate rewards for entities on a valid gold path; penalties for repeated hops and schema-inconsistent picks (wrong data type or unit); efficiency penalty for excess hops.
Answer head: smaller fine-tuned LLM or template extractor produces the answer from selected entities and values.
Baselines: (1) SFT on question/table/answer with no hop structure; (2) trace-level RL (GRPO or RLVR, Table-R1 style); (3) region/attention-based selection (soft table-region scoring); (4) MA-TPath/MASKGR-style dual-agent model on a table converted to a graph.
Datasets: HybridQA (multi-hop over cells and linked text), MultiHiertt (hierarchical multi-table finance), TAT-QA or FinQA (supplementary numerical). OOD split holds out whole schemas/domains (e.g., train on financial tables, test on sports or scientific).
Metrics: Exact Match and F1; path validity rate; hop efficiency (hops taken vs minimum); agreement with gold supporting-fact paths; in-domain to OOD accuracy gap.
Testing the claim: table encoder and answer head fixed; only the training and action-space paradigm varies (SFT, trace-level RL, hop-level PPO, soft attention). Bootstrapped confidence intervals across seeds and splits. Extra ablation removes intermediate rewards (terminal-only PPO) to show the dense reward, not just the discrete action space, drives any gain.

## Success criteria (targets, not promises)
| Deliverable | Bar |
|---|---|
| Codebase | Modular, documented; third party can rerun training and evaluation on HybridQA with provided scripts and land close to reported numbers |
| Reproduced baselines | SFT and trace-level RL within ~2-3 EM/F1 points of the original papers on the same splits (else report under identical settings and say so) |
| Hop-level PPO | EM/F1 >= trace-level RL in domain; path validity >= 10-15% higher than best baseline |
| Ablations | Full method beats terminal-reward-only PPO on path validity or hop efficiency, bootstrapped CIs over 3-5 seeds |
| Generalization | In-domain to OOD drop at least 5 points smaller (relative) than SFT and trace-level RL |
| Manuscript | Complete tables, stated significance, qualitative path case studies, honest limitations, ACL format |
| Presentation | Central claim with supporting evidence |

## Resources (original plan)
Compute: PPO on 7B-class open backbone (Qwen2.5 or Llama-3 8B) needs 1-2 GPUs of 40-80 GB; evaluation fits one 24-48 GB GPU; 50-100 GB storage. Access: IIT Bombay institute/department GPUs (not fully confirmed). Fallback: academic cloud credits or 1.5B-3B backbone. Software: PyTorch, HF Transformers, PPO lib (TRL, OpenRLHF or custom), pandas, W&B, Git.

## Risks and fallbacks
1. Compute limits full-scale PPO. Fallback: 1.5B-3B backbone, drop TAT-QA/FinQA, train on 20-30% subset; keep identical matched compute across methods.
2. Intermediate reward noisy or hackable (HybridQA/MultiHiertt gold paths are imperfect). Fallback: terminal-reward-only PPO as the main method and reframe contribution as discrete hop-level actions vs trace-level RL, without claiming gains from dense shaping.
3. If held-out schema or KG-style baseline do not fit: stretch goals.
