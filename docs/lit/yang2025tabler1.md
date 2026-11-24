# Table-R1 (yang2025tabler1) - verified 2026-11-13
Sources: https://aclanthology.org/2025.emnlp-main.1040/ (DOI 10.18653/v1/2025.emnlp-main.1040, pp. 20605-20624); https://arxiv.org/abs/2505.23621 (v2, 2025-09-26); full text read from arXiv PDF.
Problem: inference-time scaling / reasoning post-training for table tasks.
Method: Table-R1-SFT (distill DeepSeek-R1 traces) and Table-R1-Zero (RLVR with GRPO, DAPO-style clipping; reward = accuracy + format, one reward per whole response, no per-step credit).
Base models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct. RL: lr 1e-6, 2 epochs; SFT: lr 1e-5, 3 epochs.
Train data (Table 1): WTQ 13,706; HiTab 6,793; TabFact 20,740; FeTaQA 7,324.
Eval: 13 datasets. In-domain: WTQ, HiTab, TabFact, FeTaQA. OOD: ToTTo, QTSumm, RotoWire, InfoTabs, PubHealthTab, Feverous, TabMCQ, TabMWP, FinQA.
Metrics: EM accuracy (TQA, ambiguous cases rechecked by GPT-4.1 mini), accuracy (TFV), avg of BLEU/ROUGE-L (FF-TQA; reward), BLEU/ROUGE reported per Table 2 caption.
Key numbers (Table 2, Qwen2.5 series): Table-R1-Zero FeTaQA 30.6, TabFact 87.6, WTQ 79.8, HiTab 78.1; Table-R1-SFT 25.3/89.9/81.9/78.3. Llama series Zero: 32.7/87.6/81.2/81.4.
HybridQA / MultiHiertt: NOT used, no numbers. R1 reproduction check cannot reference this paper for them.
Limitations (stated): SFT data only from DeepSeek-R1; filtering may drop hard examples; Llama-3.1-8B RL unstable.
Relation: our trace-level GRPO baseline mirrors its recipe; we add hop-level PPO credit. No explicit multi-hop/path validity evaluation here.
