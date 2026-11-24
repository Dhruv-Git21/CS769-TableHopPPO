# tablehop.baselines

| File | Baseline |
|---|---|
| `sft.py` | **B1.** flan-t5-small fine-tuned on the question plus the BM25-ranked, linearized table rows and the first 200 characters of each linked passage. Budget: 48,000 examples. |
| `soft_region.py` | **B3.** An MLP scores hop-0 candidates and their neighbours. It is trained by the marginal likelihood of candidates that contain the answer (answer signal only). At inference, the top-3 candidates go to the frozen answer head. |

Both register through `scripts/train.py` (configs `b1_sft.yaml`, `b3_soft.yaml`) and are evaluated by `scripts/evaluate.py`.
