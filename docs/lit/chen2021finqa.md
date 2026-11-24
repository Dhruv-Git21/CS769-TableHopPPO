# chen2021finqa: FinQA: A Dataset of Numerical Reasoning over Financial Data
Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, William Yang Wang | EMNLP 2021 | https://arxiv.org/abs/2109.00122 | verified: yes

- Problem: numerical reasoning over financial reports (table + text) producing programs.
- Code/data: https://github.com/czyssrs/FinQA (MIT per GitHub API, fetched 2026-11-13).
- Split (Sec 4.4, pdf p.5-6): 8,281 examples: train 6,251 / val 883 / test 1,147 (75/10/15), no report overlap.
- Metrics are Execution Accuracy / Program Accuracy (not EM/F1). Table 2 (pdf p.7), Exe/Prog: FinQANet (RoBERTa-large) 61.24/58.86; RoBERTa-base 56.10/54.38; BERT-large 53.52/51.62; BERT-base 50.00/48.00; FinBert 50.10/47.52; Retriever+NeRd (BERT-base) 48.57/46.76; Pre-trained Longformer 21.90/20.48; FinQANet-Gold (RoBERTa-large) 70.00/68.76; Human expert 91.16/87.49; General crowd 50.68/48.17.
- Relation: numeric program generation; program steps resemble hops.
- Verification: arXiv abs page (comments: EMNLP 2021) + PDF text. Schema not inspected.
