# zhao2022multihiertt: MultiHiertt: Numerical Reasoning over Multi Hierarchical Tabular and Textual Data
Yilun Zhao, Yunxiang Li, Chenying Li, Rui Zhang | ACL 2022 | https://aclanthology.org/2022.acl-long.454/ | verified: yes

- Problem: numerical reasoning over documents with multiple hierarchical tables plus text.
- Official code/data: https://github.com/psunlpgroup/MultiHiertt (found via GitHub search, MIT; not confirmed as the paper-linked URL). Source data licenses per paper: MIT and CDLA-Permissive-1.0 (FinTabNet).
- Split (Table 3, pdf p.5): Train 7,830 (75%) / Dev 1,044 (10%) / Test 1,566 (15%).
- Baselines EM/F1 (Table 4, pdf p.7), Dev EM F1 | Test EM F1:
  - Longformer + Reasoning 2.71 6.93 | 2.86 6.23
  - Facts Retrieving + TAPAS 8.94 10.70 | 7.67 10.04
  - Facts Retrieving + NumNet 10.32 12.59 | 10.77 12.02
  - TAGOP (RoBERTa-large) 19.16 21.08 | 17.81 19.35
  - Facts Retrieving + Seq2Prog 26.19 28.74 | 24.58 26.30
  - FinQANet (RoBERTa-large) 32.41 35.37 | 31.72 33.60
  - MT2Net (BERT-base) 33.68 35.94 | 32.07 33.67
  - MT2Net (BERT-large) 34.03 36.13 | 33.25 34.98
  - MT2Net (RoBERTa-base) 35.69 37.81 | 34.32 36.17
  - MT2Net (RoBERTa-large) 37.05 39.96 | 36.22 38.43
  - Human expert (test only) 83.12 87.03
- Relation: hierarchical tables, multi-step numeric programs; harder OOD / stress benchmark. Paper reports numeracy-focused F1.
- Verification: ACL Anthology page (title, authors, ACL 2022, pp. 6588-6600, DOI) + arXiv PDF text. JSON schema not inspected.
