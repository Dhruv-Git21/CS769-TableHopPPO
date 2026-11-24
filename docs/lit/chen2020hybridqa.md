# chen2020hybridqa: HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data
Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, William Wang | Findings of EMNLP 2020 | https://arxiv.org/abs/2004.07347 | verified: yes

- Problem: multi-hop QA over Wikipedia tables plus linked passages.
- Official download/code: https://github.com/wenhuchen/HybridQA (license MIT per GitHub API, fetched 2026-11-13).
- Split (Table 3, p.4 of arXiv PDF): Train 62,682 / Dev 3,466 / Test 3,463 / Total 69,611. In-Table 26,803 / 1,349 / 1,346; In-Passage 35,215 / 2,025 / 2,045 (test In-Passage printed "20,45" in text extraction; 2,045 is inferred from sum 3,463); Computed 664 / 92 / 72.
- Table stats (Table 2): 13,000 tables, avg 15.7 rows / 4.4 cols; 293,269 passages.
- Baselines, EM/F1 (Table 5, pdf p.8), Dev Total | Test Total:
  - Table-Only 8.4/12.1 | 8.3/11.7
  - Passage-Only 19.5/25.1 | 19.1/25.0
  - HYBRIDER (BERT-base, tau=0.7) 42.9/50.0 | 41.8/49.5
  - HYBRIDER (BERT-base, tau=0.8) 43.5/50.6 | 42.2/49.9
  - HYBRIDER (BERT-base, tau=0.9) 43.7/50.9 | 42.5/50.2
  - HYBRIDER (BERT-large, tau=0.8) 44.0/50.7 | 43.8/50.6
  - Test In-Table / In-Passage for best: 56.2/63.3 | 37.5/44.4.
- Human (Sec 5.2): EM=88.2, F1=93.5 on a 500-question held-out test subset.
- Relation: primary multi-hop table(+text) benchmark; HYBRIDER's hop model is a cell-hop precedent. JSON schema fields not inspected (data not downloaded).
- Verification: arXiv abs page + PDF text extracted locally.
