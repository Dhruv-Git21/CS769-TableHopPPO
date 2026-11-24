# zhu2021tatqa: TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance
Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, Tat-Seng Chua | ACL 2021 | https://arxiv.org/abs/2105.07624 | verified: yes

- Problem: QA over financial hybrid contexts (table + paragraphs) with arithmetic.
- Code/data: https://github.com/NExTplusplus/TAT-QA (MIT per GitHub API, fetched 2026-11-13).
- Split (Table 1, pdf p.4): hybrid contexts 2,201 / 278 / 278; questions Train 13,215 / Dev 1,668 / Test 1,669.
- Baselines EM/F1 (Table 3, pdf p.7), Dev | Test: Human test 84.1/90.8; BERT-RC 9.5/17.9 | 9.1/18.7; NumNet+ V2 38.1/48.3 | 37.0/46.9; TaPas for WTQ 18.9/26.5 | 16.6/22.8; HyBrider 6.6/8.3 | 6.3/7.5; TAGOP 55.2/62.7 | 50.1/58.0.
- Relation: optional OOD dataset; mixed table+text.
- Verification: arXiv abs page (comments: Accepted by ACL 2021) + PDF text. Table numbers were located by text extraction; table numbering (1 and 3) not double-checked.
- Schema not inspected.
