# L1: anchor papers (fetched 2026-11-13)

## Discrepancies vs docs/PROPOSAL.md
- Table-R1: ACL Anthology title is "...for Table Reasoning Tasks" (matches); arXiv title omits "Tasks". Venue, pages, authors all match.
- MA-TPath: matches (Bai, Chen, Xiao; Appl. Soft Comput. 2024). Added: vol 160, article 111727, DOI. Not the same as the 2021 Bai et al. paper (10.1016/j.asoc.2021.107144), a different earlier single-agent work.
- MASKGR: author list confirmed as Liu, Fu, Fu, Wu, Bai (Hongfei Bai), matching PROPOSAL. Added: ESWA vol 290, article 128289, DOI. Acronym MASKGR taken from Bohrium search snippet only (Elsevier page not readable).
- Table-R1 reports NO HybridQA or MultiHiertt number (full text grep: zero hits). Reproduction check R1 cannot compare against Table-R1 on those sets. Its datasets are WTQ, HiTab, TabFact, FeTaQA (train/in-domain) plus 9 OOD sets.
- Table-R1 is single-table, not multi-hop-specific; its reward is trace-level (accuracy + format).
- MA-TPath and MASKGR: only bibliographic metadata verified (Crossref + DOI resolves HTTP 200). Method/reward claims in PROPOSAL are UNVERIFIED (no abstract or full text accessible).

| key | title | venue-year | URL | verified | role |
|---|---|---|---|---|---|
| yang2025tabler1 | Table-R1: Inference-Time Scaling for Table Reasoning Tasks | EMNLP 2025 | https://aclanthology.org/2025.emnlp-main.1040/ (also arXiv 2505.23621) | yes | anchor |
| bai2024matpath | Multi-hop temporal knowledge graph reasoning with multi-agent reinforcement learning | Applied Soft Computing 160, 2024 | https://doi.org/10.1016/j.asoc.2024.111727 | yes (metadata via Crossref; content not read) | anchor |
| liu2025maskgr | Multi-hop reasoning over sparse knowledge graphs with deep reinforcement learning | Expert Systems with Applications 290, 2025 | https://doi.org/10.1016/j.eswa.2025.128289 | yes (metadata via Crossref; content not read) | anchor |
