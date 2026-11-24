# Datasets (inspected 2026-11-13; raw data in data/raw/, gitignored)

## HybridQA (required)
- Source: https://github.com/wenhuchen/HybridQA (commit db22fda8c5951438fade3c69d75b350335ba93b3) questions in `released_data/`; tables+passages: https://github.com/wenhuchen/WikiTables-WithLinks (commit dc066e1a6d5281511d8b73a6107d5ad2824cc2b2). Both fetched via GitHub (no HF mirror needed). Paper: Chen et al., Findings of EMNLP 2020 (per repo README).
- License: HybridQA repo MIT (LICENSE file read). WikiTables-WithLinks repo has no LICENSE file; data is crawled Wikipedia (assume CC BY-SA; unverified).
- Size on disk: HybridQA repo 94MB, WikiTables-WithLinks 785MB (checkout; 2 files differ from index, likely case-collision names). Total ~879MB, under 2GB.
- Splits (counted by loading): train 62,682; dev 3,466; test 3,463. train/dev.traced.json same counts. tables_tok and request_tok each hold 15,314 files.
- Question fields (train/dev): `question_id, question, table_id, answer-text, question_postag`; test lacks `answer-text`. Traced files add `answer-node`.
- `answer-node`: list of nodes, each `[cell_text, [row, col], link_or_null, "passage"|"table"]` e.g. `['Kabul', [0, 2], '/wiki/Kabul', 'passage']`; README says it is approximated by `trace_answer.py` (weak label, not guaranteed correct). No ordered hop path or explicit supporting-fact field; anchor hops must be inferred (heuristic).
- Table file (`tables_tok/<table_id>.json`) keys: `url,title,header,data,intro,section_title,section_text,uid`. `header` items and `data` cells are `[text, [hyperlink,...]]`. `request_tok/<table_id>.json` is a dict mapping hyperlink -> passage text; in the example all 28 table links resolved.
- Test answers: official test labels are via CodaLab. Surprise: the GitHub repo contains `test_answers.json` (2.7MB). I did NOT open it; keep test sealed for the study (decide policy).
- Join snippet: `scripts/inspect_hybridqa.py`. Printed output:
  ```
  question keys: ['answer-node','answer-text','question','question_id','question_postag','table_id']
  answer-node n: 4 node len: 4 types: ['str','list','str','str']
  table keys: ['data','header','intro','section_text','section_title','title','uid','url'] n_rows: 10 n_cols: 5
  header item: list len 2 / cell item: list len 2
  request type: dict n: 26
  links in table: 28 resolved in request: 28
  ```

## MultiHiertt (SHOULD)
- Source: Google Drive folder https://drive.google.com/drive/folders/1ituEWZ5F7G9T9AZ0kzZZLrHNhRigHCZJ (linked from https://github.com/psunlpgroup/MultiHiertt, fetched 2026-11-13). Downloaded with gdown into data/raw/multihiertt/. Paper: Zhao et al., ACL 2022 (per README).
- License: MIT (README link; LICENSE file fetched, header "MIT License"). Size 345MB (train 262MB, dev 36MB, test 52MB).
- Splits: train 7,830; dev 1,044; test 1,566 documents (one `qa` each).
- Fields: `uid, paragraphs (list of str), tables (list, HTML), table_description (dict), qa`. `qa` (train/dev): `question, answer (str), program, question_type, table_evidence, text_evidence`. Evidence: `table_evidence` list of "table-row-col" strings (e.g. '0-2-4'), `text_evidence` list of paragraph indices. No `supporting_facts` field; unordered. Test has only `question`: test answers not public (CodaLab).

## TAT-QA (COULD)
- Source: https://github.com/NExTplusplus/TAT-QA/tree/master/dataset_raw (raw files `tatqa_dataset_{train,dev,test}.json` downloaded; `tatqa_dataset_test_gold.json` exists in repo listing, not downloaded). Paper: Zhu et al., ACL 2021 (per README).
- License: dataset CC BY 4.0 (README line 105); repo MIT.
- Size 15MB. Docs/questions: train 2,201 / 13,215; dev 278 / 1,668; test 278 / 1,669.
- Fields: doc `paragraphs, questions, table`; table `{table, uid}`; paragraph `{uid, order, text}`; question `answer, answer_from, answer_type, derivation, order, question, rel_paragraphs, req_comparison, scale, uid`. Test file questions have only `order, question, uid`; gold released separately (test_gold, per README Jan 2024 note).

## Total download
~1.2GB under data/raw/ (hybridqa 879MB, multihiertt 345MB, tatqa 15MB).

## Model licenses (model cards fetched 2026-11-24)
- BAAI/bge-small-en-v1.5: MIT (https://huggingface.co/BAAI/bge-small-en-v1.5)
- google/flan-t5-small: Apache-2.0 (https://huggingface.co/google/flan-t5-small)
- GRPO attribution verified 2026-11-24: DeepSeekMath abstract (arXiv 2402.03300) "we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO)".
