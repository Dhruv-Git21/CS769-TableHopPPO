# Interim presentation summary and visual identity

Source deck: https://claude.ai/artifact/Gzm2LtdQabhstdKsMbHpNS (5 slides: cover, related work, problem, approach, progress, next steps). Dated Oct 2026.

## Content established in the interim
- Illustrative example (made up, NOT from any dataset): "In which city is the team of the top scorer based?" Hop 1 Players table: pick row with most goals (Maria Lopez); Hop 2 same row: read team cell (Harbor FC); Hop 3 Teams table: find Harbor FC, read its city (Lisbon).
- Contrast: trace-level RL gives no signal at hops 1-2 and a reward only at the final answer; hop-level PPO gives a reward after every hop (valid hop, valid hop, correct answer).
- Pipeline loop: State (question, visited entities, current sub-table) -> Policy (PPO; table encoder + policy head scores every reachable entity) -> Action (next cell/row/linked value, or STOP) -> Table environment (moves to chosen entity, checks hop) -> feedback (hop reward = valid step minus redundancy; next state; final reward = answer correctness). On STOP a small LLM reads selected entities and writes the answer.
- Table encoder and answer head are held fixed across baselines so only the training paradigm varies.
- Progress status at interim: literature and gap done; formulation and spec done (POMDP state/action/stop/hybrid reward); data/encoder, baselines, PPO were still in progress with no results yet.

## Visual identity for the final deck and paper figures
- Palette: navy `#14213D`, teal `#1F6F8B`, burnt orange `#B4501A`, light orange `#F2B48A`, cream background `#F6F5F0`, card `#FBFAF6`, border `#DAD7CC`, body text `#3D4757`, muted `#6B7280`, banded row `#ECEAE0`, pale teal `#E4EEF1`.
- Fonts: headings Source Serif 4 (fallback Georgia), body IBM Plex Sans (fallback Arial). Use fallbacks if fonts are not installed.
- Layout: 16:9, 128px-equivalent margins, small caps section label in orange above a serif title, footer "Final presentation | Dhruv Goyal and Shaurya | n of 10", cover on navy.
- Chart colors: ours = orange `#B4501A`; baselines = teal `#1F6F8B`, grey `#6B7280`, navy `#14213D`, light teal `#7FB3C4`. Keep colorblind-safe (differ in lightness as well as hue).
