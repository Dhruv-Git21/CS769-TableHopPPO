# Live demo guide (5 minutes)

Start the portal before the talk (loading takes about 20 s), then open http://127.0.0.1:8769 :

```bash
make webapp
```

All demo questions come from the **dev** split. Nothing from the test split is shown or scored. The outputs below are from the run named in the page header and were checked on 2026-11-24 with the Hop-PPO seed-0 model (`h1_hoppo_hybridqa_s0_20261123-051021`). Re-check them with the final model before presenting: run `python scripts/pick_showcase.py --run <final H1 run>` and click through the list once.

## What to say first (30 s)
"The model answers a question about a table by hopping. It picks one cell, moves along the row or follows the cell's Wikipedia link, then decides to stop. Only what it picked is passed to a small frozen answer model. So every answer comes with the exact evidence it was based on."

## Script

| # | Where to click | Question | What happens | Point to make |
|---|---|---|---|---|
| 1 | Tab "Hop-PPO wins" | *Who was the coach of the gymnast who finished 8th at the 2001 World Artistic Gymnastics Championships floor exercise?* | Hop 1: cell "Alexander Beresh (UKR)" (p≈0.79). Hop 2: his linked passage (p=1.00). STOP. Answer **Igor Korobchinsky** (correct). SFT answers "Kevin Harmse" (wrong). | A two-hop question: the answer is not in the table at all, only in the passage the policy chose to open. |
| 2 | Follow-up button under the box | *In which city did the gymnast who finished 8th ... die?* | Same path, answer **Kiev** (the passage says he died in Kiev). | Same table, new question, no retraining. The evidence panel shows where "Kiev" comes from. |
| 3 | Tab "Hop-PPO wins" | *Which ride is the longest inverted roller coaster in the world?* | Hop 1: the Banshee description cell (p≈0.99). Hop 2: linked passage. Answer **Banshee** (correct). SFT answers "Thunder Alley and Son of Beast" (wrong). | The first hop matches the question's wording ("longest of its kind in the world"). |
| 4 | Follow-up button | *Which company manufactured the longest inverted roller coaster in the world?* | Answer **Bolliger & Mabillard**, read from the Banshee row (the answer model sees the whole selected row). | Row evidence: one hop gives access to the neighbouring columns. |
| 5 | Follow-up button | *In what year did the longest inverted roller coaster in the world open?* | Answer **2014**. | |
| 6 | Tab "Both right" | *What style of cooking is mastered by the chef of Maison Lameloise?* | Both Hop-PPO and SFT answer **French**. | Easy questions are solved by both; the differences are on multi-hop ones. |
| 7 | Tab "Failure cases" | *River Phoenix played Brian in a sitcom that aired on what network?* | Hop-PPO answers "ABC Afterschool Special" (wrong; gold NBC). The oracle path is right. | Honest limitation: a wrong first hop cannot be repaired later. Most errors are first-hop errors (figure F9). |
| 8 | Follow-up button on the gymnast table | *At which Olympic Games did the gymnast who finished 8th ... win a medal?* | Right evidence (the passage mentions the 2000 Sydney Games), but the answer model says "Kiev". | A second failure type: the path is right but the small answer model misreads it. |

## If asked "can I type anything?"
Yes, about the table currently shown. Expect failures on phrasings unlike HybridQA. For example, "Who coached the gymnast who finished 4th?" returns the 8th-place gymnast's coach, because the word "4th" does not match the rank cell "4" lexically. This is a known limitation of the small first-hop model (README section 12).

## Terminal fallback (no browser)
```bash
bash scripts/demo.sh                 # questions 1, 3 and one more, with B1 and oracle comparison
bash scripts/demo.sh ask             # type questions about the gymnast table
```

## Backup
If the laptop cannot run the model during the talk, show the saved traces and figures from `results/demo/` (`bash scripts/demo.sh` writes them). The deck's demo slide uses the same example.
