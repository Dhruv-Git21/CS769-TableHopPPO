# Talking script: TableHop-PPO final presentation

Generated with the deck by `scripts/slides/build_final_deck.py`; numbers come from `results/tables` (dev split, 1 seed).
Planned speaking time: about 14.8 minutes for 15 slides (limit 15 min), then questions.
Backup slides (time 0) are only shown if a question needs them.

| Slide | Title | Presenter | Time |
|---|---|---|---|
| 1 | Title | Dhruv | 0.5 min |
| 2 | The problem | Dhruv | 1 min |
| 3 | Words we use | Shaurya | 0.5 min |
| 4 | Data used | Shaurya | 1 min |
| 5 | Our approach | Dhruv | 1.25 min |
| 6 | System architecture | Dhruv | 0.5 min |
| 7 | Architecture (paper figure) | Dhruv | 0.75 min |
| 8 | How the code is organised | Shaurya | 0.75 min |
| 9 | What we compared | Shaurya | 1 min |
| 10 | Main results | Shaurya | 1.5 min |
| 11 | Learning during training | Dhruv | 0.75 min |
| 12 | Which reward parts matter? | Dhruv | 1.5 min |
| 13 | Where it goes wrong | Shaurya | 1 min |
| 14 | Live demo | Dhruv | 1.75 min |
| 15 | Key findings | Dhruv | 1 min |
| 16 | References | — | 0 min |
| 17 | Thank you | Both | 0 min |
| 18 | Backup: what we fixed | Dhruv | 0 min |
| 19 | Backup: reward details | Dhruv | 0 min |
| 20 | Backup: setup and compute | Shaurya | 0 min |
| 21 | Backup: statistics | Shaurya | 0 min |

## Slide 1: Title
*Dhruv · about 0.5 min*

Hi, we are Dhruv and Shaurya. Our project is about answering questions over tables. Some questions need several lookups in a row, and we train a small model to do those lookups one step at a time. We call each step a hop. We then check whether rewarding every single hop works better than only rewarding the final answer. Everything runs on one laptop, and every number we show comes straight from our results files.

## Slide 2: The problem
*Dhruv · about 1 min*

Here is a real question from our dataset. The table only lists gymnasts and scores. The coach is not in the table; he is on the gymnast's Wikipedia page. So a system has to do three things: find the right row, follow the link, and read the answer. If the first step picks the wrong row, nothing later can fix it. The usual way to train such a system gives one score at the very end, for the whole chain. Our question is simple: what if we give a small reward at every step, so the model learns which steps were good?

## Slide 3: Words we use
*Shaurya · about 0.5 min*

Before we go on, here are the words we will use. A hop is one step: one cell or one linked page. The evidence is what the model picked; the final answer is written only from that. The gold path is the correct chain. The policy is the small network making the choices, and the reward is the score we give it. Our main measure is exact match: how often the answer is exactly right. Path validity says how often the hops were on the correct path.

## Slide 4: Data used
*Shaurya · about 1 min*

We use HybridQA, a public dataset where each question needs a Wikipedia table and usually a linked page. Because we run on a laptop, we train on 3,000 questions. We tune on 1,000 dev questions and report final numbers on 2,045 test questions. No table is shared between these parts, so the model is always tested on new tables, and a script checks that. The dataset does not say which cells lead to the answer, so we build that correct path automatically; it works for almost every question and is usually two hops long.

## Slide 5: Our approach
*Dhruv · about 1.25 min*

This is the core idea. At each step the model gets a short list of possible moves: other cells in the same row or column, the page linked from the current cell, or stop. Equation 1: it scores every move and turns the scores into probabilities. Equation 2 is the reward for each hop: plus 0.2 if the hop is on the correct path, minus 0.1 for revisiting, and a tiny cost per hop. Equation 3 is the reward at the end: how well the answer matches. Equation 4 is PPO, the standard training rule that moves the model towards better-scored choices in small steps. A separate answer model, which we keep fixed, writes the answer from the picked evidence only, so every answer comes with the evidence it used.

## Slide 6: System architecture
*Dhruv · about 0.5 min*

Here is the whole system for one question. One: the question, the table and its linked pages come in. Two: we shortlist possible moves by keyword and meaning similarity. Three: the policy, the only part we train with RL, picks one. Four: the environment makes the move and gives the hop reward. This loops until the policy says stop. Five: the answer model reads only what was picked and writes the answer. The text encoder and the answer model are frozen and shared by every method we compare, so differences come only from how hops are chosen and rewarded.

## Slide 7: Architecture (paper figure)
*Dhruv · about 0.75 min*

This is the same system as our paper figure. Left, the input: the question, the table and a linked page; the orange numbers show the two hops for our example. Middle, the part we train: candidates are shortlisted, the policy scores them and picks one, the environment moves and gives the hop reward, and this loops until stop. Right, the frozen answer model reads only the selected evidence. At the bottom, PPO updates only the policy, using the hop rewards and the final reward.

## Slide 8: How the code is organised
*Shaurya · about 0.75 min*

The code has six building blocks. Data reads the dataset and makes the splits. Env is the game: it lists the moves, makes a move and checks it. Models holds the three networks. RL holds the rewards and the training loop. Baselines are the methods we compare against, and eval computes the scores. Everything is driven by four commands: setup, data, reproduce-all, and webapp for the demo. There are 76 automatic tests, for example that the model can never stop before its first hop.

## Slide 9: What we compared
*Shaurya · about 1 min*

We compare against four kinds of methods. The oracle gives the answer model the correct evidence; it shows the best score possible. The full-table reader is the standard approach: fine-tune on the whole table, no hops. Two RL methods also hop but are scored only at the end; the second one also gets a path score at the end, so it has the same hints as ours. Soft selection picks evidence in one shot. Then our method, and four versions of it with one part removed. Every method gets the same data, the same training budget, three tuning tries and three seeds.

## Slide 10: Main results
*Shaurya · about 1.5 min*

This is the main result, measured as exact match on dev questions. Full-table reader (SFT) gets 26.3 percent; Answer-only RL gets 10.1 percent; Answer+path RL gets 7.4 percent; Soft selection gets 13.9 percent; Hop-PPO (ours) gets 11.4 percent; the oracle gets 60.3 percent, which is the ceiling of the answer model. The full-table reader is clearly the strongest. Compared with the answer-only RL method our method differs by +1.3 points [-0.9, +3.5], which is not a reliable difference. Compared with the answer+path RL method it differs by +4.0 points [+2.0, +6.1]. The table on the right also shows path validity: how often the hops were on the correct path.

## Slide 11: Learning during training
*Dhruv · about 0.75 min*

Left: accuracy on dev questions during training for each method. Right: what our model is actually being paid for. The orange line, the reward for correct hops, rises fast, so the model does learn to land on the correct path. Around update 200 it learns to stop after about two hops, the typical path length, so the cost per hop shrinks. The answer reward improves only slowly. This hints at our main finding on the next slide.

## Slide 12: Which reward parts matter?
*Dhruv · about 1.5 min*

This is the key test of our idea. Each bar is our model with one part removed. The second bar keeps only the final-answer reward, no step rewards. Dropping the step rewards (answer reward only) gives MORE correct answers: ours minus it = -8.9 points [-11.3, -6.5]. The reason is in the hops column: with step rewards the model learns to stop after about two hops, the length of the correct path, while without them it keeps going to four hops and hands the answer model more evidence. So step rewards make the paths shorter and tidier, but did not make the answers more correct. We report this honestly as a negative result. Removing the repeat penalty, the type check, or the move limits changes less.

## Slide 13: Where it goes wrong
*Shaurya · about 1 min*

Left: for each hop number, how often the hop was on the correct path. It drops sharply after the first hop for every method. Right: for every wrong answer, where the first mistake happened. The dark orange part, a wrong first hop, dominates for every method. The first cell is chosen mostly by matching words from the question, and later steps cannot repair a wrong start. That tells us where to improve next: the first step.

## Slide 14: Live demo
*Dhruv · about 1.75 min*

Now the live demo. (Open http://127.0.0.1:8769; full click-by-click guide in docs/DEMO_GUIDE.md.) Click the gymnast question. Point at hop 1, the highlighted row; hop 2, the linked page; then STOP. Show the evidence box: the answer comes only from these two pieces. Point at the comparison: the full-table reader got it wrong. Click the follow-up 'which city did he die': a new question on the same table, answered Kiev. Then click one failure case and say honestly why it fails: a wrong first hop. If the laptop has a problem, use this screenshot.

## Slide 15: Key findings
*Dhruv · about 1 min*

To sum up, four findings. One: Reading the whole table (SFT) is still best at our scale: 26.3% vs 11.4% for our method; all hop-based methods are far below the oracle (60.3%). Two: Our method vs answer-only RL: +1.3 points [-0.9, +3.5], no reliable difference; vs answer+path RL: +4.0 points [+2.0, +6.1]. Three: Step rewards make paths shorter and more on-path, but did not give more correct answers. Four: Most errors start at the first hop: a better first step is the lever, not more reward shaping. Next we would build a better first-hop finder and try a larger answer model. Thank you, we are happy to take questions.

## Slide 16: References
*— · about 0 min*

(Do not read out. Leave on screen only if asked.)

## Slide 17: Thank you
*Both · about 0 min*

Thank you. We are happy to take questions. (Backup slides follow, for questions on what we changed, the reward, the setup and the statistics.)

## Slide 18: Backup: what we fixed
*Dhruv · about 0 min*

If asked why numbers changed during the project: these four changes were all decided on dev questions, applied to every method, before any test score existed.

## Slide 19: Backup: reward details
*Dhruv · about 0 min*

Reward weights were fixed before tuning; tuning only changed the learning rate and the exploration bonus.

## Slide 20: Backup: setup and compute
*Shaurya · about 0 min*

We never claim results for large models; scores are not comparable to published HybridQA systems that use them.

## Slide 21: Backup: statistics
*Shaurya · about 0 min*

Use if someone asks whether a difference is real.
