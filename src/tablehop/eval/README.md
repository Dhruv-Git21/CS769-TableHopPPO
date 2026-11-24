# tablehop.eval

| File | What it does |
|---|---|
| `metrics.py` | EM and F1 (SQuAD-style normalization), strict/loose path validity, gold-path set-F1 and LCS, hop efficiency, `aggregate()` |
| `stats.py` | Bootstrap CIs (10k resamples), paired bootstrap tests, seed summaries, ID-to-OOD gap |
| `trace_viz.py` | Greedy hop-by-hop trace with probabilities, runner-up candidates and evidence, plus a path figure. Used only by the demo and the web portal. |
