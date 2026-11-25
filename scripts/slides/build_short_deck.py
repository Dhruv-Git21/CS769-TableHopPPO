"""Short deck: "Pre-end-of-semester project presentation" (title + 5 slides), same template and data as the
final deck (scripts/slides/build_final_deck.py).

usage: python scripts/slides/build_short_deck.py [--out deliverables/pre_end_semester_presentation.pptx]
Slides: 1 problem + objectives, 2 methodology (architecture), 3 implementation + experiments,
4 results, 5 key findings. Every number comes from results/tables via the shared Results class.
"""
# ruff: noqa: ISC004
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).parent))
from build_final_deck import (
    BLACK,
    BODY,
    BODY_B,
    FIG,
    RED,
    ROOT,
    TITLE_FONT,
    WHITE,
    Deck,
    Results,
    pct,
    signed,
)


def build(out: Path) -> None:
    R = Results()
    D = Deck()
    T = R.tag
    split_word = "test" if R.split == "test" else "dev"
    pb2a, pb2b, pa1 = R.pair("b2a_grpo"), R.pair("b2b_grpo_path"), R.pair("a1_terminal")

    def pts(p):
        return "n/a" if not p or p["ci"] is None else f"{signed(p['diff'])} pts [{signed(p['ci'][0])}, {signed(p['ci'][1])}]"

    # ---------------------------------------------------------------- title (not counted)
    s = D.slide("white", number=False)
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.26), Inches(1.32), Inches(16.36), Inches(1.32))
    ln.line.color.rgb = RGBColor.from_string(RED)
    ln.line.width = Pt(2)
    D.text(s, 16.5, 1.09, 2.38, 0.46, "NOV 2026", 25, "Canva Sans Bold", RED)
    D.text(s, 1.26, 1.97, 14, 0.59, "Dhruv Goyal  ·  Shaurya Goyal", 30, BODY_B, RED)
    D.text(s, 1.12, 2.75, 17.5, 2.2, "TABLEHOP", 140, TITLE_FONT, RED)
    D.text(s, 2.78, 4.85, 16.5, 2.2, "PPO", 140, TITLE_FONT, RED)
    D.text(s, 7.9, 5.35, 11.0, 2.2, ["Teaching a small model to answer questions", "over tables one hop at a time"],
           28, BODY, RED)
    l2 = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(11.13), Inches(8.21), Inches(20), Inches(8.21))
    l2.line.color.rgb = RGBColor.from_string(RED)
    l2.line.width = Pt(2)
    D.text(s, 1.26, 9.56, 12, 0.37, "CS769 course project", 19, BODY, RED)
    D.text(s, 11.4, 9.56, 7.5, 0.37, "Pre-end-of-semester project presentation", 19, BODY_B, RED, align=PP_ALIGN.RIGHT)
    D.notes(s, "Hi, we are Dhruv and Shaurya. This is our pre-end-of-semester update on TableHop-PPO: answering "
               "questions over tables one hop at a time, trained with reinforcement learning.")
    D.n = 0  # number the five content slides 1-5

    # ---------------------------------------------------------------- 1 problem + objectives
    s = D.slide("blue", "Problem and objectives", "Some table questions need a chain of lookups, not one")
    D.card(s, 1.12, 2.95, 9.4, 7.0)
    D.text(s, 1.5, 3.2, 8.7, 0.5, "EXAMPLE (HYBRIDQA)", 18, BODY_B, RED)
    D.text(s, 1.5, 3.75, 8.7, 1.3, "“Who was the coach of the gymnast who finished 8th at the 2001 World "
                                   "Championships floor exercise?”", 19, BODY, BLACK)
    for i, (a, b) in enumerate([("Find the row ranked 8th in the table", "Alexander Beresh (UKR)"),
                                ("Follow the cell's link to his Wikipedia page", "“...coached by Igor Korobchinsky...”"),
                                ("Read the answer from the page", "Igor Korobchinsky")]):
        y = 5.3 + i * 1.45
        o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.5), Inches(y), Inches(0.55), Inches(0.55))
        o.fill.solid()
        o.fill.fore_color.rgb = RGBColor.from_string(RED)
        o.line.fill.background()
        o.text_frame.text = str(i + 1)
        o.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        o.text_frame.paragraphs[0].runs[0].font.size = Pt(18)
        o.text_frame.paragraphs[0].runs[0].font.name = BODY_B
        D.text(s, 2.3, y - 0.05, 8.0, 0.5, a, 18, BODY_B, RED)
        D.text(s, 2.3, y + 0.45, 8.0, 0.5, b, 17, BODY, BLACK)
    D.text(s, 11.2, 3.05, 7.7, 0.6, "Problem", 24, BODY_B, WHITE)
    D.text(s, 11.2, 3.7, 7.7, 2.6, ["The answer is often in a page linked from a cell, not in the table.",
                                    "A wrong first step ruins the whole chain.",
                                    "Usual RL scores only the final answer, so no step gets credit."],
           19, BODY, WHITE, bullets=True, space_after=8)
    D.text(s, 11.2, 6.55, 7.7, 0.6, "Objectives", 24, BODY_B, WHITE)
    D.text(s, 11.2, 7.2, 7.7, 2.8, ["Learn hop-by-hop table QA with PPO and a reward at every step.",
                                    "Compare fairly with SFT and answer-only RL on accuracy and paths.",
                                    "Check which reward parts matter (ablations)."],
           19, BODY, WHITE, bullets=True, space_after=8)
    D.notes(s, "The problem: some table questions need several lookups. Here the table lists gymnasts; the coach is only "
               "on the linked Wikipedia page. A wrong first step ruins the chain, and usual RL rewards only the final "
               "answer. Our objectives: learn the lookups step by step with PPO and a reward at every step, compare "
               "fairly with standard methods, and test which reward parts matter.")

    # ---------------------------------------------------------------- 2 methodology
    s = D.slide("white", "Methodology", "Each hop is an action; a reward at every hop; a fixed reader writes the answer",
                pin=True)
    D.figure(s, ROOT / "paper/figures/architecture.png", 1.12, 2.9, 12.6, 6.6,
             "TableHop-PPO architecture: input (a), hop-level policy trained with PPO (b), frozen answer head (c).",
             cap_color=RED)
    D.text(s, 14.2, 2.95, 4.7, 0.6, "Rewards", 22, BODY_B, RED)
    D.text(s, 14.2, 3.6, 4.7, 3.0, [
        [("(1) per hop: ", "b"), ("r", "math"), ("t", "sub"), (" = 0.2·[on gold] − 0.1·[repeat] − 0.02", "math")],
        [("(2) at the end: ", "b"), ("R = F1(answer, gold) − 0.05·[wrong type]", "math")]], 16, color=BLACK,
        space_after=10)
    D.text(s, 14.2, 6.0, 4.7, 4.0, ["Moves: cells in the same row or column, the linked page, or STOP (≤4 hops).",
                                    "Only the small policy (~0.3M params) is trained; encoder and answer model "
                                    "are frozen and shared by all methods."], 16, BODY, RED, bullets=True,
           space_after=8)
    D.notes(s, "Our method in one picture. Left, the input with the two hops numbered. Middle, the only trained part: the "
               "policy scores the candidate moves and picks one; the environment gives a small reward for each hop on "
               "the correct path (equation 1); this repeats until STOP. Right, a frozen answer model reads only the "
               "picked evidence; the answer gives the final reward (equation 2). PPO updates the policy from both.")

    # ---------------------------------------------------------------- 3 implementation + experiments
    s = D.slide("blue", "Implementation and experiments", "One laptop, one shared pipeline, a fair comparison")
    D.text(s, 1.12, 2.95, 7.6, 0.6, "Implementation", 24, BODY_B, WHITE)
    blocks = [("data/", "dataset, gold paths, splits"), ("env/", "moves, rewards, checks"),
              ("models/", "encoder, policy, answer model"), ("rl/", "PPO / GRPO training loop"),
              ("baselines/", "SFT, soft selection"), ("eval/", "metrics, CIs, demo")]
    for i, (a, b) in enumerate(blocks):
        x, y = 1.12 + (i % 2) * 3.9, 3.7 + (i // 2) * 1.55
        D.box(s, x, y, 3.6, 1.25, [a, b], fg=BLACK, size=18)
    D.text(s, 1.12, 8.45, 7.6, 1.6, ["Apple M4 laptop, no GPU; 76 automatic tests; web demo (make webapp)."],
           17, BODY, WHITE)
    rows = [["Method", "What it does"],
            ["Full-table reader (SFT)", "reads the whole table, no hops"],
            ["Answer-only RL (GRPO)", "hops, scored only at the end"],
            ["Answer+path RL (GRPO)", "same + path bonus at the end"],
            ["Soft selection", "picks evidence in one shot"],
            ["Hop-PPO (ours)", "hops, reward at every step"],
            ["Ours minus one part (×4)", "ablations"]]
    D.table(s, 9.6, 3.55, 9.3, rows, [4.2, 5.1], "Methods compared (HybridQA, same budget).", size=16, highlight=5,
            row_h=0.58, cap_color=WHITE)
    D.text(s, 9.6, 7.9, 9.3, 2.2, ["Data: 3,000 training / 1,000 dev / 2,045 test questions on unseen tables.",
                                   "Same 48k training episodes, 3 tuning tries and 3 seeds per method.",
                                   "Metrics: exact match, F1, path validity; 95% bootstrap CIs."],
           17, BODY, WHITE, bullets=True, space_after=6)
    D.notes(s, "Implementation: six code blocks from data to evaluation, run on one laptop with automatic tests and a web "
               "demo. Experiments: HybridQA with 3,000 training questions and unseen tables for dev and test. We compare "
               "against four methods plus ablations, each with the same training budget, tuning tries and seeds, and "
               "we measure exact match, F1 and how often hops follow the correct path.")

    # ---------------------------------------------------------------- 4 results
    s = D.slide("white", "Results", f"Exact match on {split_word} questions ({T})", pin=True)
    D.figure(s, FIG / "results_table.png", 1.12, 2.9, 11.9, 5.6,
             "All methods: exact match with 95% CI, F1, path validity, hops. Green = best learned value per column.",
             cap_color=RED)
    D.figure(s, FIG / "F2_main.png", 13.5, 2.9, 5.4, 3.6, "Exact match per main method.", cap_color=RED)
    D.text(s, 13.5, 7.2, 5.4, 2.8, [f"SFT best: {pct(R.em('b1_sft'))}% vs ours {pct(R.em('h1_hoppo'))}%.",
                                    f"Ours vs answer+path RL: {pts(pb2b)} (every seed).",
                                    f"Ours vs answer-only RL: {pts(pb2a)} (seeds disagree)."],
           16, BODY, RED, bullets=True, space_after=6)
    D.text(s, 1.12, 9.15, 11.9, 1.0, f"Oracle (correct evidence given): {pct(R.m0['em']) if R.m0 else 'n/a'}% — the "
                                     "reader is not the bottleneck; choosing evidence is.", 17, BODY_B, RED)
    D.notes(s, "The results. The full-table reader is the most accurate. Our method has the most valid paths of the learned "
               "methods and beats the RL baseline that gets the same path signal only at the end; against answer-only RL "
               "the difference is within noise. The oracle shows that with the right evidence the same reader gets far "
               "higher, so evidence selection is the bottleneck.")

    # ---------------------------------------------------------------- 5 key findings
    s = D.slide("blue", "Key findings", f"What our results support ({T})")
    D.card(s, 1.12, 2.9, 9.0, 7.15)
    cats, vals = [], []
    for name, v in (("Oracle path", R.m0["em"] if R.m0 else None), ("Full-table reader", R.em("b1_sft")),
                    ("Hop-PPO (ours)", R.em("h1_hoppo")), ("Ours, no step reward", R.em("a1_terminal")),
                    ("Answer-only RL", R.em("b2a_grpo"))):
        if v is not None:
            cats.append(name)
            vals.append(round(100 * v, 1))
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("Exact match (%)", vals)
    ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(1.4), Inches(3.05), Inches(8.45), Inches(6.2), cd).chart
    ch.has_legend = False
    ch.has_title = True
    ch.chart_title.text_frame.text = "Exact match, %"
    tp = ch.chart_title.text_frame.paragraphs[0].runs[0]
    tp.font.size, tp.font.name = Pt(16), BODY_B
    pl = ch.plots[0]
    pl.gap_width = 45
    pl.has_data_labels = True
    pl.data_labels.number_format, pl.data_labels.number_format_is_linked = '0.0"%"', False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    pl.data_labels.font.size, pl.data_labels.font.name = Pt(15), BODY_B
    for k, name in enumerate(cats):
        pt_ = pl.series[0].points[k]
        pt_.format.fill.solid()
        pt_.format.fill.fore_color.rgb = RGBColor.from_string(
            RED if name == "Hop-PPO (ours)" else "F2A37E" if name.startswith("Ours,") else
            "9CA3AF" if name == "Oracle path" else "1F6F8B")
    ca, va = ch.category_axis, ch.value_axis
    ca.reverse_order = True
    ca.tick_labels.font.size, ca.tick_labels.font.name = Pt(15), BODY
    va.visible = False
    va.tick_label_position = XL_TICK_LABEL_POSITION.NONE
    va.major_tick_mark = XL_TICK_MARK.NONE
    va.has_major_gridlines = False
    va.minimum_scale, va.maximum_scale = 0, max(vals) * 1.25
    D.text(s, 1.4, 9.35, 8.5, 0.6, "Red = ours; light orange = ours without step rewards; grey = oracle ceiling.",
           14, color=BLACK)
    a1_bad = pa1 and pa1["ci"] and pa1["ci"][1] < 0 and pa1.get("agree", True)
    findings = ["Reading the whole table (SFT) is still best at our laptop scale.",
                f"Step rewards beat end-of-trace path rewards ({pts(pb2b)}) and give the most valid paths.",
                ("But dropping step rewards gives more correct answers: shorter paths, less evidence."
                 if a1_bad else "Not better than answer-only RL; step rewards mainly shorten paths."),
                "Accuracy tracks how much evidence is read; most errors start at the first hop."]
    for i, t in enumerate(findings):
        D.numbered(s, 10.75, 2.95 + i * 1.6, i + 1, 8.2, t, 19)
    D.notes(s, "Four findings. One, reading the whole table is still best at our scale. Two, rewarding each hop beats "
               "giving the same path signal only at the end, and gives the most valid paths. Three, honestly, removing the "
               "step rewards gives more correct answers, because the model then reads more evidence. Four, most errors "
               "start at the first hop, so that is where we go next. Thank you.")
    D.save(out)
    print(f"[short deck] wrote {out} ({D.n} content slides + title; {T})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "deliverables/pre_end_semester_presentation.pptx"))
    build(Path(ap.parse_args().out))


if __name__ == "__main__":
    main()
