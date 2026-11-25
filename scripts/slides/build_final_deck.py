# ruff: noqa: ISC004, PLC0206  (long slide text is split across adjacent string literals on purpose)
"""Final presentation in the team's template style (theme, size and fonts from scripts/slides/template/theme_template.pptx).

usage: python scripts/slides/build_final_deck.py [--out deliverables/final_presentation.pptx]

Every number on a result slide is read from results/tables/*.csv (written by scripts/summarize.py) or
results/runs/*/metrics_*.json; figures come from results/figures/ (scripts/make_figures.py). The split shown
is "test" when every main method has test results, else "dev" (and the slides say so).
Template: blank layout + full-bleed paper background (white or blue), Montserrat Ultra-Bold titles, Poppins
body, red #DC0E20, white cards with a soft shadow, push-pin accent, page number bottom right.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
TPL = ROOT / "scripts/slides/template"
RES = ROOT / "results"
FIG = RES / "figures"
RED, WHITE, BLACK, GREY, BLUE = "DC0E20", "FFFFFF", "1E1E1E", "6B7280", "1D4ED8"
TITLE_FONT, BODY, BODY_B, MATH = "Montserrat Ultra-Bold", "Poppins", "Poppins Bold", "Cambria Math"
SHADOW = ('<a:effectLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:outerShdw blurRad="190500" '
          'dist="38100" dir="5400000" algn="t" rotWithShape="0"><a:srgbClr val="000000"><a:alpha val="22000"/>'
          '</a:srgbClr></a:outerShdw></a:effectLst>')
DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
NAMES = {"m0": "M0 oracle path", "b1_sft": "B1 SFT", "b2a_grpo": "B2a GRPO", "b2b_grpo_path": "B2b GRPO + path",
         "b3_soft": "B3 soft region", "h1_hoppo": "H1 Hop-PPO (ours)", "a1_terminal": "A1 terminal only",
         "a2_noredund": "A2 no redundancy", "a3_noschema": "A3 no schema", "a4_unrestricted": "A4 unrestricted"}


# ----------------------------------------------------------------------------------------------- data
def read_csv(name: str) -> list[dict]:
    f = RES / "tables" / f"{name}.csv"
    return list(csv.DictReader(f.open())) if f.exists() else []


def num(s: str) -> float | None:
    m = re.match(r"\s*([+-]?\d*\.?\d+)", s or "")
    return float(m.group(1)) if m else None


def ci(s: str) -> tuple[float, float] | None:
    m = re.findall(r"[+-]?\d*\.\d+", s or "")
    return (float(m[0]), float(m[1])) if len(m) >= 2 else None


class Results:
    def __init__(self) -> None:
        self.main, self.abl, self.ood, self.paired = (read_csv(n) for n in ("main", "ablations", "ood", "paired"))
        mf = RES / "tables" / "macros.tex"
        self.macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", mf.read_text())) if mf.exists() else {}
        mains = ["b1_sft", "b2a_grpo", "b2b_grpo_path", "b3_soft", "h1_hoppo"]
        have = {r["method"] for r in self.main if r["split"] == "test"}
        self.split = "test" if all(m in have for m in mains) else "dev"
        rows = [r for r in self.main + self.abl if r["split"] == self.split]
        self.by = {r["method"]: r for r in rows}
        self.seeds = max([int(r["n_seeds"]) for r in rows] or [0])
        m0 = RES / "runs/m0_oracle" / f"metrics_{self.split}.json"
        self.m0 = json.loads(m0.read_text()) if m0.exists() else None

    def em(self, m: str) -> float | None:
        return num(self.by[m]["EM mean±std"]) if m in self.by else None

    def cell(self, m: str, key: str) -> str:
        return self.by[m][key] if m in self.by else "n/a"

    def pair(self, other: str, split: str | None = None) -> dict | None:
        sp = split or self.split
        for r in self.paired:
            if r["split"] == sp and r["comparison"] == f"h1_hoppo - {other}":
                key = "".join(p.title().replace("_", "") for p in ("seedagree", other, sp))
                key = "".join(DIGITS[int(c)] if c.isdigit() else c for c in key if c.isalnum())
                agree = self.macros.get(key)  # "yes"/"no"; None when fewer than 2 seeds
                sd = self.macros.get(key.replace("Seedagree", "Seeddiff"), "")
                return {"diff": num(r["diff EM"]), "ci": ci(r["95% CI"]), "p": num(r["p"]),
                        "agree": agree != "no", "seeds": sd}
        return None

    @property
    def tag(self) -> str:
        return f"{self.split} split, {self.seeds} seed{'s' if self.seeds != 1 else ''}"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def signed(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:+.1f}"


# ------------------------------------------------------------------------------------------- drawing
class Deck:
    def __init__(self) -> None:
        self.prs = Presentation(str(TPL / "theme_template.pptx"))
        sldIdLst = self.prs.slides._sldIdLst  # drop the template's slides, keep its master/theme/size
        for sid in list(sldIdLst):
            self.prs.part.drop_rel(sid.rId)
            sldIdLst.remove(sid)
        self.blank = next(l for l in self.prs.slide_layouts if l.name == "Blank")
        self.n = 0
        self.fig_no = 0
        self.tab_no = 0

    # basic primitives --------------------------------------------------------------------------
    def slide(self, bg: str, title: str | None = None, kicker: str | None = None, pin: bool = False,
              number: bool = True):
        s = self.prs.slides.add_slide(self.blank)
        crop = (1 - 11.25 / (20 * 3092 / 2384)) / 2  # template: image scaled to width, centred vertically
        p = s.shapes.add_picture(str(TPL / f"bg_{bg}.jpg"), 0, 0, Inches(20), Inches(11.25))
        p.crop_top = p.crop_bottom = crop
        p.name = "Background"
        self.n += 1
        fg = WHITE if bg == "blue" else RED
        s._fg = fg
        if title:
            self.text(s, 1.12, 0.62, 17.0 if not pin else 15.2, 1.25, title.upper(), 58, TITLE_FONT, fg,
                      anchor=MSO_ANCHOR.BOTTOM, name="Title")
        if kicker:
            self.text(s, 1.12, 1.98, 17.8, 0.6, kicker, 24, BODY, fg, name="Kicker")
        if pin:
            s.shapes.add_picture(str(TPL / "pin.png"), Inches(18.09), Inches(0.79), Inches(1.57), Inches(1.77))
        if number:
            self.text(s, 17.4, 10.42, 1.5, 0.4, str(self.n), 16, BODY, fg, align=PP_ALIGN.RIGHT)
        return s

    def text(self, s, x, y, w, h, content, size=19, font=BODY, color=None, bold_font=BODY_B, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, bullets=False, space_after=6, name=None, line=None):
        """content: str | list of paragraphs; a paragraph is str or list of (text, style) runs where style
        in {'b','i','math','sub','sup',None}. '**x**' inside a str marks bold."""
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        if name:
            tb.name = name
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = anchor
        paras = content if isinstance(content, list) else [content]
        color = color or s._fg
        for i, para in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.space_after = Pt(space_after)
            if line:
                p.line_spacing = line
            runs = para if isinstance(para, list) else self._md(para)
            if bullets:
                self._bullet(p, color)
            for t, st in runs:
                r = p.add_run()
                r.text = t
                f = r.font
                f.size = Pt(size)
                f.color.rgb = RGBColor.from_string(color)
                f.name = MATH if st == "math" else (bold_font if st == "b" else font)
                f.italic = st == "i"
                if st in ("sub", "sup"):
                    f.name = font
                    r._r.get_or_add_rPr().set("baseline", "-25000" if st == "sub" else "30000")
        return tb

    @staticmethod
    def _md(s: str) -> list[tuple[str, str | None]]:
        out = []
        for k, part in enumerate(re.split(r"\*\*", s)):
            if part:
                out.append((part, "b" if k % 2 else None))
        return out

    @staticmethod
    def _bullet(p, color):
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.32))))
        pPr.set("indent", str(-Emu(Inches(0.32))))
        for tag in ("a:buClr", "a:buFont", "a:buChar"):
            el = pPr.find(qn(tag))
            if el is not None:
                pPr.remove(el)
        clr = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(clr, qn("a:srgbClr")).set("val", color)
        etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
        etree.SubElement(pPr, qn("a:buChar")).set("char", "•")

    def card(self, s, x, y, w, h, fill=WHITE):
        r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        r.fill.solid()
        r.fill.fore_color.rgb = RGBColor.from_string(fill)
        r.line.fill.background()
        sp = r._element.spPr
        sp.append(etree.fromstring(SHADOW))
        r.text_frame.text = ""
        return r

    def figure(self, s, path: Path, x, y, w, h, caption: str, cap_color=None):
        """White card with the figure fitted inside, caption 'Figure N. ...' underneath."""
        self.fig_no += 1
        self.card(s, x, y, w, h)
        if path.exists():
            from PIL import Image
            iw, ih = Image.open(path).size
            pad = 0.15
            bw, bh = w - 2 * pad, h - 2 * pad
            sc = min(bw / iw, bh / ih)
            pw, ph = iw * sc, ih * sc
            s.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
        else:
            self.text(s, x + 0.3, y + h / 2 - 0.3, w - 0.6, 0.6, f"[figure pending: {path.name}]", 18, BODY, GREY,
                      align=PP_ALIGN.CENTER)
        self.text(s, x, y + h + 0.12, w, 0.6, [[(f"Figure {self.fig_no}. ", "b"), (caption, None)]], 15,
                  color=cap_color or s._fg)

    def table(self, s, x, y, w, rows: list[list[str]], col_w: list[float], caption: str, size=15,
              highlight: int | None = None, row_h=0.5, cap_color=None, center: set | None = None):
        self.tab_no += 1
        self.text(s, x, y - 0.55, w, 0.45, [[(f"Table {self.tab_no}. ", "b"), (caption, None)]], 15,
                  color=cap_color or s._fg)
        shp = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                                 Inches(row_h * len(rows)))
        tbl = shp.table
        tblPr = tbl._tbl.tblPr
        for a in ("bandRow", "firstRow"):
            tblPr.set(a, "0")
        sid = tblPr.find(qn("a:tableStyleId"))
        if sid is not None:
            sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # "No Style, No Grid"
        for j, cw in enumerate(col_w):
            tbl.columns[j].width = Inches(cw)
        for i, row in enumerate(rows):
            tbl.rows[i].height = Inches(row_h)
            for j, val in enumerate(row):
                c = tbl.cell(i, j)
                c.margin_left = c.margin_right = Inches(0.08)
                c.margin_top = c.margin_bottom = Inches(0.04)
                c.vertical_anchor = MSO_ANCHOR.MIDDLE
                c.fill.solid()
                head = i == 0
                hl = highlight is not None and i == highlight
                c.fill.fore_color.rgb = RGBColor.from_string(RED if head else ("FCE4E6" if hl else WHITE))
                tf = c.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.alignment = PP_ALIGN.CENTER if (center and j in center) else PP_ALIGN.LEFT
                r = p.add_run()
                r.text = str(val)
                r.font.size = Pt(size)
                r.font.name = BODY_B if (head or hl) else BODY
                r.font.color.rgb = RGBColor.from_string(WHITE if head else BLACK)
                # thin bottom border
                tcPr = c._tc.get_or_add_tcPr()
                ln = etree.SubElement(tcPr, qn("a:lnB"), w="6350")
                sf = etree.SubElement(ln, qn("a:solidFill"))
                etree.SubElement(sf, qn("a:srgbClr")).set("val", "E5E5E5")
        return shp

    def numbered(self, s, x, y, k, w, body, size=19, color=None):
        color = color or s._fg
        o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(0.62), Inches(0.62))
        o.fill.solid()
        o.fill.fore_color.rgb = RGBColor.from_string(WHITE if s._fg == WHITE else RED)
        o.line.fill.background()
        tf = o.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = str(k)
        r.font.size, r.font.name = Pt(20), BODY_B
        r.font.color.rgb = RGBColor.from_string(RED if s._fg == WHITE else WHITE)
        self.text(s, x + 0.9, y + 0.02, w - 0.9, 2.0, body, size, color=color)

    def stat(self, s, x, y, w, big, small, big_size=48, color=None):
        self.text(s, x, y, w, 1.0, big, big_size, TITLE_FONT, color or s._fg)
        self.text(s, x, y + 1.1, w, 1.5, small, 17, BODY, color or s._fg)

    def box(self, s, x, y, w, h, label, fill=WHITE, fg=RED, size=17, bold=True, line=None):
        r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        r.adjustments[0] = 0.12
        r.fill.solid()
        r.fill.fore_color.rgb = RGBColor.from_string(fill)
        if line:
            r.line.color.rgb = RGBColor.from_string(line)
            r.line.width = Pt(2.25)
        else:
            r.line.fill.background()
        r._element.spPr.append(etree.fromstring(SHADOW))
        tf = r.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.12)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        lines = label if isinstance(label, list) else [label]
        for i, ln in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            rr = p.add_run()
            rr.text = ln
            rr.font.size = Pt(size if i == 0 else size - 3)
            rr.font.name = BODY_B if (bold and i == 0) else BODY
            rr.font.color.rgb = RGBColor.from_string(fg)
        return r

    def arrow(self, s, x1, y1, x2, y2, color=None, width=2.5):
        c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        c.line.color.rgb = RGBColor.from_string(color or s._fg)
        c.line.width = Pt(width)
        ln = c.line._get_or_add_ln()
        etree.SubElement(ln, qn("a:tailEnd"), type="triangle", w="med", len="med")
        return c

    def notes(self, s, text: str):
        s.notes_slide.notes_text_frame.text = text

    def save(self, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(out))


# --------------------------------------------------------------------------------------------- slides
def build(out: Path, script_out: Path | None = None) -> None:
    R = Results()
    D = Deck()
    T = R.tag
    split_word = "test" if R.split == "test" else "dev"
    SCRIPT: list[dict] = []  # talking script: one entry per slide

    def say(s, title, minutes, who, text):
        D.notes(s, f"[{who} · ~{minutes} min] " + text)
        SCRIPT.append({"n": D.n, "title": title, "min": minutes, "who": who, "text": text})

    def key(s, x, y, w, rows, title="Key"):
        """Small 'names used on this slide' table."""
        D.table(s, x, y, w, [["Name on slide", "What it means"]] + rows, [w * 0.36, w * 0.64], title, size=13,
                row_h=0.4, cap_color=s._fg)

    order = ["b1_sft", "b2a_grpo", "b2b_grpo_path", "b3_soft", "h1_hoppo"]
    pb1, pb2a, pb2b, pa1 = R.pair("b1_sft"), R.pair("b2a_grpo"), R.pair("b2b_grpo_path"), R.pair("a1_terminal")

    def pts(p):  # "+1.3 points [−0.9, +3.5]"
        return "n/a" if not p or p["ci"] is None else f"{signed(p['diff'])} points [{signed(p['ci'][0])}, {signed(p['ci'][1])}]"

    def reliable(p):  # claim rule: 95% CI excludes 0 AND every seed has the same sign
        return bool(p and p["ci"] and (p["ci"][0] > 0 or p["ci"][1] < 0) and p.get("agree", True))

    # 1 title -----------------------------------------------------------------------------------
    s = D.slide("white", number=False)
    line = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.26), Inches(1.32), Inches(16.36), Inches(1.32))
    line.line.color.rgb = RGBColor.from_string(RED)
    line.line.width = Pt(2)
    D.text(s, 16.5, 1.09, 2.38, 0.46, "NOV 2026", 25, "Canva Sans Bold", RED)
    D.text(s, 1.26, 1.97, 14, 0.59, "Dhruv Goyal  ·  Shaurya Goyal", 30, BODY_B, RED)
    D.text(s, 1.12, 2.75, 17.5, 2.2, "TABLEHOP", 140, TITLE_FONT, RED)
    D.text(s, 2.78, 4.85, 16.5, 2.2, "PPO", 140, TITLE_FONT, RED)
    D.text(s, 7.9, 5.35, 11.0, 2.2, ["Teaching a small model to answer questions",
                                     "over tables one hop at a time"], 28, BODY, RED)
    l2 = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(11.13), Inches(8.21), Inches(20), Inches(8.21))
    l2.line.color.rgb = RGBColor.from_string(RED)
    l2.line.width = Pt(2)
    D.text(s, 1.26, 9.56, 12, 0.37, "CS769 course project  ·  [TEAM TO CONFIRM: department, institute]", 19, BODY, RED)
    D.text(s, 13.9, 9.56, 5.0, 0.37, "Final presentation", 19, BODY, RED, align=PP_ALIGN.RIGHT)
    say(s, "Title", 0.5, "Dhruv",
        "Hi, we are Dhruv and Shaurya. Our project is about answering questions over tables. Some questions need "
        "several lookups in a row, and we train a small model to do those lookups one step at a time. We call each "
        "step a hop. We then check whether rewarding every single hop works better than only rewarding the final "
        "answer. Everything runs on one laptop, and every number we show comes straight from our results files.")

    # 2 problem ---------------------------------------------------------------------------------
    s = D.slide("blue", "The problem", "Some questions need a chain of lookups, not one")
    D.card(s, 1.12, 2.95, 9.6, 6.9)
    D.text(s, 1.5, 3.2, 8.9, 0.5, "A REAL QUESTION FROM THE DATA", 18, BODY_B, RED)
    D.text(s, 1.5, 3.75, 8.9, 1.3, "“Who was the coach of the gymnast who finished 8th at the 2001 World "
                                   "Artistic Gymnastics Championships floor exercise?”", 19, BODY, BLACK)
    steps = [("1", "Look in the table: find the row ranked 8th", "Alexander Beresh (UKR)"),
             ("2", "Follow that cell's link to his Wikipedia page", "“...coached by Igor Korobchinsky...”"),
             ("3", "Read the answer from the page", "Igor Korobchinsky")]
    for i, (k, a, b) in enumerate(steps):
        y = 5.35 + i * 1.45
        o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.5), Inches(y), Inches(0.55), Inches(0.55))
        o.fill.solid(); o.fill.fore_color.rgb = RGBColor.from_string(RED); o.line.fill.background()
        o.text_frame.text = k
        o.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        o.text_frame.paragraphs[0].runs[0].font.size = Pt(18)
        o.text_frame.paragraphs[0].runs[0].font.name = BODY_B
        D.text(s, 2.3, y - 0.05, 8.1, 0.5, a, 18, BODY_B, RED)
        D.text(s, 2.3, y + 0.45, 8.1, 0.5, b, 17, BODY, BLACK)
    D.text(s, 11.4, 3.05, 7.5, 0.6, "Why it is hard", 24, BODY_B, WHITE)
    D.text(s, 11.4, 3.75, 7.5, 3.0, [
        "The answer is often not in the table at all; it is in a page linked from a cell.",
        "If the first step is wrong, every later step is wrong too.",
        "Usual training gives one score for the whole chain, so the model cannot tell which step was good."],
        19, BODY, WHITE, bullets=True, space_after=10)
    D.text(s, 11.4, 7.15, 7.5, 0.6, "Our question", 24, BODY_B, WHITE)
    D.text(s, 11.4, 7.85, 7.5, 2.2, "If we **reward every step** instead of only the final answer, does the model "
                                    "answer more questions correctly and follow better paths?", 20, BODY, WHITE)
    say(s, "The problem", 1.0, "Dhruv",
        "Here is a real question from our dataset. The table only lists gymnasts and scores. The coach is not in the "
        "table; he is on the gymnast's Wikipedia page. So a system has to do three things: find the right row, follow "
        "the link, and read the answer. If the first step picks the wrong row, nothing later can fix it. The usual way "
        "to train such a system gives one score at the very end, for the whole chain. Our question is simple: what if "
        "we give a small reward at every step, so the model learns which steps were good?")

    # 3 glossary --------------------------------------------------------------------------------
    s = D.slide("white", "Words we use", "Every technical term in this talk, in plain words", pin=True)
    rows = [["Term", "Plain meaning"],
            ["Hop", "One step: look at one table cell or one linked Wikipedia page"],
            ["Evidence", "The cells and pages the model picked; the answer is written from these only"],
            ["Gold path", "The correct chain of cells and pages for a question"],
            ["Policy", "The small neural network that decides the next hop"],
            ["Reward", "A score that tells the policy how good a decision was"],
            ["RL (PPO, GRPO)", "Reinforcement learning: learn by trying and being scored. PPO and GRPO are two "
                               "standard training recipes"],
            ["SFT", "Supervised fine-tuning: train directly on question and answer pairs"],
            ["Exact match", "Share of questions where the answer is exactly right (our main score)"],
            ["Path validity", "Share of hops that land on the gold path"],
            ["Dev / test", "Practice questions used for tuning / unseen questions for the final score"],
            ["95% CI", "Confidence interval: the range the true score very likely lies in"]]
    D.table(s, 1.12, 3.2, 17.8, rows, [3.6, 14.2], "Glossary.", size=17, row_h=0.53, cap_color=RED)
    say(s, "Words we use", 0.5, "Shaurya",
        "Before we go on, here are the words we will use. A hop is one step: one cell or one linked page. The evidence "
        "is what the model picked; the final answer is written only from that. The gold path is the correct chain. The "
        "policy is the small network making the choices, and the reward is the score we give it. Our main measure is "
        "exact match: how often the answer is exactly right. Path validity says how often the hops were on the correct "
        "path.")

    # 4 data ------------------------------------------------------------------------------------
    s = D.slide("blue", "Data used", "HybridQA: questions over Wikipedia tables and linked pages (Chen et al., 2020)")
    rows = [["Part", "Questions", "Where it comes from", "Used for"],
            ["Training", "3,000", "official training set", "learning (every method)"],
            ["Dev", "1,000", "official training set, unseen tables", "tuning and choosing settings"],
            ["Test", "2,045", "official dev set, unseen tables", "final scores (once per model)"]]
    D.table(s, 1.12, 3.55, 11.0, rows, [1.9, 1.8, 4.1, 3.2], "How we split the data.",
            size=17, center={1}, row_h=0.62, cap_color=WHITE)
    D.text(s, 12.8, 3.05, 6.1, 0.6, "Keeping it fair", 24, BODY_B, WHITE)
    D.text(s, 12.8, 3.75, 6.1, 4.2, [
        "No table appears in two parts, so the model is always tested on tables it has never seen.",
        "The official test answers are never used.",
        "A script checks this automatically before every run."], 19, BODY, WHITE, bullets=True, space_after=10)
    dsr = {r["split"]: r for r in read_csv("data_stats")}
    cov = f"{100 * float(dsr['train']['frac_heuristic']):.0f}%" if "train" in dsr else "n/a"
    D.stat(s, 1.12, 7.0, 5.4, cov, "of training questions get a gold path (the correct chain), built automatically "
                                   "from the dataset's answer location", color=WHITE)
    D.stat(s, 7.0, 7.0, 5.4, "2 hops", "is the typical length of a gold path: a table cell, then the answer cell or a "
                                      "linked page", color=WHITE)
    D.stat(s, 12.8, 7.0, 6.1, "Free", "HybridQA is MIT licensed; both pretrained models we use are open source "
                                     "(MIT, Apache-2.0)", color=WHITE)
    say(s, "Data used", 1.0, "Shaurya",
        "We use HybridQA, a public dataset where each question needs a Wikipedia table and usually a linked page. "
        "Because we run on a laptop, we train on 3,000 questions. We tune on 1,000 dev questions and report final "
        "numbers on 2,045 test questions. No table is shared between these parts, so the model is always tested on "
        "new tables, and a script checks that. The dataset does not say which cells lead to the answer, so we build "
        "that correct path automatically; it works for almost every question and is usually two hops long.")

    # 5 approach + equations ----------------------------------------------------------------------
    s = D.slide("white", "Our approach", "Answering = a short sequence of choices, trained with rewards", pin=True)
    D.card(s, 1.12, 2.95, 10.6, 7.1)
    eq = [
        ("(1)", "Choose the next hop", [("P(hop to c", "math"), ("i", "sub"), (") = softmax", "math"), ("i", "sub"),
                                        ("[ score(state, c", "math"), ("i", "sub"), (") ]", "math")]),
        ("(2)", "Reward for each hop", [("r", "math"), ("t", "sub"), (" = 0.2·[on gold path] − 0.1·[repeat] − 0.02", "math")]),
        ("(3)", "Reward at the end", [("R = F1(answer, gold answer) − 0.05·[wrong answer type]", "math")]),
        ("(4)", "Training rule (PPO)", [("L = E[ min( ρ·A, clip(ρ, 0.8, 1.2)·A ) ]", "math")]),
    ]
    for i, (k, lab, runs) in enumerate(eq):
        y = 3.2 + i * 1.05
        D.text(s, 1.45, y, 3.0, 0.9, lab, 16, BODY_B, RED)
        D.text(s, 4.6, y, 6.3, 0.9, [runs], 19, color=BLACK)
        D.text(s, 10.85, y, 0.7, 0.5, k, 18, BODY_B, RED, align=PP_ALIGN.RIGHT)
    D.text(s, 1.45, 7.45, 10.0, 2.5, [
        "**Symbols.** c_i = a candidate cell or page; state = the question plus the hops made so far; softmax turns "
        "scores into probabilities; [x] = 1 if x is true, else 0; F1 = word overlap between answers (0 to 1); "
        "ρ = how much the new policy changed a choice's probability; A = how much better that choice was than "
        "expected; clip keeps each update small."], 15, color=BLACK, space_after=4)
    D.text(s, 12.4, 3.05, 6.5, 0.6, "In plain words", 24, BODY_B, BLUE)
    D.text(s, 12.4, 3.75, 6.5, 6.2, [
        "At each step the model sees a short list of moves: cells in the same row or column, the cell's linked "
        "page, or STOP.",
        "It scores each move (1) and picks one; at most 4 hops.",
        "Each hop earns a small reward if it is on the gold path (2); the final answer earns the big reward (3).",
        "PPO (4) nudges the model towards choices that scored better.",
        "A separate, fixed answer model writes the answer from the picked evidence only."], 19, BODY, BLUE,
        bullets=True, space_after=10)
    say(s, "Our approach", 1.25, "Dhruv",
        "This is the core idea. At each step the model gets a short list of possible moves: other cells in the same row "
        "or column, the page linked from the current cell, or stop. Equation 1: it scores every move and turns the "
        "scores into probabilities. Equation 2 is the reward for each hop: plus 0.2 if the hop is on the correct path, "
        "minus 0.1 for revisiting, and a tiny cost per hop. Equation 3 is the reward at the end: how well the answer "
        "matches. Equation 4 is PPO, the standard training rule that moves the model towards better-scored choices in "
        "small steps. A separate answer model, which we keep fixed, writes the answer from the picked evidence only, "
        "so every answer comes with the evidence it used.")

    # 6 architecture ----------------------------------------------------------------------------
    s = D.slide("blue", "System architecture", "How one question becomes an answer with evidence")
    xs = [1.12, 4.85, 8.58, 12.31, 16.04]
    labels = [["1. Question + table", "and the linked pages"], ["2. Shortlist moves", "keyword + meaning match"],
              ["3. Policy picks a hop", "small network (trained)"], ["4. Move and score", "reward for this hop"],
              ["5. Answer model", "reads evidence, writes answer"]]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        D.box(s, x, 3.2, 3.0, 1.8, lab, fill=RED if i == 2 else WHITE, fg=WHITE if i == 2 else RED, size=18)
        if i < 4:
            D.arrow(s, x + 3.05, 4.1, xs[i + 1] - 0.05, 4.1, WHITE)
    for (a, b, c, d) in ((13.81, 5.05, 13.81, 5.85), (13.81, 5.85, 10.08, 5.85)):
        ln_ = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(a), Inches(b), Inches(c), Inches(d))
        ln_.line.color.rgb = RGBColor.from_string(WHITE)
        ln_.line.width = Pt(2.5)
    D.arrow(s, 10.08, 5.85, 10.08, 5.05, WHITE)
    D.text(s, 7.6, 5.95, 8.5, 0.5, "repeat until the policy says STOP (at most 4 hops)", 17, BODY, WHITE,
           align=PP_ALIGN.CENTER)
    D.text(s, 15.8, 5.15, 3.4, 0.9, "final reward = how good the answer is", 15, BODY, WHITE, align=PP_ALIGN.CENTER)
    parts = [("Frozen text encoder", "turns every cell and page into numbers (bge-small, not trained)"),
             ("Policy network", "~0.3M parameters, the only part trained by RL; runs on a laptop CPU"),
             ("Frozen answer model", "flan-t5-small, trained once, then shared unchanged by every method")]
    for i, (a, b) in enumerate(parts):
        x = 1.12 + i * 6.0
        D.card(s, x, 7.0, 5.6, 2.3)
        D.text(s, x + 0.3, 7.25, 5.0, 0.6, a, 20, BODY_B, RED)
        D.text(s, x + 0.3, 7.95, 5.0, 1.8, b, 17, BODY, BLACK)
    say(s, "System architecture", 0.5, "Dhruv",
        "Here is the whole system for one question. One: the question, the table and its linked pages come in. Two: we "
        "shortlist possible moves by keyword and meaning similarity. Three: the policy, the only part we train with "
        "RL, picks one. Four: the environment makes the move and gives the hop reward. This loops until the policy "
        "says stop. Five: the answer model reads only what was picked and writes the answer. The text encoder and the "
        "answer model are frozen and shared by every method we compare, so differences come only from how hops are "
        "chosen and rewarded.")

    # 6b publication architecture figure (TikZ, paper/figures/architecture.tex)
    s = D.slide("white", "Architecture (paper figure)", "Full model: input, hop-level policy loop, frozen answer head, PPO update", pin=True)
    D.figure(s, ROOT / "paper/figures/architecture.png", 1.12, 2.9, 17.8, 6.9,
             "TableHop-PPO architecture (same figure as in the paper).", cap_color=RED)
    say(s, "Architecture (paper figure)", 0.75, "Dhruv",
        "This is the same system as our paper figure. Left, the input: the question, the table and a linked page; the "
        "orange numbers show the two hops for our example. Middle, the part we train: candidates are shortlisted, the "
        "policy scores them and picks one, the environment moves and gives the hop reward, and this loops until stop. "
        "Right, the frozen answer model reads only the selected evidence. At the bottom, PPO updates only the policy, "
        "using the hop rewards and the final reward.")

    # 7 code structure --------------------------------------------------------------------------
    s = D.slide("white", "How the code is organised", "Six building blocks and four commands", pin=True)
    blocks = [("data/", "reads the dataset, builds gold paths, splits the data"),
              ("env/", "the game: lists moves, makes a move, checks it"),
              ("models/", "text encoder, policy network, answer model"),
              ("rl/", "rewards and the training loop (PPO, GRPO)"),
              ("baselines/", "the comparison methods (SFT, soft selection)"),
              ("eval/", "scores, confidence intervals, demo traces")]
    for i, (a, b) in enumerate(blocks):
        x = 1.12 + (i % 3) * 3.9
        y = 3.05 + (i // 3) * 2.75
        D.box(s, x, y, 3.5, 1.0, a, fg=BLACK, size=20)
        D.text(s, x, y + 1.15, 3.5, 1.3, b, 16, BODY, BLACK, align=PP_ALIGN.CENTER)
        if i % 3 < 2:
            D.arrow(s, x + 3.55, y + 0.5, x + 3.85, y + 0.5, RED)
    D.text(s, 1.12, 8.65, 11.4, 1.5, ["All in src/tablehop/; every folder has a README. 76 automatic tests check "
                                      "the rules (for example: STOP is never allowed as the first move)."], 16, BODY, BLACK)
    D.text(s, 13.4, 3.05, 5.5, 0.6, "Run it in four commands", 22, BODY_B, BLACK)
    cmds = [("make setup", "install everything"), ("make data", "download and split the data"),
            ("make reproduce-all", "train, tune and score every method"),
            ("make webapp", "open the live demo")]
    for i, (c, d) in enumerate(cmds):
        y = 3.8 + i * 1.55
        D.box(s, 13.4, y, 5.5, 1.15, [c, d], fill=RED, fg=WHITE, size=19)
        if i < 3:
            D.arrow(s, 16.15, y + 1.17, 16.15, y + 1.5, RED)
    say(s, "How the code is organised", 0.75, "Shaurya",
        "The code has six building blocks. Data reads the dataset and makes the splits. Env is the game: it lists the "
        "moves, makes a move and checks it. Models holds the three networks. RL holds the rewards and the training "
        "loop. Baselines are the methods we compare against, and eval computes the scores. Everything is driven by "
        "four commands: setup, data, reproduce-all, and webapp for the demo. There are 76 automatic tests, for "
        "example that the model can never stop before its first hop.")

    # 8 what we compared --------------------------------------------------------------------------
    s = D.slide("blue", "What we compared", "Same data, same budget, same answer model for every method")
    rows = [["Method", "Code", "What it does", "Why it is here"],
            ["Oracle path", "M0", "answer model given the correct evidence", "best possible score"],
            ["Full-table reader (SFT)", "B1", "reads the whole table at once, no hops", "the standard approach"],
            ["Answer-only RL (GRPO)", "B2a", "hops, scored only on the final answer", "reward only at the end"],
            ["Answer+path RL (GRPO)", "B2b", "same, plus a path score at the end", "same hints as ours, at the end"],
            ["Soft selection", "B3", "picks evidence in one shot, no sequence", "no step-by-step choices"],
            ["Hop-PPO (ours)", "H1", "hops, reward at every step", "our method"],
            ["Ours, minus one part", "A1–A4", "answer reward only / no repeat penalty / no type check / any-cell "
                                             "moves", "which parts matter"]]
    D.table(s, 1.12, 3.5, 17.8, rows, [4.2, 1.3, 7.3, 5.0], "Methods (codes are the names used in the paper).",
            size=16, highlight=6, row_h=0.66, cap_color=WHITE, center={1})
    D.text(s, 1.12, 9.15, 17.8, 1.0, "Every method: 48,000 practice questions of training, 3 tuning tries on dev, "
                                     "3 random seeds, test scored once.", 19, BODY_B, WHITE)
    say(s, "What we compared", 1.0, "Shaurya",
        "We compare against four kinds of methods. The oracle gives the answer model the correct evidence; it shows "
        "the best score possible. The full-table reader is the standard approach: fine-tune on the whole table, no "
        "hops. Two RL methods also hop but are scored only at the end; the second one also gets a path score at the "
        "end, so it has the same hints as ours. Soft selection picks evidence in one shot. Then our method, and four "
        "versions of it with one part removed. Every method gets the same data, the same training budget, three tuning "
        "tries and three seeds.")

    # 9 main results ----------------------------------------------------------------------------
    s = D.slide("white", "Main results", f"How often each method answers exactly right ({split_word} questions)", pin=True)
    D.figure(s, FIG / "F2_main.png", 1.12, 2.95, 9.2, 6.6,
             f"Exact-match accuracy per method; bars = 95% CI ({T}).", cap_color=RED)
    PL = {"b1_sft": "Full-table reader (SFT)", "b2a_grpo": "Answer-only RL", "b2b_grpo_path": "Answer+path RL",
          "b3_soft": "Soft selection", "h1_hoppo": "Hop-PPO (ours)"}
    rows = [["Method", "Exact match", "Path valid"]]
    if R.m0:
        rows.append(["Oracle path", f"{pct(R.m0['em'])}%", f"{100 * R.m0['path_validity_strict']:.0f}%"])
    for m in order:
        if m in R.by:
            v = num(R.cell(m, "strict validity"))
            rows.append([PL[m], f"{pct(R.em(m))}%", "n/a" if v is None else f"{100 * v:.0f}%"])
    D.table(s, 10.9, 3.5, 8.0, rows, [4.0, 2.0, 2.0], f"Mean over seeds ({T}).", size=16,
            highlight=len(rows) - 1, cap_color=RED, center={1, 2})

    def cmp_line(p, name):
        if not p or p["ci"] is None:
            return f"vs {name}: n/a"
        word = "reliably" if reliable(p) else "not reliably"
        return f"vs {name}: {pts(p)}, {word} different"
    D.text(s, 10.9, 7.55, 8.0, 0.5, "Ours minus each method (95% CI)", 19, BODY_B, RED)
    D.text(s, 10.9, 8.15, 8.0, 2.2, [cmp_line(pb1, "full-table reader"), cmp_line(pb2a, "answer-only RL"),
                                    cmp_line(pb2b, "answer+path RL")], 17, BODY, RED, bullets=True)
    say(s, "Main results", 1.5, "Shaurya",
        f"This is the main result, measured as exact match on {split_word} questions. "
        + "; ".join(f"{PL[m]} gets {pct(R.em(m))} percent" for m in order if m in R.by)
        + (f"; the oracle gets {pct(R.m0['em'])} percent, which is the ceiling of the answer model. " if R.m0 else ". ")
        + "The full-table reader is clearly the strongest. Compared with the answer-only RL method our method differs "
        f"by {pts(pb2a)}, which is {'a reliable' if reliable(pb2a) else 'not a reliable'} difference. Compared with the "
        f"answer+path RL method it differs by {pts(pb2b)}. The table on the right also shows path validity: how often "
        "the hops were on the correct path.")

    # 10 learning dynamics ------------------------------------------------------------------------
    s = D.slide("blue", "Learning during training", "What the step-by-step reward teaches the model")
    D.figure(s, FIG / "F3_learning_curves.png", 1.12, 2.95, 8.6, 5.9,
             "Dev accuracy (300 questions) as training goes on.", cap_color=WHITE)
    D.figure(s, FIG / "F10_reward_components.png", 10.3, 2.95, 8.6, 5.9,
             "Our model's reward, split into its parts, during training.", cap_color=WHITE)
    D.text(s, 1.12, 9.55, 17.8, 1.0, [
        "The hop reward rises quickly: the model learns to land on the correct path. Around update 200 it learns to "
        "STOP after about two hops, so the cost per hop shrinks. The answer reward grows slowly and stays far below "
        "the full-table reader."], 18, BODY, WHITE)
    say(s, "Learning during training", 0.75, "Dhruv",
        "Left: accuracy on dev questions during training for each method. Right: what our model is actually being "
        "paid for. The orange line, the reward for correct hops, rises fast, so the model does learn to land on the "
        "correct path. Around update 200 it learns to stop after about two hops, the typical path length, so the "
        "cost per hop shrinks. The answer reward improves only slowly. This hints at our main finding on the next "
        "slide.")

    # 11 ablations ------------------------------------------------------------------------------
    s = D.slide("white", "Which reward parts matter?", "Our model with one part removed at a time", pin=True)
    D.figure(s, FIG / "F5_ablations.png", 1.12, 2.95, 9.2, 6.6, f"Exact-match accuracy of each version ({T}).",
             cap_color=RED)
    AL = {"h1_hoppo": "Hop-PPO (full)", "a1_terminal": "Answer reward only", "a2_noredund": "No repeat penalty",
          "a3_noschema": "No type check", "a4_unrestricted": "Any-cell moves"}
    rows = [["Version", "Exact match", "Avg hops"]]
    for m in AL:
        if m in R.by:
            rows.append([AL[m], f"{pct(R.em(m))}%", R.cell(m, "mean hops").split("±")[0]])
    D.table(s, 10.9, 3.5, 8.0, rows, [4.0, 2.0, 2.0], f"Our model vs its reduced versions ({T}).", size=16,
            highlight=1, cap_color=RED, center={1, 2})
    if reliable(pa1) and pa1["diff"] < 0:
        a1_txt = f"Dropping the step rewards (answer reward only) gives MORE correct answers: ours minus it = {pts(pa1)}."
    elif reliable(pa1):
        a1_txt = f"The step rewards help: ours minus 'answer reward only' = {pts(pa1)}."
    else:
        a1_txt = (f"Step rewards do not reliably help: ours minus 'answer reward only' = {pts(pa1)}"
                  + (f"; seeds disagree ({pa1['seeds']})." if pa1 and pa1.get("seeds") else "."))
    D.text(s, 10.9, 6.85, 8.0, 3.4, [a1_txt,
                                    "Seeds that learn to stop after ~2 hops read less evidence and answer fewer "
                                    "questions; seeds that use all 4 hops answer more.",
                                    "So step rewards mainly make paths shorter, not answers more correct."],
           18, BODY, RED, bullets=True, space_after=8)
    say(s, "Which reward parts matter?", 1.5, "Dhruv",
        "This is the key test of our idea. Each bar is our model with one part removed. The second bar keeps only the "
        f"final-answer reward, no step rewards. {a1_txt} The reason is in the hops column: with step rewards the model "
        "learns to stop after about two hops, the length of the correct path, while without them it keeps going to "
        "four hops and hands the answer model more evidence. So step rewards make the paths shorter and tidier, but "
        "did not make the answers more correct. We report this honestly as a negative result. Removing the repeat "
        "penalty, the type check, or the move limits changes less.")

    # 12 errors ---------------------------------------------------------------------------------
    s = D.slide("blue", "Where it goes wrong", "Following the path, and the first wrong step")
    D.figure(s, FIG / "F4_hop_validity.png", 1.12, 2.95, 8.6, 5.9, "How often each hop is on the correct path.",
             cap_color=WHITE)
    D.figure(s, FIG / "F9_error_taxonomy.png", 10.3, 2.95, 8.6, 5.9, "For wrong answers: where the first mistake was.",
             cap_color=WHITE)
    D.text(s, 1.12, 9.55, 17.8, 1.0, [
        "Most wrong answers start with a wrong FIRST hop. The first cell is chosen mostly by matching words in the "
        "question, and later hops cannot repair it."], 18, BODY, WHITE)
    say(s, "Where it goes wrong", 1.0, "Shaurya",
        "Left: for each hop number, how often the hop was on the correct path. It drops sharply after the first hop "
        "for every method. Right: for every wrong answer, where the first mistake happened. The dark orange part, a "
        "wrong first hop, dominates for every method. The first cell is chosen mostly by matching words from the "
        "question, and later steps cannot repair a wrong start. That tells us where to improve next: the first step.")

    # 13 demo -----------------------------------------------------------------------------------
    s = D.slide("white", "Live demo", "Ask a question, see each hop, the evidence and the answer", pin=True)
    shot = sorted((RES / "demo").glob("webapp_screenshot*.png"))
    D.figure(s, shot[-1] if shot else RES / "demo/webapp_screenshot.png", 1.12, 2.95, 11.4, 6.6,
             "Demo web page (make webapp): picked cells highlighted, one card per hop, evidence and answer.",
             cap_color=RED)
    D.text(s, 13.1, 3.05, 5.8, 0.6, "Questions to try", 22, BODY_B, RED)
    D.text(s, 13.1, 3.7, 5.8, 6.0, [
        "Coach of the gymnast who finished 8th (2001 Worlds) → Igor Korobchinsky (full-table reader says Kevin Harmse)",
        "Same table: city where he died → Kiev",
        "Longest inverted roller coaster → Banshee; its maker → Bolliger & Mabillard",
        "A failure: River Phoenix sitcom network (wrong first hop)"], 18, BODY, RED, bullets=True, space_after=10)
    say(s, "Live demo", 1.75, "Dhruv",
        "Now the live demo. (Open http://127.0.0.1:8769; full click-by-click guide in docs/DEMO_GUIDE.md.) Click the "
        "gymnast question. Point at hop 1, the highlighted row; hop 2, the linked page; then STOP. Show the evidence "
        "box: the answer comes only from these two pieces. Point at the comparison: the full-table reader got it wrong. "
        "Click the follow-up 'which city did he die': a new question on the same table, answered Kiev. Then click one "
        "failure case and say honestly why it fails: a wrong first hop. If the laptop has a problem, use this "
        "screenshot.")

    # 14 findings -------------------------------------------------------------------------------
    s = D.slide("blue", "Key findings", f"What our results support ({T})")
    em_h1, em_b1 = R.em("h1_hoppo"), R.em("b1_sft")
    f1 = (f"Reading the whole table (SFT) is still best at our scale: {pct(em_b1)}% vs {pct(em_h1)}% for our "
          "method; all hop-based methods are far below the oracle" + (f" ({pct(R.m0['em'])}%)." if R.m0 else "."))
    f2 = (f"Step rewards beat the same path signal given only at the end (answer+path RL): {pts(pb2b)}"
          f"{', on every seed' if reliable(pb2b) else ''}; but not answer-only RL ({pts(pb2a)}).")
    if reliable(pa1) and pa1["diff"] < 0:
        f3 = "Step rewards make paths shorter and more on-path, but did not give more correct answers."
    elif reliable(pa1):
        f3 = "Step rewards give more correct answers than the answer-only reward."
    else:
        f3 = "Step rewards mainly make paths shorter; accuracy tracks how much evidence the model reads."
    f4 = "Most errors start at the first hop: a better first step is the lever, not more reward shaping."
    # left: native bar chart of exact match (editable in PowerPoint)
    D.card(s, 1.12, 2.9, 9.0, 7.15)
    cats, vals = [], []
    for name, v in (("Oracle path", R.m0["em"] if R.m0 else None), ("Full-table reader", em_b1),
                    ("Hop-PPO (ours)", em_h1), ("Ours, no step reward", R.em("a1_terminal")),
                    ("Answer-only RL", R.em("b2a_grpo"))):
        if v is not None:
            cats.append(name)
            vals.append(round(100 * v, 1))
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("Exact match (%)", vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(1.4), Inches(3.05), Inches(8.45), Inches(6.2), cd)
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = True
    ch.chart_title.text_frame.text = f"Exact match, % ({T})"
    tp = ch.chart_title.text_frame.paragraphs[0]
    tp.runs[0].font.size, tp.runs[0].font.name = Pt(16), BODY_B
    tp.runs[0].font.color.rgb = RGBColor.from_string(BLACK)
    pl = ch.plots[0]
    pl.gap_width = 45
    pl.has_data_labels = True
    pl.data_labels.number_format, pl.data_labels.number_format_is_linked = '0.0"%"', False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    pl.data_labels.font.size, pl.data_labels.font.name = Pt(15), BODY_B
    ser = pl.series[0]
    for k, name in enumerate(cats):
        pt = ser.points[k]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = RGBColor.from_string(
            RED if name == "Hop-PPO (ours)" else "F2A37E" if name.startswith("Ours,") else
            "9CA3AF" if name == "Oracle path" else "1F6F8B")
    ca, va = ch.category_axis, ch.value_axis
    ca.reverse_order = True
    ca.tick_labels.font.size, ca.tick_labels.font.name = Pt(15), BODY
    va.visible = False
    va.tick_label_position = XL_TICK_LABEL_POSITION.NONE
    va.major_tick_mark = XL_TICK_MARK.NONE
    va.format.line.fill.background()
    va.has_major_gridlines = False
    va.maximum_scale = max(vals) * 1.25 if vals else 1
    va.minimum_scale = 0
    ca.format.line.color.rgb = RGBColor.from_string("D1D5DB")
    D.text(s, 1.4, 9.35, 8.5, 0.6, [[("Figure. ", "b"), ("Red = our method; light orange = our method without step rewards; grey = oracle ceiling.", None)]],
           14, color=BLACK)
    # right: four short findings
    short = ["Reading the whole table (SFT) is still best at our scale.",
             f"Step rewards beat end-of-trace path rewards ({pts(pb2b)}); not better than answer-only RL.",
             f3, "Most errors start at the first hop: improve the first step next."]
    for i, t in enumerate(short):
        D.numbered(s, 10.75, 2.95 + i * 1.55, i + 1, 8.2, t, 19)
    D.text(s, 10.75, 9.35, 8.2, 0.8, "Next: better first-hop finder, larger answer model, more datasets.", 17,
           BODY_B, WHITE)
    say(s, "Key findings", 1.0, "Dhruv",
        "To sum up, four findings. One: " + f1 + " Two: " + f2 + " Three: " + f3 + " Four: " + f4 + " Next we would "
        "build a better first-hop finder and try a larger answer model. Thank you, we are happy to take questions.")

    # 15 references -----------------------------------------------------------------------------
    s = D.slide("white", "References", pin=True)
    refs = [
        "Chen, W. et al. (2020). HybridQA: A dataset of multi-hop QA over tabular and textual data. Findings of EMNLP.",
        "Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.",
        "Schulman, J. et al. (2015). High-dimensional continuous control using generalized advantage estimation.",
        "Shao, Z. et al. (2024). DeepSeekMath (introduces GRPO). arXiv:2402.03300.",
        "Ng, A. Y. et al. (1999). Policy invariance under reward transformations. ICML.",
        "Das, R. et al. (2018). Go for a walk and arrive at the answer (MINERVA). ICLR.",
        "Xiong, W. et al. (2017). DeepPath: RL for knowledge graph reasoning. EMNLP.",
        "Lin, X. V. et al. (2018). Multi-hop knowledge graph reasoning with reward shaping. EMNLP.",
        "Yang et al. (2025). Table-R1: Inference-time scaling for table reasoning. EMNLP.",
        "Herzig, J. et al. (2020). TAPAS. ACL.  ·  Liu, Q. et al. (2022). TAPEX. ICLR.",
        "Models: BAAI/bge-small-en-v1.5 (MIT); google/flan-t5-small (Apache-2.0), Hugging Face model cards.",
        "Code: this project's repository (README; make reproduce-all)."]
    D.text(s, 1.12, 2.3, 17.8, 8.0, refs, 19, BODY, RED, bullets=True, space_after=9)
    say(s, "References", 0.0, "—", "(Do not read out. Leave on screen only if asked.)")

    # 16 thank you ------------------------------------------------------------------------------
    s = D.slide("blue", number=False)
    D.text(s, 1.12, 6.6, 6.0, 0.7, "Questions?", 30, BODY, WHITE)
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(4.6), Inches(6.95), Inches(16.0), Inches(6.95))
    ln.line.color.rgb = RGBColor.from_string(WHITE)
    ln.line.width = Pt(2)
    D.text(s, 16.4, 6.6, 2.5, 0.7, "2026", 30, BODY, WHITE, align=PP_ALIGN.RIGHT)
    D.text(s, 1.12, 7.6, 18, 2.6, "THANK YOU.", 150, TITLE_FONT, WHITE)
    say(s, "Thank you", 0.0, "Both", "Thank you. We are happy to take questions. (Backup slides follow, for questions "
                                     "on what we changed, the reward, the setup and the statistics.)")

    # ------------------------------------------------------------------------------ backup slides
    s = D.slide("white", "Backup: what we fixed", "Changes made on dev questions, before any test score", pin=True)
    rows = [["Change", "What we saw on dev", "Applied to"],
            ["Word-match scores for each move", "trained directly on correct paths, the policy still chose the right "
                                                "first cell only ~20% of the time: it could not see word matches",
             "all methods"],
            ["Answer model reads the whole row", "seeing only single cells, the answer model missed context; with the "
                                                 "row, our dev score rose from 8.3% to 11.4% (oracle unchanged)",
             "all hop-based methods"],
            ["3 tuning tries per method", "fits a laptop; every method gets the same number", "all trained methods"],
            ["No reward for revisiting", "the model could earn reward by looping on a correct cell; fixed and tested",
             "our method and its versions"]]
    D.table(s, 1.12, 3.55, 17.8, rows, [4.4, 9.8, 3.6], "Design changes made during development.", size=16,
            row_h=1.15, cap_color=RED)
    say(s, "Backup: what we fixed", 0.0, "Dhruv", "If asked why numbers changed during the project: these four changes "
        "were all decided on dev questions, applied to every method, before any test score existed.")

    s = D.slide("blue", "Backup: reward details", "Every reward part and the switch that removes it")
    rows = [["Reward part", "Value", "When it is given", "Removed in"],
            ["Correct hop", "+0.2", "hop lands on the next gold-path entity (first visit only)", "answer reward only"],
            ["Repeat penalty", "−0.1", "the model visits something again", "no repeat penalty"],
            ["Cost per hop", "−0.02", "every hop", "answer reward only"],
            ["Type check", "−0.05", "question asks for a number/date but the last cell is not one", "no type check"],
            ["Final answer", "+1.0 × F1", "at STOP or after 4 hops", "never"],
            ["Path score", "+0.2 × path F1", "end of the episode; only for answer+path RL", "—"]]
    D.table(s, 1.12, 3.55, 17.8, rows, [3.4, 2.4, 8.4, 3.6], "Reward parts of equations (2) and (3).", size=17,
            cap_color=WHITE, row_h=0.6, center={1})
    D.text(s, 1.12, 8.1, 17.8, 2.5, ["We log reward-hacking signals every update (repeats, early stops, each reward "
                                     "part).", "An automatic test checks a loop on a correct cell is rewarded only once."],
           19, BODY, WHITE, bullets=True)
    say(s, "Backup: reward details", 0.0, "Dhruv", "Reward weights were fixed before tuning; tuning only changed the "
        "learning rate and the exploration bonus.")

    s = D.slide("white", "Backup: setup and compute", "Hardware, budgets and settings", pin=True)
    rows = [["Item", "Value"],
            ["Hardware", "Apple M4 MacBook Air, 16 GB, CPU + MPS, no GPU"],
            ["Frozen models", "bge-small-en-v1.5 text encoder; flan-t5-small answer model (trained once)"],
            ["Policy", "GRU 256, ~0.3M parameters; Adam 3e-4; PPO clip 0.2, 4 epochs"],
            ["GRPO", "8 tries per question, scored against each other; same clipped update"],
            ["Budget", "48,000 practice episodes (RL) or examples (SFT, soft selection) per run"],
            ["Tuning", "RL: default, learning rate 1e-3, exploration 0.03; others: 3 learning rates"],
            ["Time", "one RL run ~15-30 min; one SFT run ~75 min; whole study ~15 h"]]
    D.table(s, 1.12, 3.55, 17.8, rows, [4.2, 13.6], "Everything runs on one laptop.", size=17, cap_color=RED,
            row_h=0.68)
    say(s, "Backup: setup and compute", 0.0, "Shaurya", "We never claim results for large models; scores are not "
        "comparable to published HybridQA systems that use them.")

    s = D.slide("blue", "Backup: statistics", "How we report uncertainty")
    D.text(s, 1.12, 3.0, 17.8, 6.5, [
        "**Seeds**: 3 runs per method with different random seeds; we show the mean.",
        "**95% CI**: resample the questions 10,000 times to get the range of the score.",
        "**Paired comparison**: compare two methods on the same questions; 'reliable' = the range excludes 0.",
        "**Choosing settings**: best of 3 tries on dev; test is never used for any choice and is scored once.",
        "**Claims**: a claim is 'supported' only if the range excludes 0 and all seeds agree."],
        22, BODY, WHITE, bullets=True, space_after=16)
    say(s, "Backup: statistics", 0.0, "Shaurya", "Use if someone asks whether a difference is real.")

    D.save(out)
    print(f"[slides] wrote {out} ({D.n} slides; results split = {R.split}, seeds = {R.seeds})")
    if script_out:
        write_script(SCRIPT, script_out, T)


def write_script(items: list[dict], out: Path, tag: str) -> None:
    """Talking script (markdown) from the same text as the speaker notes."""
    main = [x for x in items if x["min"] > 0]
    total = sum(x["min"] for x in main)
    lines = ["# Talking script: TableHop-PPO final presentation", "",
             f"Generated with the deck by `scripts/slides/build_final_deck.py`; numbers come from `results/tables` ({tag}).",
             f"Planned speaking time: about {total:.1f} minutes for {len(main)} slides (limit 15 min), then questions.",
             "Backup slides (time 0) are only shown if a question needs them.", "",
             "| Slide | Title | Presenter | Time |", "|---|---|---|---|"]
    lines += [f"| {x['n']} | {x['title']} | {x['who']} | {x['min']:g} min |" for x in items]
    lines.append("")
    for x in items:
        lines += [f"## Slide {x['n']}: {x['title']}", f"*{x['who']} · about {x['min']:g} min*", "", x["text"], ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"[slides] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "deliverables/final_presentation.pptx"))
    ap.add_argument("--script", default=str(ROOT / "deliverables/TALK_SCRIPT.md"))
    a = ap.parse_args()
    build(Path(a.out), Path(a.script) if a.script else None)


if __name__ == "__main__":
    main()
