// Draft final presentation (10 slides). Methodology, baselines and metrics are filled in;
// every result slot is an explicit placeholder. No numbers from experiments appear here.
// Run: node scripts/slides/build_draft_deck.js  -> deliverables/final_presentation_draft.pptx
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const OUT = path.resolve(__dirname, "../../deliverables/final_presentation_draft.pptx");

// Palette from docs/INTERIM_SUMMARY.md
const HEX = {
  navy: "14213D", teal: "1F6F8B", orange: "B4501A", lightOrange: "F2B48A",
  cream: "F6F5F0", card: "FBFAF6", border: "DAD7CC", text: "3D4757",
  muted: "6B7280", band: "ECEAE0", paleTeal: "E4EEF1", lightTeal: "7FB3C4", white: "FFFFFF",
};
const THEME = {
  name: "TableHop",
  headFontFace: "Georgia",
  bodyFontFace: "Arial",
  colors: {
    dk1: HEX.text, lt1: HEX.cream, dk2: HEX.navy, lt2: HEX.card,
    accent1: HEX.orange, accent2: HEX.teal, accent3: HEX.muted, accent4: HEX.navy,
    accent5: HEX.lightTeal, accent6: HEX.lightOrange, hlink: HEX.teal, folHlink: HEX.muted,
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Learning Multi-Hop Table Reasoning through RL";
pres.author = "Dhruv Goyal and Shaurya";
const C = pres.SchemeColor;
const W = 13.333, M = 0.6;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "COVER",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "label", type: "body", x: M, y: 1.95, w: 8, h: 0.4,
      fontSize: 14, bold: true, color: C.accent6, charSpacing: 3, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.45, w: 8.2, h: 2.0,
      fontFace: "Georgia", fontSize: 40, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.65, w: 8.2, h: 1.6,
      fontSize: 20, color: C.background1, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, M, 0.6, M],
  objects: [
    { placeholder: { options: { name: "label", type: "body", x: M, y: 0.4, w: 9, h: 0.35,
      fontSize: 13, bold: true, color: C.accent1, charSpacing: 3, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.75, w: W - 2 * M, h: 0.8,
      fontFace: "Georgia", fontSize: 32, bold: true, color: C.text2, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Final presentation  |  Dhruv Goyal and Shaurya",
      options: { x: M, y: 7.0, w: 7, h: 0.3, fontSize: 11, color: C.accent3, margin: 0 } } },
  ],
});

// ---------- helpers ----------
let pageNo = 1;
const content = (label, title, section) => {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  pageNo += 1;
  s.addText(`${pageNo} of 10`, { x: 11.23, y: 7.0, w: 1.5, h: 0.3, fontSize: 11, color: C.accent3, align: "right", margin: 0, isTextBox: true });
  s.addText(label, { placeholder: "label" });
  s.addText(title, { placeholder: "title" });
  return s;
};
const card = (s, x, y, w, h, fill = C.background2, name = "card") =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: fill }, line: { color: HEX.border, width: 1 }, objectName: name });
const badge = (s, x, y, n, d = 0.5, fill = C.accent1) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: `badge ${n}` });
  s.addText(String(n), { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: d > 0.45 ? 18 : 14,
    bold: true, color: C.background1, margin: 0, isTextBox: true });
};
const text = (s, t, o) => s.addText(t, { isTextBox: true, margin: 0, color: C.text1, fontSize: 18, valign: "top", ...o });
const placeholderFrame = (s, x, y, w, h, head, sub) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 },
    line: { color: HEX.muted, width: 1.25, dashType: "dash" }, objectName: "results placeholder" });
  text(s, [
    { text: "RESULT PENDING", options: { fontSize: 12, bold: true, color: C.accent1, charSpacing: 2, breakLine: true } },
    { text: head, options: { fontSize: 18, bold: true, color: C.text2, breakLine: true } },
    { text: sub, options: { fontSize: 14, color: C.accent3 } },
  ], { x: x + 0.3, y: y + 0.3, w: w - 0.6, h: h - 0.6, valign: "middle", align: "center", paraSpaceAfter: 6 });
};
const bullets = (items, size = 18) => items.map((t, i) => ({ text: t,
  options: { bullet: true, breakLine: i < items.length - 1, fontSize: size, paraSpaceAfter: 8 } }));

// ---------- 1. Cover ----------
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Opening" });
  s.addText("CS769 FINAL PRESENTATION", { placeholder: "label" });
  s.addText("Learning Multi-Hop Table Reasoning through RL", { placeholder: "title" });
  s.addText([
    { text: "PPO-based entity selection with hop-level credit", options: { breakLine: true } },
    { text: "Dhruv Goyal and Shaurya", options: { bold: true, color: C.accent6 } },
  ], { placeholder: "body" });
  // motif: a three-hop path
  const xs = [9.6, 10.9, 12.2], ys = [2.3, 3.6, 4.9];
  for (let i = 0; i < 2; i++) s.addShape(pres.shapes.LINE, { x: xs[i] + 0.3, y: ys[i] + 0.3, w: 1.3, h: 1.3,
    line: { color: HEX.lightOrange, width: 2.5, endArrowType: "triangle" }, objectName: "hop arrow" });
  xs.forEach((x, i) => badge(s, x, ys[i], i + 1, 0.6));
  s.addNotes("Presenters: Dhruv and Shaurya. Introduce the team and the one-line idea: treat each hop of multi-hop table QA as a discrete action and give it its own reward, then train with PPO. This deck is a draft: the method, baselines and metrics are final; result slides are placeholders until the final runs finish.");
}

// ---------- 2. Problem ----------
pres.addSection({ title: "Problem" });
{
  const s = content("PROBLEM AND MOTIVATION", "Multi-hop table QA is a sequence of choices", "Problem");
  text(s, "“In which city is the team of the top scorer based?”", { x: M, y: 1.75, w: 7.4, h: 0.45, fontSize: 18, italic: true, color: C.text2 });
  const rows1 = [["Player", "Goals", "Team"], ["Maria Lopez", "21", "Harbor FC"], ["Ana Silva", "14", "Ridge United"]];
  const rows2 = [["Team", "City"], ["Harbor FC", "Lisbon"], ["Ridge United", "Porto"]];
  const hi = { fill: { color: HEX.lightOrange }, bold: true, color: HEX.navy };
  const mk = (rows, hits) => rows.map((r, i) => r.map((c, j) => ({ text: c, options: i === 0
    ? { bold: true, color: HEX.white, fill: { color: HEX.teal } }
    : hits.some(([a, b]) => a === i && b === j) ? hi : { fill: { color: i % 2 ? HEX.card : HEX.band } } })));
  s.addTable(mk(rows1, [[1, 1], [1, 2]]), { x: M, y: 2.45, w: 4.3, colW: [1.7, 0.9, 1.7], fontSize: 15, rowH: 0.42,
    border: { type: "solid", color: HEX.border, pt: 1 }, color: HEX.text, objectName: "Players table" });
  s.addTable(mk(rows2, [[1, 1]]), { x: 5.3, y: 2.45, w: 2.7, colW: [1.6, 1.1], fontSize: 15, rowH: 0.42,
    border: { type: "solid", color: HEX.border, pt: 1 }, color: HEX.text, objectName: "Teams table" });
  badge(s, 2.0, 3.92, 1, 0.42); badge(s, 3.75, 3.92, 2, 0.42); badge(s, 6.85, 3.92, 3, 0.42);
  text(s, [
    { text: "1 ", options: { bold: true, color: C.accent1 } }, { text: "pick the row with most goals   ", options: {} },
    { text: "2 ", options: { bold: true, color: C.accent1 } }, { text: "read its team   ", options: {} },
    { text: "3 ", options: { bold: true, color: C.accent1 } }, { text: "look up that team’s city", options: {} },
  ], { x: M, y: 4.55, w: 7.4, h: 0.4, fontSize: 15 });
  text(s, "Illustrative example, not from any dataset.", { x: M, y: 6.45, w: 7, h: 0.3, fontSize: 12, italic: true, color: C.accent3 });

  // right: reward comparison
  card(s, 8.5, 1.75, 4.23, 4.95);
  text(s, "Where does the reward arrive?", { x: 8.8, y: 1.95, w: 3.7, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  const rowR = (y, label, marks, colorOn) => {
    text(s, label, { x: 8.8, y, w: 3.7, h: 0.35, fontSize: 15, bold: true, color: colorOn });
    ["hop 1", "hop 2", "answer"].forEach((h, i) => {
      const on = marks[i];
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.8 + i * 1.25, y: y + 0.45, w: 1.1, h: 0.75, rectRadius: 0.06,
        fill: { color: on ? colorOn : C.background1 }, line: { color: on ? colorOn : HEX.border, width: 1 }, objectName: `${label} ${h}` });
      text(s, [{ text: on ? "reward" : "no signal", options: { breakLine: true, bold: on } }, { text: h, options: { fontSize: 11 } }],
        { x: 8.8 + i * 1.25, y: y + 0.45, w: 1.1, h: 0.75, fontSize: 13, align: "center", valign: "middle", color: on ? C.background1 : C.accent3 });
    });
  };
  rowR(2.55, "Trace-level RL", [false, false, true], HEX.teal);
  rowR(4.0, "Hop-level PPO (ours)", [true, true, true], HEX.orange);
  text(s, "A trace-level reward cannot say which hop went wrong.", { x: 8.8, y: 5.6, w: 3.7, h: 0.9, fontSize: 15, italic: true, color: C.text2 });
  s.addNotes("Presenter: Dhruv. Walk through the three hops on the toy tables; the example is made up and is labelled as such. The point: if the answer is wrong, a trace-level reward gives the same zero to a policy that failed at hop 1 and one that failed at hop 3. Hop-level PPO gives a signal after each hop, which is the credit-assignment idea we test.");
}

// ---------- 3. Related work ----------
{
  const s = content("RELATED WORK AND GAP", "Prior RL work rewards whole traces, or works on graphs", "Problem");
  const papers = [
    ["Table-R1", "Yang, Chen, Cohan, Zhao · EMNLP 2025", "RL with verifiable rewards for table reasoning. The reward scores the whole trace.", "TABLES · TRACE-LEVEL CREDIT", HEX.teal],
    ["MA-TPath", "Bai, Chen, Xiao · Applied Soft Computing 2024", "Multi-agent RL for multi-hop reasoning over temporal knowledge graphs.", "GRAPHS · HOP-BY-HOP PATHS", HEX.navy],
    ["MASKGR", "Liu, Fu, Fu, Wu, Bai · Expert Syst. Appl. 2025", "Deep RL for multi-hop reasoning over sparse knowledge graphs.", "GRAPHS · HOP-BY-HOP PATHS", HEX.navy],
  ];
  const cw = 3.85, gap = 0.3;
  papers.forEach(([name, who, what, tag, col], i) => {
    const x = M + i * (cw + gap);
    card(s, x, 1.8, cw, 3.25, C.background2, `paper ${name}`);
    text(s, tag, { x: x + 0.3, y: 2.05, w: cw - 0.6, h: 0.3, fontSize: 11, bold: true, color: col, charSpacing: 1 });
    text(s, name, { x: x + 0.3, y: 2.4, w: cw - 0.6, h: 0.5, fontFace: "Georgia", fontSize: 24, bold: true, color: C.text2 });
    text(s, who, { x: x + 0.3, y: 2.95, w: cw - 0.6, h: 0.5, fontSize: 12, color: C.accent3 });
    text(s, what, { x: x + 0.3, y: 3.55, w: cw - 0.6, h: 1.4, fontSize: 16 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.35, w: W - 2 * M, h: 1.35, rectRadius: 0.08,
    fill: { color: HEX.navy }, line: { color: HEX.navy }, objectName: "gap statement" });
  text(s, [
    { text: "GAP  ", options: { bold: true, color: C.accent6, charSpacing: 2, fontSize: 14 } },
    { text: "We found no prior work that trains hop-level entity selection on tables with PPO and gives credit at every hop. Graph agents do not transfer unchanged: tables have rows, columns and linked text, not triples.", options: { color: C.background1 } },
  ], { x: M + 0.35, y: 5.35, w: W - 2 * M - 0.7, h: 1.35, fontSize: 17, valign: "middle" });
  s.addNotes("Presenter: Shaurya. Table-R1 (EMNLP 2025, verified on ACL Anthology) shows RL with verifiable rewards works for tables but rewards the whole trace; it reports no HybridQA or MultiHiertt numbers, so we cannot reproduce it on our data. MA-TPath and MASKGR build multi-hop paths with RL on knowledge graphs; we verified their bibliographic details, but we could not access their full text, so we do not describe their reward designs in detail. Our position: bring hop-level credit to tables.");
}

// ---------- 4. Method ----------
pres.addSection({ title: "Method" });
{
  const s = content("METHOD", "Each hop is a discrete, credit-assignable action", "Method");
  const steps = [
    ["State", "question, visited entities, reachable candidates"],
    ["Policy (PPO)", "frozen bge-small encoder + trainable scorer; masked softmax"],
    ["Action", "next cell, row or linked passage, or STOP (at most 4 hops)"],
    ["Environment", "moves to the entity and checks the hop"],
    ["Answer head", "frozen flan-t5-small reads the selected entities"],
  ];
  const bw = 2.2, bg = 0.28, y = 1.8, bh = 2.25;
  steps.forEach(([h, d], i) => {
    const x = M + i * (bw + bg);
    card(s, x, y, bw, bh, i === 1 ? HEX.paleTeal : C.background2, `step ${h}`);
    badge(s, x + 0.2, y + 0.2, i + 1, 0.42, i === 4 ? HEX.teal : HEX.orange);
    text(s, h, { x: x + 0.2, y: y + 0.72, w: bw - 0.4, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
    text(s, d, { x: x + 0.2, y: y + 1.12, w: bw - 0.35, h: 1.05, fontSize: 13 });
    if (i < 4) s.addShape(pres.shapes.LINE, { x: x + bw + 0.03, y: y + bh / 2, w: bg - 0.06, h: 0,
      line: { color: HEX.muted, width: 1.5, endArrowType: "triangle" }, objectName: "flow arrow" });
  });
  text(s, "loop until STOP: next state + hop reward feed back to the policy", { x: M, y: 4.12, w: 9.6, h: 0.35, fontSize: 13, italic: true, color: C.accent3 });

  card(s, M, 4.6, W - 2 * M, 2.1, C.background2, "reward");
  text(s, "Hybrid reward (default weights; tuned on dev only)", { x: M + 0.3, y: 4.75, w: 8, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
  const terms = [
    ["+0.2", "hop lands on the gold path", HEX.orange],
    ["−0.1", "revisits an entity", HEX.teal],
    ["−0.05", "wrong type for the answer", HEX.teal],
    ["−0.02", "per step taken", HEX.teal],
    ["+1.0 × F1", "answer at STOP", HEX.navy],
  ];
  const tw = (W - 2 * M - 0.6) / 5;
  terms.forEach(([v, d, col], i) => {
    const x = M + 0.3 + i * tw;
    text(s, v, { x, y: 5.22, w: tw - 0.15, h: 0.6, fontFace: "Arial", fontSize: 26, bold: true, color: col });
    text(s, d, { x, y: 5.85, w: tw - 0.2, h: 0.7, fontSize: 14 });
  });
  s.addNotes("Presenter: Dhruv. Formulation: a POMDP over table entities. The state is the question plus the partial trace and the candidates reachable from the last entity (same row, same column, header, linked passage, cross-references). The policy scores candidates with a frozen sentence encoder and a small trainable scorer; illegal actions are masked out. On STOP, a frozen answer head shared by all path-based methods writes the answer. Reward: four per-hop terms plus terminal answer F1; weights shown are the starting defaults, tuned on dev with the same trial budget as every baseline. Hops on examples without a derivable gold path get no hop term.");
}

// ---------- 5. Setup ----------
pres.addSection({ title: "Setup" });
{
  const s = content("EXPERIMENTAL SETUP", "A controlled setup: only the training signal changes", "Setup");
  card(s, M, 1.8, 5.9, 4.9, C.background2, "data");
  text(s, "Data: HybridQA", { x: M + 0.3, y: 2.0, w: 5.3, h: 0.45, fontSize: 20, bold: true, color: C.text2 });
  text(s, "Multi-hop questions over Wikipedia tables and linked passages (Chen et al., 2020). Official: 62,682 train / 3,466 dev / 3,463 test questions.",
    { x: M + 0.3, y: 2.5, w: 5.3, h: 1.0, fontSize: 15 });
  text(s, "Planned splits", { x: M + 0.3, y: 3.6, w: 5.3, h: 0.35, fontSize: 15, bold: true, color: C.accent1 });
  text(s, bullets([
    "train: 3,000-question subset",
    "dev: 1,000 questions, unseen tables (all tuning)",
    "ID test: official dev, unseen tables",
    "OOD test: 3 held-out table clusters",
    "official test labels never used",
  ], 15), { x: M + 0.3, y: 3.95, w: 5.3, h: 2.6 });

  const items = [
    ["Same encoder and answer head", "for every path-based method"],
    ["Matched compute", "equal training rollouts; GRPO’s 8 samples count 8×"],
    ["Matched tuning", "6 trials per method, dev only"],
    ["3 seeds per final config", "test split evaluated once"],
  ];
  items.forEach(([h, d], i) => {
    const y = 1.85 + i * 1.0;
    badge(s, 6.85, y + 0.05, "✓", 0.48, HEX.teal);
    text(s, h, { x: 7.55, y, w: 5.2, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
    text(s, d, { x: 7.55, y: y + 0.4, w: 5.2, h: 0.4, fontSize: 15, color: C.accent3 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.85, y: 5.95, w: 5.88, h: 0.75, rectRadius: 0.08,
    fill: { color: HEX.paleTeal }, line: { color: HEX.paleTeal }, objectName: "compute note" });
  text(s, "Compute: one Apple M4 laptop, small frozen encoder. We make no 7B-scale claims.",
    { x: 7.05, y: 5.95, w: 5.5, h: 0.75, fontSize: 14, valign: "middle", color: C.text2 });
  s.addNotes("Presenter: Shaurya. HybridQA split sizes are counted from the official files. Our splits keep tables disjoint: train and dev come from the official train set, ID and OOD test from the official dev set; OOD holds out three whole clusters of table schemas (k-means on table title and header embeddings). The official test labels are never opened because they have no answer-node annotations, which our path metrics need. Fairness: one encoder, one frozen answer head, the same rollout budget and the same number of tuning trials for every method. Compute is a laptop, and the paper says so.");
}

// ---------- 6. Baselines ----------
{
  const s = content("BASELINES", "Each baseline rules out one alternative explanation", "Setup");
  const head = ["", "Method", "Training signal", "Question it answers"].map((t) => ({ text: t,
    options: { bold: true, color: HEX.white, fill: { color: HEX.navy } } }));
  const rows = [
    ["M0", "Oracle path", "gold evidence → answer head", "What is the ceiling for path-based methods?"],
    ["B1", "SFT", "question + linearized table → answer", "Is RL needed at all?"],
    ["B2a", "Trace-level GRPO", "answer F1 at end of trace", "Is per-hop credit better than per-trace credit?"],
    ["B2b", "Trace GRPO + path F1", "answer F1 + path F1 at end of trace", "Key control: is the gain just extra supervision?"],
    ["B3", "Soft region attention", "answer loss, no discrete actions", "Do discrete hops matter?"],
    ["H1", "Hop-PPO (ours)", "per-hop + terminal reward", "—"],
  ];
  const body = rows.map((r, i) => r.map((c, j) => ({ text: c, options: {
    bold: j < 2 || (i === 3 && j === 3),
    color: i === 5 ? HEX.white : (j === 0 ? HEX.orange : HEX.text),
    fill: { color: i === 5 ? HEX.orange : i === 3 ? HEX.paleTeal : i % 2 ? HEX.card : HEX.band } } })));
  s.addTable([head, ...body], { x: M, y: 1.8, w: W - 2 * M, colW: [0.75, 2.85, 4.0, 4.533], fontSize: 15, rowH: 0.48,
    border: { type: "solid", color: HEX.border, pt: 1 }, valign: "middle", margin: [0, 0.1, 0, 0.1], objectName: "baseline table" });
  text(s, "Ablations of H1", { x: M, y: 5.35, w: 4, h: 0.35, fontSize: 15, bold: true, color: C.accent1 });
  const abl = [["A1", "terminal reward only"], ["A2", "no revisit penalty"], ["A3", "no type term"], ["A4", "unrestricted actions"]];
  const aw = (W - 2 * M - 0.9) / 4;
  abl.forEach(([id, d], i) => {
    const x = M + i * (aw + 0.3);
    card(s, x, 5.8, aw, 0.85, i === 0 ? HEX.paleTeal : C.background2, `ablation ${id}`);
    text(s, [{ text: id + "  ", options: { bold: true, color: C.accent1 } }, { text: d, options: {} }],
      { x: x + 0.2, y: 5.8, w: aw - 0.4, h: 0.85, fontSize: 15, valign: "middle" });
  });
  s.addNotes("Presenter: Shaurya. Every row uses the same data, encoder and frozen answer head. B2a is Table-R1-style trace-level credit inside our controlled setup. B2b is the supervision control: it sees the same gold-path information as Hop-PPO, but only as one score at the end of the trace, so if H1 beats B2b the gain comes from where the credit lands, not from extra labels. A1 removes all per-hop terms: if A1 matches H1, the discrete action space, not the dense reward, explains any gain. KG dual-agent and text-generation GRPO baselines were cut for compute.");
}

// ---------- 7. Metrics ----------
{
  const s = content("METRICS", "We measure answers, paths and transfer", "Setup");
  const cols = [
    ["Answers", "Exact match and token F1", "HybridQA-style normalization; F1 = max over gold answers.", "The end task every method must solve.", HEX.navy],
    ["Paths", "Path validity, gold agreement, hop efficiency", "Validity (strict): legal, not a revisit, on the gold path. Agreement: set F1 and ordered LCS. Efficiency: gold hops / hops taken.", "Credit assignment should show up in the path first (H1, H2).", HEX.orange],
    ["Transfer", "ID → OOD gap", "EM on in-domain test minus EM on held-out schemas, absolute and relative.", "Tests whether a traversal strategy transfers to unseen tables (H3).", HEX.teal],
  ];
  const cw = 3.85, gap = 0.3;
  cols.forEach(([h, m, def, why, col], i) => {
    const x = M + i * (cw + gap);
    card(s, x, 1.8, cw, 3.95, C.background2, `metric ${h}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 2.05, w: 0.42, h: 0.42, fill: { color: col }, line: { color: col }, objectName: `${h} dot` });
    text(s, h, { x: x + 0.85, y: 2.05, w: cw - 1.1, h: 0.42, fontFace: "Georgia", fontSize: 22, bold: true, color: C.text2, valign: "middle" });
    text(s, m, { x: x + 0.3, y: 2.65, w: cw - 0.6, h: 0.65, fontSize: 16, bold: true, color: col });
    text(s, def, { x: x + 0.3, y: 3.3, w: cw - 0.6, h: 1.45, fontSize: 13 });
    text(s, [{ text: "Why: ", options: { bold: true, color: C.text2 } }, { text: why, options: {} }],
      { x: x + 0.3, y: 4.8, w: cw - 0.6, h: 0.85, fontSize: 13 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.95, w: W - 2 * M, h: 0.75, rectRadius: 0.08,
    fill: { color: HEX.paleTeal }, line: { color: HEX.paleTeal }, objectName: "statistics" });
  text(s, [
    { text: "Statistics  ", options: { bold: true, color: C.text2 } },
    { text: "3 seeds; 95% bootstrap CIs (10k resamples); paired bootstrap tests of H1 vs each baseline. HybridQA gold paths are heuristic, so path metrics are reported with coverage." },
  ], { x: M + 0.3, y: 5.95, w: W - 2 * M - 0.6, h: 0.75, fontSize: 14, valign: "middle" });
  s.addNotes("Presenter: Shaurya. Answer metrics follow the HybridQA normalization. Path metrics are computed only on examples with a derivable gold path; HybridQA has no ordered path annotation, so we derive paths from the answer-node labels plus row anchors, mark them heuristic, and report coverage. Strict path validity is the metric used for claims. Hop efficiency is set to 0 when the answer is wrong. The OOD gap compares in-domain test with held-out schema clusters. We only say 'significant' when a paired bootstrap test supports it.");
}

// ---------- 8. Main results (placeholder) ----------
pres.addSection({ title: "Results" });
{
  const s = content("RESULTS", "Main in-domain comparison", "Results");
  placeholderFrame(s, M, 1.8, 6.0, 4.9, "F2: EM, F1 and strict path validity per method",
    "bars with 95% CIs over 3 seeds; generated from results/ after the final runs");
  const methods = ["M0 Oracle", "B1 SFT", "B2a Trace GRPO", "B2b GRPO + path F1", "B3 Soft region", "H1 Hop-PPO (ours)"];
  const head = ["Method", "EM", "F1", "Validity", "Hop eff."].map((t) => ({ text: t,
    options: { bold: true, color: HEX.white, fill: { color: HEX.navy } } }));
  const body = methods.map((m, i) => [m, "—", "—", "—", "—"].map((c, j) => ({ text: c, options: {
    bold: i === 5 && j === 0, align: j ? "center" : "left", color: j ? HEX.muted : HEX.text,
    fill: { color: i === 5 ? HEX.lightOrange : i % 2 ? HEX.card : HEX.band } } })));
  s.addTable([head, ...body], { x: 6.9, y: 1.8, w: 5.83, colW: [2.13, 0.8, 0.8, 1.1, 1.0], fontSize: 14, rowH: 0.5,
    border: { type: "solid", color: HEX.border, pt: 1 }, valign: "middle", margin: [0, 0.08, 0, 0.08], objectName: "results table placeholder" });
  text(s, "ID test, mean over 3 seeds with 95% CI. Cells stay empty until scripts fill them from results/.",
    { x: 6.9, y: 5.5, w: 5.83, h: 0.7, fontSize: 13, italic: true, color: C.accent3 });
  s.addNotes("Presenter: Shaurya. PLACEHOLDER SLIDE. After the final runs, scripts will insert figure F2 and fill this table from results/tables. What to look for: H1 versus B2a and B2b on EM/F1 and strict path validity, with M0 as the ceiling. If H1 does not beat the baselines, we say so and present the study as an analysis of when hop-level credit helps.");
}

// ---------- 9. Analysis (placeholder) ----------
{
  const s = content("ANALYSIS", "Credit assignment, ablations and transfer", "Results");
  const panels = [
    ["F3 / F4", "Learning curves and per-hop validity", "Supports H1 if: H1 keeps later hops valid more often than B2a and B2b."],
    ["F5", "Reward ablations", "Supports H1c if: H1 beats A1 (terminal only) on path validity or hop efficiency."],
    ["F6", "ID vs OOD gap", "Supports H3 if: H1’s drop on held-out schemas is smaller than B1’s and B2a’s."],
  ];
  const cw = 3.85, gap = 0.3;
  panels.forEach(([f, h, crit], i) => {
    const x = M + i * (cw + gap);
    placeholderFrame(s, x, 1.8, cw, 3.3, `${f}: ${h}`, "generated from results/");
    card(s, x, 5.3, cw, 1.4, C.background2, `criterion ${f}`);
    text(s, crit, { x: x + 0.25, y: 5.3, w: cw - 0.5, h: 1.4, fontSize: 14, valign: "middle" });
  });
  s.addNotes("Presenter: Dhruv. PLACEHOLDER SLIDE. Each panel will show one figure from results/figures. The cards below state, before seeing results, what outcome would support each hypothesis, so the reading cannot be adjusted after the fact. We will also report reward-hacking checks: immediate STOP, loops among a few cells, or hop reward rising while answer F1 does not.");
}

// ---------- 10. Conclusion ----------
pres.addSection({ title: "Close" });
{
  const s = content("HYPOTHESES AND NEXT STEPS", "What we will be able to claim", "Close");
  const head = ["Hypothesis", "Test", "Status"].map((t) => ({ text: t, options: { bold: true, color: HEX.white, fill: { color: HEX.navy } } }));
  const hyps = [
    ["H1  Hop credit limits error propagation", "vs B2a, B2b"],
    ["H1b  Not just extra supervision", "vs B2b"],
    ["H1c  The dense reward matters", "vs A1"],
    ["H2  Paths are valid and auditable", "strict validity"],
    ["H3  Smaller ID → OOD gap", "vs B1, B2a"],
  ];
  const body = hyps.map((r, i) => [...r, "Pending"].map((c, j) => ({ text: c, options: {
    bold: j === 0, color: j === 2 ? HEX.orange : HEX.text, italic: j === 2,
    fill: { color: i % 2 ? HEX.card : HEX.band } } })));
  s.addTable([head, ...body], { x: M, y: 1.8, w: 7.3, colW: [4.45, 1.65, 1.2], fontSize: 15, rowH: 0.52,
    border: { type: "solid", color: HEX.border, pt: 1 }, valign: "middle", margin: [0, 0.1, 0, 0.1], objectName: "hypothesis table" });

  card(s, 8.2, 1.8, 4.53, 2.15, C.background2, "limitations");
  text(s, "Limitations", { x: 8.45, y: 1.95, w: 4, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  text(s, bullets(["Small frozen encoder on a laptop", "Heuristic HybridQA gold paths", "One dataset in the core study"], 15),
    { x: 8.45, y: 2.4, w: 4.1, h: 1.7 });
  card(s, 8.2, 4.15, 4.53, 2.15, HEX.paleTeal, "next steps");
  text(s, "Next steps", { x: 8.45, y: 4.3, w: 4, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  text(s, bullets(["Final runs, 3 seeds; test once", "Fill result slides from results/", "Target: ACL 2027 via ARR"], 15),
    { x: 8.45, y: 4.75, w: 4.1, h: 1.5 });
  text(s, "Every hypothesis is reported as supported, partially supported, not supported or inconclusive.",
    { x: M, y: 5.15, w: 7.3, h: 0.8, fontSize: 14, italic: true, color: C.accent3 });
  s.addNotes("Presenters: Dhruv, then Shaurya for questions. Status is Pending for every hypothesis until the final runs finish; statuses come from docs/CLAIMS.md, with supported meaning a 95% CI that excludes zero and all three seeds agreeing in sign. Limitations: laptop-scale model, heuristic gold paths, HybridQA only (MultiHiertt is cut unless time allows). Next: final runs and the paper; the ACL 2027 call page lists an ARR deadline of January 4, 2027.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
