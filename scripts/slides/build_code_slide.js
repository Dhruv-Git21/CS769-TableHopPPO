// One standalone slide: how the code is organized and its key components.
// Run: node scripts/slides/build_code_slide.js -> deliverables/code_structure_slide.pptx
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const OUT = path.resolve(__dirname, "../../deliverables/code_structure_slide.pptx");
const HEX = {
  navy: "14213D", teal: "1F6F8B", orange: "B4501A", lightOrange: "F2B48A", cream: "F6F5F0",
  card: "FBFAF6", border: "DAD7CC", text: "3D4757", muted: "6B7280", band: "ECEAE0",
  paleTeal: "E4EEF1", lightTeal: "7FB3C4", white: "FFFFFF",
};
const THEME = {
  name: "TableHop", headFontFace: "Georgia", bodyFontFace: "Arial",
  colors: { dk1: HEX.text, lt1: HEX.cream, dk2: HEX.navy, lt2: HEX.card, accent1: HEX.orange, accent2: HEX.teal,
    accent3: HEX.muted, accent4: HEX.navy, accent5: HEX.lightTeal, accent6: HEX.lightOrange, hlink: HEX.teal, folHlink: HEX.muted },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "TableHop-PPO code structure";
pres.author = "Dhruv Goyal and Shaurya";
const C = pres.SchemeColor;
const W = 13.333, M = 0.6, MONO = "Courier New";

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "label", type: "body", x: M, y: 0.4, w: 9, h: 0.35,
      fontSize: 13, bold: true, color: C.accent1, charSpacing: 3, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.75, w: W - 2 * M, h: 0.8,
      fontFace: "Georgia", fontSize: 32, bold: true, color: C.text2, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Final presentation  |  Dhruv Goyal and Shaurya", options: { x: M, y: 7.0, w: 7, h: 0.3, fontSize: 11, color: C.accent3, margin: 0 } } },
  ],
});

const text = (s, t, o) => s.addText(t, { isTextBox: true, margin: 0, color: C.text1, fontSize: 14, valign: "top", ...o });
const card = (s, x, y, w, h, fill, name) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
  fill: { color: fill }, line: { color: HEX.border, width: 1 }, objectName: name });
const chip = (s, x, y, built) => {
  const col = built ? HEX.teal : HEX.muted;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 0.85, h: 0.26, rectRadius: 0.13,
    fill: { color: built ? HEX.teal : C.background1 }, line: { color: col, width: 1 }, objectName: built ? "built chip" : "planned chip" });
  text(s, built ? "BUILT" : "PLANNED", { x, y, w: 0.85, h: 0.26, fontSize: 9, bold: true, charSpacing: 1, align: "center", valign: "middle",
    color: built ? HEX.white : HEX.muted });
};
const lines = (arr) => arr.map((t, i) => ({ text: t, options: { breakLine: i < arr.length - 1, paraSpaceAfter: 3 } }));

const s = pres.addSlide({ masterName: "CONTENT" });
s.addText("IMPLEMENTATION", { placeholder: "label" });
s.addText("How the code is organized", { placeholder: "title" });

// Row 1: the pipeline, left to right
text(s, "src/tablehop/  —  one question flows left to right", { x: M, y: 1.65, w: 8, h: 0.3, fontSize: 13, italic: true, color: C.accent3 });
const flow = [
  ["data/", "Load and split", ["HybridQA → unified schema", "gold paths (heuristic)", "table-disjoint + OOD splits", "synthetic tables for tests"], HEX.navy],
  ["env/", "Table environment", ["candidates: row, column, link, cross-ref", "reset / step / STOP", "legal-action mask"], HEX.navy],
  ["models/", "Encoder and heads", ["frozen bge-small + cache", "policy: trace + candidate scorer", "value head; frozen flan-t5 answer head"], HEX.orange],
  ["rl/", "Training", ["reward.py: hybrid + ablation switches", "ppo.py: GAE, clipped loss", "grpo.py: trace-level baselines"], HEX.orange],
  ["eval/", "Metrics", ["EM / F1, path validity", "hop efficiency, gold agreement", "bootstrap CIs, paired tests"], HEX.teal],
];
const gap = 0.2, bw = (W - 2 * M - 4 * gap) / 5, y1 = 2.05, bh = 2.75;
flow.forEach(([dir, role, items, col], i) => {
  const x = M + i * (bw + gap);
  card(s, x, y1, bw, bh, C.background2, `module ${dir}`);
  text(s, dir, { x: x + 0.18, y: y1 + 0.18, w: bw - 0.36, h: 0.4, fontFace: MONO, fontSize: 19, bold: true, color: col });
  text(s, role, { x: x + 0.18, y: y1 + 0.6, w: bw - 0.36, h: 0.3, fontSize: 14, bold: true, color: C.text2 });
  text(s, lines(items), { x: x + 0.18, y: y1 + 0.95, w: bw - 0.3, h: 1.35, fontSize: 12 });
  chip(s, x + 0.18, y1 + bh - 0.42, false);
  if (i < flow.length - 1) s.addShape(pres.shapes.LINE, { x: x + bw + 0.02, y: y1 + 0.38, w: gap - 0.04, h: 0,
    line: { color: HEX.muted, width: 1.5, endArrowType: "triangle" }, objectName: "flow arrow" });
});

// Row 2: shared pieces
const y2 = 5.05, h2 = 1.7;
const bwide = (W - 2 * M - 2 * 0.3) / 3;
const row2 = [
  ["baselines/", "Baselines on the same parts", ["SFT (B1); soft region attention (B3)", "GRPO variants (B2a, B2b) reuse rl/", "same env, encoder, answer head"], false, HEX.paleTeal],
  ["utils/", "Shared infrastructure", ["YAML configs with inheritance", "seeding, device (MPS / CPU)", "run metadata: seed, git hash, config"], true, C.background2],
  ["scripts/ tests/", "Entry points and outputs", ["train.py, evaluate.py, audit_leakage.py", "make_tables / make_figures → paper, slides", "results/runs/<id>/: meta, metrics, predictions"], false, C.background2],
];
row2.forEach(([dir, role, items, built, fill], i) => {
  const x = M + i * (bwide + 0.3);
  card(s, x, y2, bwide, h2, fill, `module ${dir}`);
  text(s, dir, { x: x + 0.2, y: y2 + 0.15, w: bwide - 1.3, h: 0.35, fontFace: MONO, fontSize: 16, bold: true, color: C.text2 });
  chip(s, x + bwide - 1.05, y2 + 0.19, built);
  text(s, role, { x: x + 0.2, y: y2 + 0.52, w: bwide - 0.4, h: 0.3, fontSize: 14, bold: true, color: C.text2 });
  text(s, lines(items), { x: x + 0.2, y: y2 + 0.87, w: bwide - 0.4, h: 0.95, fontSize: 12 });
});

s.addNotes("Presenter: Shaurya. The package mirrors the method. data/ turns HybridQA into one schema with heuristic gold paths and table-disjoint splits; env/ builds the reachable candidates at each hop and masks illegal moves; models/ holds the frozen sentence encoder, the policy and value heads, and the frozen answer head; rl/ holds the reward (with switches for every ablation) and the PPO and GRPO trainers; eval/ computes all metrics with bootstrap confidence intervals. Baselines plug into the same environment, encoder and answer head, which is what keeps the comparison fair. Every run writes its seed, config and git hash, and every number in the paper and slides is generated from results/ by scripts. Status: shared utilities are built and tested; the other modules are specified and being implemented.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
