# scripts/slides/

| File | Purpose |
|---|---|
| `build_final_deck.py` | Builds `deliverables/final_presentation.pptx` (`make slides`). It reads every number from `results/tables/*.csv` and every figure from `results/figures/`. |
| `template/` | The team template reduced to its theme (`theme_template.pptx`: master, fonts, 20 x 11.25 in), the paper backgrounds (`bg_white.jpg`, `bg_blue.jpg`) and the pin graphic |
| `build_draft_deck.js`, `build_code_slide.js`, `apply_theme.js` | Earlier draft decks (pptxgenjs); `npm install` first if you want to rebuild them |

Fonts used by the template are Montserrat (titles) and Poppins (body), both free on Google Fonts. Install them on the presenting machine.
