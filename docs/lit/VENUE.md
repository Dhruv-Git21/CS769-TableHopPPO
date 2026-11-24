# Venue facts: ACL 2027 / ARR (all fetched 2026-11-13)

| Item | Fact | URL |
|---|---|---|
| ACL 2027 ARR deadline (long and short) | Jan 4, 2027, 11:59PM UTC-12 (AoE). ACL 2027 call page lists it. | https://2027.aclweb.org/calls/main/ |
| ARR dates page | Lists ACL 2027 final ARR submission only as "January, 2027" with no day; no Jan 2027 cycle row posted. Treat Jan 4 as the call-page date, ARR dates page not yet confirming. Commitment date: not posted as of 2026-11-13. | https://aclrollingreview.org/dates |
| Latest posted ARR cycle | October 2026: submit Oct 12, reviews due Nov 16, author response Nov 24-30, meta-reviews Dec 17, end Dec 20 (feeds NAACL/COLING 2027 commitment Dec 23, 2026; not ACL 2027 per the page). | https://aclrollingreview.org/dates |
| Conference dates | Main Aug 20-22, 2027; workshops Aug 18-19; tutorials Aug 17. City not listed. | https://2027.aclweb.org/calls/main/ |
| Long paper limit | 8 pages content (9 camera-ready); short 4 (5). Limitations, ethics section and references unlimited; over-length is desk rejected. Appendices after references, not counted, double column. | https://2027.aclweb.org/calls/main/ ; https://aclrollingreview.org/cfp |
| Limitations section | Mandatory, titled "Limitations", at end before references, outside page limit; no new methods/results in it; missing = desk reject. | https://aclrollingreview.org/cfp |
| Responsible NLP checklist | Mandatory, part of submission form; sections A risks/limitations, B artifacts (licenses, stats), C compute/hyperparameters/seeds, D human annotators, E AI-assistant disclosure. Wrong answers can desk-reject. | https://aclrollingreview.org/responsibleNLPresearch/ |
| Ethics section | Recommended, "Ethical considerations", at end, outside page limit. | https://aclrollingreview.org/cfp |
| Anonymity | Double blind; no names/affiliations; third-person self-citation; anonymize supplements/repos; no Dropbox-type links. Non-anonymous preprints allowed during review (since 2024-02-15) unless binding no-preprint option chosen. | https://aclrollingreview.org/cfp |
| Dual submission | Not allowed. | https://2027.aclweb.org/calls/main/ |
| Special theme | Homogenization and knowledge collapse in LLMs (secondary source only, aiweekly.co; not verified on official page). | https://aiweekly.co/node/16101 |
| Template | github.com/acl-org/acl-style-files, commit d5adc823ff0f80f98c80405ca0ab66c68e684409 (2026-06-29). Copied acl.sty, acl_natbib.bst, acl_latex.tex into paper/. Do not modify style files. Use `\usepackage[review]{acl}` for submission. | https://github.com/acl-org/acl-style-files |

## Template compile test (pdflatex, scratch copy, 2026-11-13)
FAILED to produce a PDF on this machine: TeX Live 2026 "basic" scheme lacks Helvetica metrics (`phvb.tfm`, psnfss/helvetic) and the `mwe` example images (`example-image-a/b/golden`) used by acl_latex.tex. Template itself is unmodified. Fix: `sudo tlmgr install helvetic mwe` (not done; system change). main.tex need not use example images, but needs phvb for the `\aclhv` font in the title/section code.
