# scripts/dev: history, not needed to reproduce

One-off drivers used while building the project (kept for the record of what was actually run):
- `stage1.sh`, `g4_seed0.sh`: first end-to-end pass and gate G4 (not included in the release; machine-specific paths).
- `run_all.sh`: early pipeline draft (before tuning and the row-evidence answer head). Superseded by `scripts/reproduce_all.sh`.
- `inspect_hybridqa.py`: raw-data inspection used to write docs/lit/DATASETS.md.
