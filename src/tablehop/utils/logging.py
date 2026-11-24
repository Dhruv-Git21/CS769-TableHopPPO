"""CSV train logger and stdlib logger."""
from __future__ import annotations

import csv
import logging
from pathlib import Path


def get_logger(name: str = "tablehop") -> logging.Logger:
    """Stdlib logger with a single stream handler."""
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


class CSVLogger:
    """Appends dict rows to train_log.csv; header fixed by first row (new keys rewrite file)."""

    def __init__(self, run_dir: str | Path, filename: str = "train_log.csv"):
        self.path = Path(run_dir) / filename
        self.rows: list[dict] = []

    def log(self, row: dict) -> None:
        self.rows.append(dict(row))
        keys: list[str] = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
