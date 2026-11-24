"""Run directory and meta.json."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

from tablehop.utils.device import get_device


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001
        return "unknown"


class RunMeta:
    """Handle to a run directory; writes meta.json."""

    def __init__(self, run_dir: Path, meta: dict):
        self.run_dir = run_dir
        self.meta = meta

    def _write(self) -> None:
        (self.run_dir / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))

    def finish(self, status: str = "done") -> None:
        """Set status (done|failed) and end time."""
        if status not in ("done", "failed"):
            raise ValueError(status)
        self.meta["status"] = status
        self.meta["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._write()


def env_lock_sha256(lock: str | Path = "requirements-lock.txt") -> str:
    """sha256 of the pinned dependency lock file (same value => same environment)."""
    p = Path(lock)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"


def new_run_dir(method: str, dataset: str, seed: int, cfg: dict | None = None,
                results_dir: str | Path = "results") -> RunMeta:
    """Create results/runs/<run_id>/ and write meta.json with status=running."""
    import torch

    cfg = cfg or {}
    run_id = f"{method}_{dataset}_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(results_dir) / "runs" / run_id
    n = 1
    while run_dir.exists():
        run_dir = Path(results_dir) / "runs" / f"{run_id}_{n}"
        n += 1
    run_dir.mkdir(parents=True)
    meta = {
        "run_id": run_dir.name, "method": method, "dataset": dataset, "seed": seed,
        "config": cfg, "git_hash": _git(["rev-parse", "HEAD"]) or "unknown",
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": get_device(cfg.get("device", "auto")),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": _pkg_version("transformers"), "env_lock_sha256": env_lock_sha256(),
        "status": "running",
    }
    rm = RunMeta(run_dir, meta)
    rm._write()
    return rm
