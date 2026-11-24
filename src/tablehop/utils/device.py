"""Device selection."""
from __future__ import annotations


def get_device(pref: str = "auto") -> str:
    """Return "cuda" | "mps" | "cpu". `auto` picks best available."""
    import torch

    if pref in ("cuda", "mps", "cpu"):
        if pref == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if pref == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return pref
    if pref != "auto":
        raise ValueError(f"unknown device {pref}")
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
