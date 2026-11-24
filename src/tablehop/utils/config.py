"""YAML config loading with `base:` inheritance and CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, over: dict) -> dict:
    """Return base with over merged recursively (over wins)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    """Load YAML; `base:` (relative to this file) is loaded first and overridden."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is not None:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `key.sub=value` overrides; values parsed as YAML scalars."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value: {item}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


class Config(dict):
    """Dict with attribute access for nested keys."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v


def get_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load, inherit and override a config."""
    return Config(apply_overrides(load_config(path), overrides or []))
