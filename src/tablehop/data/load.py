"""Split loading shared by training/eval scripts (hybridqa from data/processed, or synthetic)."""
from __future__ import annotations

from pathlib import Path

from tablehop.data.schema import Example

SYN_SEEDS = {"train": 1000, "dev": 2000, "test": 3000, "ood_dev": 4000, "ood_test": 5000}


def load_split(data_cfg: dict, split: str, limit: int | None = None) -> list[Example]:
    """Load one split. `data_cfg`: dataset (hybridqa|synthetic), processed_dir, synthetic_n."""
    dataset = data_cfg.get("dataset", "hybridqa")
    if dataset == "synthetic":
        from tablehop.data.synthetic import generate

        n = int(data_cfg.get("synthetic_n", {}).get(split, 32))
        exs = generate(n, SYN_SEEDS[split])
    else:
        path = Path(data_cfg.get("processed_dir", "data/processed")) / f"{split}.jsonl"
        with open(path) as f:
            exs = [Example.from_json(line) for line in f if line.strip()]
    return exs[:limit] if limit else exs
