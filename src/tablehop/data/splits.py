"""Table-disjoint splits with held-out schema clusters (specs/01 "Split policy") and the audit."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

SPLITS = ("train", "dev", "test", "ood_test", "ood_dev")
ID_SPLITS = ("train", "dev", "test")
OOD_SPLITS = ("ood_test", "ood_dev")


def table_text(raw: dict) -> str:
    """Clustering text: title + section_title + header."""
    return " | ".join([raw.get("title", ""), raw.get("section_title", ""),
                       " ; ".join(h[0] for h in raw["header"])])


def embed_texts(texts: list[str], device: str = "cpu", batch_size: int = 128,
                model: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    """L2-normalized embeddings, shape (n, 384)."""
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model, device=device)
    return m.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                    show_progress_bar=False).astype(np.float32)


def cluster_tables(train_emb: np.ndarray, all_emb: np.ndarray, k: int = 20, seed: int = 0):
    """KMeans fitted on train-table embeddings only; returns labels for all_emb and train labels."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(train_emb)
    return km.predict(all_emb), km.labels_


def choose_heldout(sizes: dict[int, int], n_tables: int, n_held: int = 3,
                   frac: float = 0.05) -> list[int]:
    """The n_held clusters whose sizes are closest to frac*n_tables; ties by cluster id."""
    target = frac * n_tables
    order = sorted(sizes, key=lambda c: (abs(sizes[c] - target), c))
    return sorted(order[:n_held])


def _atype(q: dict) -> str:
    node = q.get("answer-node")
    return node[0][3] if node else "none"


def stratified_sample(qs: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Seeded sample of n questions, proportional to answer-node type (table/passage/none)."""
    if n >= len(qs):
        return list(qs)
    strata: dict[str, list[dict]] = defaultdict(list)
    for q in qs:
        strata[_atype(q)].append(q)
    keys = sorted(strata)
    alloc = {k: int(n * len(strata[k]) / len(qs)) for k in keys}
    rest = n - sum(alloc.values())
    for k in sorted(keys, key=lambda k: -len(strata[k]))[:rest]:
        alloc[k] += 1
    out = []
    for k in keys:
        pool = sorted(strata[k], key=lambda q: q["question_id"])
        out += rng.sample(pool, min(alloc[k], len(pool)))
    return sorted(out, key=lambda q: q["question_id"])


def assign_splits(train_q: list[dict], dev_q: list[dict], cluster_of: dict[str, int],
                  heldout: list[int], seed: int = 0, n_train: int = 3000, n_dev: int = 1000,
                  n_ood_dev: int = 500, dev_pool_frac: float = 0.25) -> dict[str, list[dict]]:
    """Return {split: [question dicts]} following specs/01 Split policy."""
    rng = random.Random(seed)
    held = set(heldout)
    is_held = lambda t: cluster_of[t] in held
    id_tables = sorted({q["table_id"] for q in train_q if not is_held(q["table_id"])})
    rng.shuffle(id_tables)
    n_dev_pool = int(len(id_tables) * dev_pool_frac)
    dev_tabs, train_tabs = set(id_tables[:n_dev_pool]), set(id_tables[n_dev_pool:])
    train = stratified_sample([q for q in train_q if q["table_id"] in train_tabs], n_train, rng)
    dev = stratified_sample([q for q in train_q if q["table_id"] in dev_tabs], n_dev, rng)
    used = {q["table_id"] for q in train} | {q["table_id"] for q in dev}
    test = [q for q in dev_q if not is_held(q["table_id"]) and q["table_id"] not in used]
    ood_test = [q for q in dev_q if is_held(q["table_id"])]
    ood_tabs = {q["table_id"] for q in ood_test}
    ood_pool = [q for q in train_q if is_held(q["table_id"]) and q["table_id"] not in ood_tabs]
    ood_dev = stratified_sample(ood_pool, n_ood_dev, rng)
    return {"train": train, "dev": dev, "test": test, "ood_test": ood_test, "ood_dev": ood_dev}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_splits(processed_dir: str | Path, manifest_path: str | Path) -> list[tuple[str, bool, str]]:
    """Run leakage checks. Returns [(check, passed, detail)]. Reads jsonl lines as raw dicts."""
    processed_dir = Path(processed_dir)
    with open(manifest_path) as f:
        manifest = json.load(f)
    held = set(manifest["heldout_clusters"])
    rows: dict[str, list[dict]] = {}
    for s in SPLITS:
        p = processed_dir / f"{s}.jsonl"
        if p.exists():
            with open(p) as f:
                rows[s] = [json.loads(line) for line in f]
        else:
            rows[s] = []
    tabs = {s: {r["table"]["id"] for r in rows[s]} for s in SPLITS}
    res = []
    bad = []
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            inter = tabs[a] & tabs[b]
            if inter:
                bad.append(f"{a}&{b}:{len(inter)}")
    res.append(("table_id_disjoint_all_pairs", not bad, "; ".join(bad) or "no overlap"))
    clus = {s: {r["domain"] for r in rows[s]} for s in SPLITS}
    leak = [s for s in ID_SPLITS if {int(c[1:]) for c in clus[s] if c} & held]
    leak += [s for s in OOD_SPLITS if {int(c[1:]) for c in clus[s] if c} - held]
    res.append(("cluster_disjoint_id_vs_ood", not leak,
                f"heldout={sorted(held)}; violating splits={leak}" if leak else "ok"))
    srcs = {"train": "train", "dev": "train", "ood_dev": "train", "test": "dev", "ood_test": "dev"}
    wrong = [s for s in SPLITS if any(r["meta"].get("source") != srcs[s] for r in rows[s])]
    res.append(("no_official_test_usage", not wrong and not manifest.get("official_test_used", False),
                f"splits with unexpected source: {wrong}" if wrong else "all ids from train/dev files"))
    ids = [r["id"] for s in SPLITS for r in rows[s]]
    res.append(("example_ids_unique", len(ids) == len(set(ids)), f"{len(ids)} ids"))
    return res
