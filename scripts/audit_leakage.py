"""Leakage audit. Exit 0 if all checks pass, 1 otherwise."""
from __future__ import annotations

import argparse
import sys

from tablehop.data.splits import audit_splits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--manifest", default="data/splits/MANIFEST.json")
    a = ap.parse_args()
    ok = True
    for name, passed, detail in audit_splits(a.processed, a.manifest):
        print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
