"""Compare two folders of evaluation results, file by file and model by model.

Used by the Colab notebooks to check that a rerun gives exactly the committed numbers:
    python 01-crop-recommendation/src/models/compare_results.py <committed folder> <new run folder>
Rows are matched on their keys (analysis, test, model, district, true, predicted), so the row order does not
matter. Text columns that are empty on both sides count as equal. Exit code 0 when everything is the same.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["analysis", "test", "model", "district", "true", "predicted", "crop"]
TOLERANCE = 1e-9


def compare_file(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    keys = [k for k in KEYS if k in a.columns and k in b.columns]
    if keys:
        a, b = a.set_index(keys).sort_index(), b.set_index(keys).sort_index()
    a, b = a.align(b, join="outer")
    gap = pd.Series(0.0, index=a.index)
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            d = (x - y).abs()
            d[x.isna() != y.isna()] = np.inf
            gap = np.maximum(gap, d.fillna(0.0))
        else:
            tx = x.astype(object).where(x.notna(), "<missing>").astype(str)
            ty = y.astype(object).where(y.notna(), "<missing>").astype(str)
            gap[(tx != ty).to_numpy()] = np.inf
    bad = gap > TOLERANCE
    if not bad.any():
        return True, "same"
    who = sorted(set(a.index[bad].get_level_values("model"))) if "model" in keys else []
    size = "rows missing or changed" if np.isinf(gap.max()) else f"largest difference {gap.max():.4f}"
    return False, size + (f" in {', '.join(who)}" if who else "")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    official, new = Path(argv[0]), Path(argv[1])
    all_same = True
    for f in sorted(official.glob("*.csv")):
        other = new / f.name
        if not other.exists():
            print(f"MISSING {f.name} in {new}")
            all_same = False
            continue
        same, what = compare_file(pd.read_csv(f), pd.read_csv(other))
        print(f"{'same' if same else 'DIFF'} {f.name}" + ("" if same else f": {what}"))
        all_same &= same
    print("\nThis run reproduces the committed numbers exactly." if all_same
          else "\nSome numbers differ. Check that the library versions match requirements-eval.txt.")
    return 0 if all_same else 1


if __name__ == "__main__":
    raise SystemExit(main())
