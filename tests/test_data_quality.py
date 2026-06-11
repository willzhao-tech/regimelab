# -*- coding: utf-8 -*-
"""Fail loud on structurally corrupt OHLCV. Scans every *_all_history.csv the book depends on and
asserts no HARD violations (High<Low, non-positive price, dup dates). Soft WARNs are printed.

Run:  python tests/test_data_quality.py   (or pytest)"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import bookopt_harness as H
from data_quality import audit_ohlcv

# the underlyings + vol indices the book actually consumes
USED = set()
for _, uf, vf, _ in H.PAIRS:
    USED.add(uf); USED.add(vf)


def _load(path):
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df


def test_no_hard_corruption():
    hard, warn = [], []
    for f in sorted(glob.glob(os.path.join(H.OUT, "*_all_history.csv"))):
        stem = os.path.basename(f).replace("_all_history.csv", "")
        try:
            issues = audit_ohlcv(_load(f), stem)
        except Exception as e:
            hard.append(f"HARD {stem}: unreadable ({type(e).__name__})"); continue
        for i in issues:
            (hard if i.startswith("HARD") else warn).append(i)
    if warn:
        print("Soft warnings (not failing):")
        for w in warn:
            print("  " + w)
    # only fail on HARD issues in series the book actually uses
    blocking = [h for h in hard if any(h.startswith(f"HARD {u}.") or h.startswith(f"HARD {u}:") for u in USED)]
    assert not blocking, "STRUCTURAL data corruption in book inputs:\n  " + "\n  ".join(blocking)
    print(f"\nPASS - no hard corruption in {len(USED)} book-input series "
          f"({len(warn)} soft warnings across all CSVs).")


if __name__ == "__main__":
    test_no_hard_corruption()
