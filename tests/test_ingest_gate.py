# -*- coding: utf-8 -*-
"""The ingestion quality gate must block corrupt fetches BEFORE they reach disk,
log vendor revisions loudly, and pass clean incremental updates. Synthetic fetchers,
no network. (A single corrupt tick once flipped a headline alpha's sign — the gate
exists so that class of bug dies at the door.)

Run:  python tests/test_ingest_gate.py   (or pytest)
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from market_data import update_dataset, OHLCV_COLS


def _frame(rows):
    df = pd.DataFrame(rows, columns=["Date"] + OHLCV_COLS)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


CLEAN = _frame([("2026-06-01", 100, 105, 99, 104, 1000),
                ("2026-06-02", 104, 106, 103, 105, 1100)])
NEXT_OK = _frame([("2026-06-02", 104, 106, 103, 105, 1100),
                  ("2026-06-03", 105, 108, 104, 107, 1200)])
NEXT_BAD = _frame([("2026-06-03", 105, 103, 108, 107, 1200)])     # High < Low -> HARD
NEXT_REV = _frame([("2026-06-02", 104, 106, 103, 99.0, 1100),     # revised close 105 -> 99
                   ("2026-06-03", 105, 108, 104, 107, 1200)])


def _seed(path):
    CLEAN.to_csv(path, index_label="Date")


def test_clean_update_passes():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.csv"); _seed(p)
        info = update_dataset(p, lambda s, x: NEXT_OK, "2026-01-01", overlap_days=1, proxy=None)
        assert info["rows_added"] == 1 and info["total_rows"] == 3 and not info.get("blocked")


def test_corrupt_fetch_blocked():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.csv"); _seed(p)
        before = open(p, "rb").read()
        info = update_dataset(p, lambda s, x: NEXT_BAD, "2026-01-01", overlap_days=1, proxy=None)
        assert info.get("blocked"), "HARD violation must block the commit"
        assert open(p, "rb").read() == before, "file must be UNCHANGED after a blocked fetch"


def test_revision_detected():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.csv"); _seed(p)
        info = update_dataset(p, lambda s, x: NEXT_REV, "2026-01-01", overlap_days=1, proxy=None)
        assert info.get("revisions") == 1, "silent vendor revision must be counted/logged"
        out = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
        assert float(out.loc["2026-06-02", "Close"]) == 99.0, "fresh bar wins on overlap (by design)"


if __name__ == "__main__":
    test_clean_update_passes(); print("PASS - clean incremental update commits")
    test_corrupt_fetch_blocked(); print("PASS - corrupt fetch blocked, file untouched")
    test_revision_detected(); print("PASS - vendor revision detected and logged")
