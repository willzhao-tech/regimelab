# -*- coding: utf-8 -*-
"""Render-from-data guard: the headline numbers in results.json must reproduce from a clean,
cache-cleared rebuild. If this fails, the paper/dashboard/README are quoting stale numbers.
(Ported from the semiconductor study's 'rebuild from raw, ignore caches' reproduction discipline.)

Run:  python tests/test_results_reproducible.py   (or pytest)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import bookopt_harness as H
import bookopt_floor as F
import bookopt_stats as S

RESULTS = os.path.join(H.OUT, "results.json")


def test_results_reproduce():
    assert os.path.exists(RESULTS), "results.json missing — run build_results.py first"
    with open(RESULTS, encoding="utf-8") as f:
        R = json.load(f)

    H._DATA.clear()                                  # force rebuild from raw CSVs, ignore any cache
    book, sleeves, _ = F.build()
    allidx = pd.DatetimeIndex(sorted(set().union(*[set(sleeves[k].index) for k in sleeves])))
    cal = book.reindex(allidx).fillna(0.0)

    checks = {
        "floor.sharpe_active": (S.sharpe_ann(book), R["floor"]["sharpe_active"]),
        "floor.sharpe_calendar": (S.sharpe_ann(cal), R["floor"]["sharpe_calendar"]),
        "floor.deployed_pct": (len(book) / len(allidx) * 100, R["floor"]["deployed_pct"]),
        "floor.maxdd": (F.stat_line(book)["maxdd"], R["floor"]["maxdd"]),
        "meta.n_markets": (len(sleeves), R["meta"]["n_markets"]),
    }
    bad = []
    for name, (live, stored) in checks.items():
        if abs(float(live) - float(stored)) > 1e-3:
            bad.append(f"{name}: live {live:.4f} != results.json {stored}")
    assert not bad, "results.json is STALE — regenerate with build_results.py:\n  " + "\n  ".join(bad)


if __name__ == "__main__":
    test_results_reproduce()
    print("PASS - results.json reproduces from a clean rebuild (no prose/code drift).")
