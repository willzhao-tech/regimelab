# -*- coding: utf-8 -*-
"""Causality / no-look-ahead test for the floor-book harness.

The gold-standard test for a backtest: FUTURE-PERTURBATION INVARIANCE.
Corrupt all input data AFTER a cutoff date T with wild noise, rebuild the book, and assert the
book P&L on dates <= T is byte-identical. If any signal, statistic, walk-forward pick, weight, or
leverage term peeks at the future, corrupting the future would leak into the past and this fails.

This is the single most important guarantee in the whole study (the session caught multiple
look-ahead artifacts by hand; this test makes the property machine-checked).

Run:  python tests/test_book_causality.py    (or: pytest tests/test_book_causality.py)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import bookopt_harness as H
import bookopt_floor as F


def _reload_pristine():
    H._DATA.clear()
    H._load()


def _corrupt_after(T, seed=7):
    """Replace every market's OHLC / returns / vol-index AFTER T with wild noise (in place on _DATA)."""
    H._load()
    rng = np.random.default_rng(seed)
    for name in list(H._DATA):
        df, ret, vi, sp0 = H._DATA[name]
        df, ret, vi = df.copy(), ret.copy(), vi.copy()
        dm = df.index > T
        if dm.any():
            fac = rng.uniform(0.3, 3.0, int(dm.sum()))
            for col in ("Open", "High", "Low", "Close"):
                if col in df.columns:
                    df.loc[dm, col] = df.loc[dm, col].values * fac
        rm = ret.index > T
        if rm.any():
            ret.loc[rm] = rng.uniform(-5.0, 5.0, int(rm.sum())) * 0.05
        vm = vi.index > T
        if vm.any():
            vi.loc[vm] = vi.loc[vm].values * rng.uniform(0.4, 4.0, int(vm.sum()))
        H._DATA[name] = (df, ret, vi, sp0)


def test_no_lookahead():
    _reload_pristine()
    clean, _, _ = F.build()
    assert H.sharpe(clean) > 0.5, "sanity: clean book should be profitable"
    T = clean.index[int(len(clean) * 0.6)]

    _reload_pristine()
    _corrupt_after(T)
    pert, _, _ = F.build()

    common = clean.loc[:T].index.intersection(pert.loc[:T].index)
    assert len(common) > 500, "expected a substantial pre-cutoff overlap to test"
    diff = float((clean.loc[common] - pert.loc[common]).abs().max())
    _reload_pristine()  # restore for any later tests
    assert diff < 1e-9, (
        f"LOOK-AHEAD DETECTED: corrupting data after {T.date()} changed pre-cutoff "
        f"book P&L by {diff:.2e} (must be 0)."
    )


if __name__ == "__main__":
    test_no_lookahead()
    print("PASS - no look-ahead: pre-cutoff book P&L is invariant to future-data corruption.")
