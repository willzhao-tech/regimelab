# -*- coding: utf-8 -*-
"""
p0_robust.py — TASK P0-3 ROBUSTNESS for the long-short vol-arb blend (NQ/VXN, proxy).

Re-runs the walk-forward A+B 50/50 blend under a 3x3 grid:
  train/test configs : (1000,200), (1260,252), (1500,252)
  start dates        : full (2001 data start), 2008-01-01, 2013-01-01
Input data are TRUNCATED before any computation (ret, forecasts, signals all
rebuilt from the truncated sample) so the first OOS block moves accordingly and
no pre-start information leaks in.

Per cell: OOS Sharpe / skew / alpha-t vs static short-vol (s=+1, same dates).
PASS criterion: Sharpe > 0.8 AND skew > 0 in >= 7/9 cells.
Output grid -> C:\\Users\\ASUS\\Desktop\\claude doc\\1\\p0_robustness_grid.csv

HARD-RULE compliance:
  - All signals causal (harness fcast_vol shifts forecasts; backtest shifts s).
  - Params chosen ONLY by trailing-train walk-forward inside each cell.
  - NO tail deletion / clipping / winsorizing anywhere; uncapped variance P&L.
  - Same fixed grids as the original study (A: 18 combos, B: 9 combos = 27 per
    block). The 9 robustness cells are exhaustively REPORTED, not selected over.
  - Numbers below are OOS-concatenated test blocks only; train numbers never shown.
"""
import os
import sys

sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd

import volarb_harness as H

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

GRID_A = [(a, b, c) for a in (2.0, 4.0, 6.0) for b in (0.0, -2.0) for c in (0.0, 1.0, 2.0)]
GRID_B = [(b1, b2) for b1 in (0.8, 1.0, 1.2) for b2 in (1.3, 1.6, 2.0)]

CONFIGS = [(1000, 200), (1260, 252), (1500, 252)]
STARTS = [("full", None), ("2008-01-01", pd.Timestamp("2008-01-01")),
          ("2013-01-01", pd.Timestamp("2013-01-01"))]

df_full, ret_full, vxn_full = H.load()
print(f"aligned data: {df_full.index[0].date()} .. {df_full.index[-1].date()}  ({len(df_full)} days)")
print(f"grids: A={len(GRID_A)} combos, B={len(GRID_B)} combos (27/block, same as original study)")
print(f"robustness cells reported exhaustively: {len(CONFIGS) * len(STARTS)} (no selection over cells)\n")


def truncate(start_ts):
    """Truncate raw inputs, then rebuild ret from the truncated closes so nothing
    before start_ts (not even the boundary-spanning return) enters the pipeline."""
    if start_ts is None:
        df = df_full
    else:
        df = df_full.loc[df_full.index >= start_ts]
    ret = df["Close"].pct_change()
    vxn = vxn_full.reindex(df.index)
    return df, ret, vxn


def alpha_t_vs_static(oos, static):
    yx = pd.concat([oos, static], axis=1).dropna().values
    y, x = yx[:, 0], yx[:, 1]
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    t_a = b[0] / np.sqrt((r @ r / (len(y) - 2)) * np.linalg.inv(X.T @ X)[0, 0])
    return float(t_a), float(b[1])


def run_cell(start_ts, train, test):
    df, ret, vxn = truncate(start_ts)
    # causal building blocks rebuilt on the truncated sample (cold start: warm-up
    # NaNs at the head mean the signal is flat there, exactly as live trading would be)
    fc21 = H.fcast_vol(df, ret, "park21")
    fc10 = H.fcast_vol(df, ret, "park10")
    fc42 = H.fcast_vol(df, ret, "park42")
    richness = vxn - fc21
    trend = fc10 - fc42
    rng = np.log(df["High"] / df["Low"]) * 100.0
    be = vxn / SQ

    def sig_A(g):
        r_hi, r_lo, d = g
        s = pd.Series(0.0, index=ret.index)
        s[((richness >= r_hi) & (trend <= -d)).fillna(False)] = 1.0
        s[((richness <= r_lo) & (trend >= d)).fillna(False)] = -1.0
        return s

    def sig_B(g):
        b1, b2 = g
        s = pd.Series(0.0, index=ret.index)
        ok = rng.notna() & be.notna()
        s[ok & (rng < b1 * be)] = 1.0
        s[ok & (rng > b2 * be)] = -1.0
        return s

    oos_A = H.walk_forward(lambda g: H.backtest(sig_A(g), ret, vxn), GRID_A, train=train, test=test)
    oos_B = H.walk_forward(lambda g: H.backtest(sig_B(g), ret, vxn), GRID_B, train=train, test=test)
    n_blocks = len(oos_A.attrs.get("picks", []))
    common = oos_A.index.intersection(oos_B.index)
    if len(common) < 60:
        return None
    blend = 0.5 * oos_A.loc[common] + 0.5 * oos_B.loc[common]
    static = H.backtest(pd.Series(1.0, index=ret.index), ret, vxn).reindex(common).dropna()
    m = H.metrics(blend)
    ms = H.metrics(static)
    t_a, beta = alpha_t_vs_static(blend, static)
    return dict(sharpe=m["sharpe"], skew=m["skew"], tstat=m["tstat"],
                alpha_t=t_a, beta=beta,
                static_sharpe=ms["sharpe"], static_skew=ms["skew"],
                n_days=m["n"], n_blocks=n_blocks,
                oos_start=str(common.min().date()), oos_end=str(common.max().date()))


rows = []
for train, test in CONFIGS:
    for sname, sts in STARTS:
        r = run_cell(sts, train, test)
        if r is None:
            r = dict(sharpe=np.nan, skew=np.nan, tstat=np.nan, alpha_t=np.nan, beta=np.nan,
                     static_sharpe=np.nan, static_skew=np.nan, n_days=0, n_blocks=0,
                     oos_start="", oos_end="")
        r.update(train=train, test=test, start=sname)
        r["pass_cell"] = bool(r["sharpe"] > 0.8 and r["skew"] > 0)
        rows.append(r)
        print(f"train={train:>4} test={test:>3} start={sname:<10} "
              f"OOS {r['oos_start']}..{r['oos_end']} ({r['n_days']:>4}d, {r['n_blocks']:>2} blocks)  "
              f"Sharpe {r['sharpe']:5.2f}  skew {r['skew']:+6.1f}  alpha-t {r['alpha_t']:+5.1f}  "
              f"[static Sh {r['static_sharpe']:5.2f} skew {r['static_skew']:+6.1f}]  "
              f"{'PASS' if r['pass_cell'] else 'fail'}")

grid = pd.DataFrame(rows)[["train", "test", "start", "oos_start", "oos_end", "n_days", "n_blocks",
                           "sharpe", "tstat", "skew", "alpha_t", "beta",
                           "static_sharpe", "static_skew", "pass_cell"]].round(4)
grid.to_csv(os.path.join(OUT, "p0_robustness_grid.csv"), index=False)

n_pass = int(grid["pass_cell"].sum())
print("\n=== Sharpe grid (rows = train/test, cols = start) ===")
print(grid.pivot(index=["train", "test"], columns="start", values="sharpe").round(2).to_string())
print("\n=== skew grid ===")
print(grid.pivot(index=["train", "test"], columns="start", values="skew").round(1).to_string())
print("\n=== alpha-t-vs-static grid ===")
print(grid.pivot(index=["train", "test"], columns="start", values="alpha_t").round(1).to_string())

print(f"\nPASS criterion: Sharpe > 0.8 AND skew > 0 in >= 7/9 cells")
print(f"RESULT: {n_pass}/9 cells pass -> {'PASS' if n_pass >= 7 else 'FAIL'}")
print(f"grid saved -> {os.path.join(OUT, 'p0_robustness_grid.csv')}")

print("""
WHERE THE SHARPE COMES FROM / WHAT BREAKS IT:
  - P&L is the PROXY instrument (1-day variance at VXN^2 strike, 0.5 vol-pt cost on
    position changes). It is not directly tradable; proxy inflates Sharpe LEVELS for
    strategy and static baseline alike. Robust claims are relative: alpha vs static
    and positive skew. A real NDX var-swap/option implementation loses roughly
    one-third to one-half of the Sharpe level.
  - If VXN close were not observable before next-day positioning (it is: signal is
    shifted a full day), or if costs exceed ~the assumed 0.5 vol-pt per turnover
    unit, the edge shrinks; the long-vol side (source of positive skew) is most
    cost-sensitive because it trades around vol spikes.
  - Short-history cells (2013 start) have few OOS blocks; their Sharpe carries wide
    confidence bands (see tstat column in the CSV).""")
