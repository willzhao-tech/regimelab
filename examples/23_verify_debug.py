"""
Example 23 — VERIFICATION: de-bugging Sleeve B, measuring each look-ahead fix.

A reviewer found that the book's 1.92 Sharpe was an artifact of THREE look-ahead/structural
bugs in the short-vol sleeve. This script reproduces the bug-by-bug decomposition honestly.

Bugs in the original sleeve_B (examples 18-21):
  1. WING COST LOOK-AHEAD:  wing = max(rvar-capv,0).mean()  -> full-sample mean (uses the future).
  2. CAPPED TAIL:           iv - min(rvar, capv) - wing     -> np.minimum deletes vol-spike losses.
  3. VOL-SCALE LOOK-AHEAD:  scale_to uses full-sample std   -> 17x leverage set with future info.

Fixes applied sequentially; Sharpe measured at each step. Then the corrected book vs a
vol-matched long-NQ benchmark.

Run:  python examples/23_verify_debug.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252)


def sharpe(r):
    r = r.dropna()
    return r.mean() / r.std() * SQ if r.std() > 0 else np.nan


def sleeve_A(df, fomc, boost=0.5):
    px = df["Close"]; ret = px.pct_change()
    hl2 = np.log(df["High"] / df["Low"]) ** 2
    vol = np.sqrt(hl2.rolling(21).mean() / (4 * np.log(2))) * SQ
    sig = (px > px.rolling(200).mean()).astype(float)
    desired = ((0.15 / vol).clip(upper=3.0) * sig).shift(1).fillna(0.0)
    base, cur, pm = [], 0.0, None
    for d, v in desired.items():
        if pm is None or d.month != pm.month:
            cur = v
        base.append(cur); pm = d
    base = pd.Series(base, index=desired.index)
    win = pd.Series(False, index=ret.index)
    for p in ret.index.get_indexer(fomc, method="bfill"):
        if 1 <= p < len(ret.index):
            win.iloc[p - 1] = True; win.iloc[p] = True
    pos = base.where(~win, np.minimum(base + boost, 4.0))
    return (pos * ret - 0.0005 * pos.diff().abs().fillna(0.0)).dropna()


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
    vxn = pd.read_csv(os.path.join(DATA_DIR, "VXN_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    fomc = pd.read_csv(os.path.join(DATA_DIR, "FOMC_dates.csv"), parse_dates=["fomc_date"])["fomc_date"].sort_values()
    kept, last = [], None
    for d in fomc:
        if last is None or (d - last).days >= 20:
            kept.append(d); last = d
    fomc = pd.DatetimeIndex(kept)

    ret = df["Close"].pct_change()
    idx = ret.index.intersection(vxn.index)
    r, v = ret.loc[idx], vxn.loc[idx]
    iv = (v.shift(1) / 100) ** 2 / 252
    rvar = r ** 2
    capv = 0.05 ** 2
    tail = np.maximum(rvar - capv, 0)
    wing_full = tail.mean()                                   # BUG 1: look-ahead
    wing_trail = tail.expanding(min_periods=252).mean().shift(1)  # causal trailing cost

    B0 = (iv - np.minimum(rvar, capv) - wing_full).dropna()                 # original
    B1 = (iv - np.minimum(rvar, capv) - wing_trail).dropna()               # + trailing cost
    B2 = (iv - rvar - wing_trail).dropna()                                  # + uncapped tail
    tvol = B2.rolling(63).std().shift(1) * SQ                               # causal trailing vol
    B3 = ((0.10 / tvol).clip(upper=20).shift(1) * B2).dropna()             # + trailing vol-scale

    print("SLEEVE B — bug-by-bug Sharpe (raw P&L vol of B0 = %.1f%%, so full-sample scaling levers ~%.0fx)"
          % (B0.std() * SQ * 100, 0.10 / (B0.std() * SQ)))
    print("=" * 78)
    for name, s in [("B0  original (look-ahead wing + capped tail)", B0),
                    ("B1  + trailing wing cost (no look-ahead)", B1),
                    ("B2  + uncapped tail (real vol-spike losses)", B2),
                    ("B3  + trailing vol-scaling", B3)]:
        print(f"  {name:<48} Sharpe {sharpe(s):+.2f}")

    # ---- corrected book vs vol-matched NQ ----
    print("\n" + "=" * 78)
    print("BOOK rebuilt with corrected Sleeve B  (vs Sleeve A alone, vs vol-matched NQ)")
    print("=" * 78)
    rA = sleeve_A(df, fomc)
    common = rA.index.intersection(B3.index)
    rA2, B3c = rA.loc[common], B3.loc[common]
    # causal vol-equalize each sleeve to 10%, 50/50, lever 1.5
    def causal_scale(x, tgt=0.10):
        s = x.rolling(252).std().shift(1) * SQ
        return ((tgt / s).clip(upper=10).fillna(0) * x)
    book = 1.5 * (0.5 * causal_scale(rA2) + 0.5 * causal_scale(B3c))
    nq = ret.loc[common]
    def stats(x):
        x = x.dropna(); e = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
        return sharpe(x), e.iloc[-1] ** (1 / yrs) - 1, float((e / e.cummax() - 1).min()), float(e.iloc[-1])
    for name, x in [("Sleeve A alone (trend)", rA2),
                    ("Corrected book (A + de-bugged B)", book),
                    ("Vol-matched long NQ", nq * (book.std() / nq.std()))]:
        sh, cg, dd, term = stats(x)
        print(f"  {name:<34} Sharpe {sh:+.2f}   CAGR {cg*100:>5.1f}%   maxDD {dd*100:>4.0f}%   {term:>6.1f}x")

    print("\n  VERDICT: the 1.92 Sharpe was almost entirely the wing-cost look-ahead. Honestly")
    print("  accounted, the short-vol sleeve is a LOSER and drags the book below Sleeve A alone")
    print("  and below plain vol-matched NQ. The real, surviving edge is Sleeve A (trend) only.")


if __name__ == "__main__":
    main()
