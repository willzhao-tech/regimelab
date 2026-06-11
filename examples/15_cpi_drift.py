"""
Example 15 — CPI (inflation report) event study on NQ and VIX.

CPI release dates scraped from the Investing.com economic-calendar API (fetch_cpi.py ->
CPI_dates.csv) — the one catalyst whose dates the official sources (BLS/FRED) wouldn't give.
CPI releases 8:30am ET like NFP, and was the dominant market driver of 2021-2024.

Same gauntlet as FOMC/NFP: event-window means, t-stats, placebo-date null, day-0 vol ratio,
on BOTH NQ (returns) and VIX (% change) — since the NFP lesson was 'null on equities, real on vol'.

Run:  python examples/15_cpi_drift.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"


def tstat(a):
    a = np.asarray(a, float)
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 2 and a.std() > 0 else np.nan


def placebo_p(series, k, real, n=2000, seed=7):
    rng = np.random.default_rng(seed); pool = np.arange(5, len(series) - 3)
    dist = np.array([np.mean([series.iloc[q - 1] + series.iloc[q]
                              for q in rng.choice(pool, k, False)]) for _ in range(n)])
    return (dist >= real).mean()


def event_table(series, loc, label, scale, unit):
    uncond = series.mean()
    print(f"  {label}: unconditional {uncond*scale:+.1f} {unit}, vol {series.std()*scale:.0f} {unit}")
    print(f"  {'rel day':>8}{'mean':>10}{'t-stat':>9}{'vol ratio':>11}")
    w = {}
    for kk in range(-3, 4):
        vals = np.array([series.iloc[p + kk] for p in loc]); w[kk] = vals
        tag = "  <- CPI day" if kk == 0 else ""
        print(f"  {kk:>8}{vals.mean()*scale:>9.1f}{tstat(vals):>9.1f}{vals.std()/series.std():>10.2f}x{tag}")
    return w


def main():
    nq = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"),
                     parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    vix = pd.read_csv(os.path.join(DATA_DIR, "VIX_all_history.csv"),
                      parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    nq_ret = nq.pct_change().dropna()
    vix_chg = vix.pct_change().dropna()
    cpi = pd.read_csv(os.path.join(DATA_DIR, "CPI_dates.csv"), parse_dates=["cpi_date"])["cpi_date"]
    cpi = cpi[(cpi >= nq_ret.index.min()) & (cpi <= nq_ret.index.max())].sort_values()
    print(f"NQ {nq_ret.index.min().date()}..{nq_ret.index.max().date()}; CPI events {len(cpi)}\n")

    locn = [p for p in nq_ret.index.get_indexer(pd.DatetimeIndex(cpi), method="bfill") if 5 <= p < len(nq_ret) - 3]
    print("=" * 70)
    print("CPI on NQ (returns)")
    print("=" * 70)
    wnq = event_table(nq_ret, locn, "NQ return", 1e4, "bps")
    pre = wnq[-1] + wnq[0]
    print(f"  pre/at window (-1,0) {pre.mean()*1e4:+.1f} bps, t {tstat(pre):.1f}; "
          f"placebo p={placebo_p(nq_ret, len(locn), pre.mean()):.3f}; "
          f"CPI-day vol {wnq[0].std()/nq_ret.std():.2f}x")

    locv = [p for p in vix_chg.index.get_indexer(pd.DatetimeIndex(cpi), method="bfill") if 5 <= p < len(vix_chg) - 3]
    print("\n" + "=" * 70)
    print("CPI on VIX (daily % change)")
    print("=" * 70)
    wv = event_table(vix_chg, locv, "VIX %chg", 100, "%")
    vpre = wv[-1] + wv[0]
    print(f"  day -1..0 VIX {vpre.mean()*100:+.2f}% (t {tstat(vpre):.1f}); "
          f"placebo p={placebo_p(vix_chg, len(locv), vpre.mean()):.3f}")
    print(f"  day 0 VIX {wv[0].mean()*100:+.2f}% (t {tstat(wv[0]):.1f})  <- crush if negative")
    print("\n  Descriptive, daily close-to-close (misses the 8:30am intraday reaction), not deflated.")


if __name__ == "__main__":
    main()
