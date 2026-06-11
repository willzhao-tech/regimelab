"""
Example 14 — NFP (jobs report) event study on NQ and VIX, with accurate release dates.

Official release-date sources are all blocked here (BLS 403s; FRED/ALFRED unreachable; WebFetch
403s on the Fed/Philly-Fed sites). But the BLS schedule is DETERMINISTIC, so we compute it:
the Employment Situation is released on the 3rd Friday AFTER the reference week (the Sun-Sat week
containing the 12th of the reference month), shifted off the observed July-4 holiday. This is
materially better than the naive 'first Friday' (which mis-dates ~15% of months).

Test (same gauntlet as FOMC): event-window means + t-stats + placebo-date null, on:
  - NQ returns (is there a tradable drift? — likely not, no documented NFP equity pre-drift);
  - VIX (the jobs report is an uncertainty event — does implied vol bid up before and crush after?).

Run:  python examples/14_nfp_drift.py
"""
import os, sys, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"


def nfp_dates(start, end):
    """BLS rule: 3rd Friday after the reference week (week containing the 12th), minus the
    observed-July-4 holiday shift. Reference month R -> release in R+1."""
    out = []
    y, m = start.year - 1, start.month
    for _ in range((end.year - start.year + 2) * 12):
        ref12 = dt.date(y, m, 12)
        sat = ref12 + dt.timedelta(days=(5 - ref12.weekday()) % 7)        # Sat ending ref week
        first_fri = sat + dt.timedelta(days=((4 - sat.weekday()) % 7) or 7)  # 1st Fri strictly after
        rel = first_fri + dt.timedelta(days=14)                            # 3rd Friday
        jul4 = dt.date(rel.year, 7, 4)
        if rel.month == 7 and jul4.weekday() == 5 and rel == jul4 - dt.timedelta(days=1):
            rel -= dt.timedelta(days=1)                                    # observed holiday -> Thu
        out.append(rel)
        m += 1
        if m > 12:
            m = 1; y += 1
    return pd.DatetimeIndex([d for d in out if start.date() <= d <= end.date()])


def tstat(a):
    a = np.asarray(a, float)
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 2 and a.std() > 0 else np.nan


def placebo_p(series, loc, real, n=2000, seed=7):
    """p = fraction of random K-date sets whose 2-day window mean >= the real one."""
    rng = np.random.default_rng(seed)
    pool = np.arange(5, len(series) - 3)
    K = len(loc)
    dist = np.array([np.mean([series.iloc[q - 1] + series.iloc[q]
                              for q in rng.choice(pool, K, False)]) for _ in range(n)])
    return (dist >= real).mean(), np.median(dist)


def event_table(series, loc, label, unit):
    uncond = series.mean()
    print(f"  {label}: unconditional {uncond*1e4:+.1f} {unit}")
    print(f"  {'rel day':>8}{'mean':>10}{'t-stat':>9}")
    w = {}
    for k in range(-3, 4):
        vals = np.array([series.iloc[p + k] for p in loc]); w[k] = vals
        tag = "  <- NFP day" if k == 0 else ""
        print(f"  {k:>8}{vals.mean()*1e4:>9.1f}{tstat(vals):>9.1f}{tag}")
    return w


def main():
    nq = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"),
                     parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    vix = pd.read_csv(os.path.join(DATA_DIR, "VIX_all_history.csv"),
                      parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    nq_ret = nq.pct_change().dropna()
    vix_chg = vix.pct_change().dropna()
    idx = nq_ret.index

    nfp = nfp_dates(idx.min(), idx.max())
    naive = pd.date_range(idx.min(), idx.max(), freq="WOM-1FRI")
    n_diff = len(set(nfp.date) ^ set(naive.date))
    print(f"NQ {idx.min().date()}..{idx.max().date()}; NFP events {len(nfp)} "
          f"(rule-based; differs from naive first-Friday on {n_diff} dates)\n")

    loc = [p for p in idx.get_indexer(nfp, method="bfill") if 5 <= p < len(idx) - 3]

    print("=" * 70)
    print("NFP on NQ (returns)")
    print("=" * 70)
    wnq = event_table(nq_ret, loc, "NQ return", "bps")
    pre = wnq[-1] + wnq[0]
    p_nq, med = placebo_p(nq_ret, loc, pre.mean())
    print(f"  pre/at window (-1,0) {pre.mean()*1e4:+.1f} bps, t {tstat(pre):.1f}; "
          f"placebo p={p_nq:.3f} ({'real' if p_nq<0.05 else 'NULL'})")

    # VIX: align to its own index
    vloc = [p for p in vix_chg.index.get_indexer(nfp, method="bfill") if 5 <= p < len(vix_chg) - 3]
    print("\n" + "=" * 70)
    print("NFP on VIX (daily % change) — uncertainty bid before, vol-crush after?")
    print("=" * 70)
    wv = event_table(vix_chg, vloc, "VIX %chg", "bps")
    vpre = wv[-1] + wv[0]
    p_v, _ = placebo_p(vix_chg, vloc, vpre.mean())
    crush = wv[0] + wv[1]
    print(f"  day -1..0 VIX %chg {vpre.mean()*100:+.2f}% (t {tstat(vpre):.1f}); placebo p={p_v:.3f}")
    print(f"  day 0..+1 VIX %chg {crush.mean()*100:+.2f}% (t {tstat(crush):.1f})  <- vol crush if negative")
    print("\n  Read: NFP's equity drift is typically null (no documented pre-drift); the cleaner")
    print("  signature is in VIX (event premium then crush). Daily data still misses the 8:30am")
    print("  intraday reaction; descriptive, not deflated.")


if __name__ == "__main__":
    main()
