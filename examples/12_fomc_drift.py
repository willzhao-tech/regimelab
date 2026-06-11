"""
Example 12 — the pre-FOMC drift on NQ (event study, the one catalyst with a real prior).

Lucca & Moench (2015): US equities drift UP in the ~24h before scheduled FOMC announcements.
We test it on NQ with daily data and FOMC announcement dates scraped from the Fed site
(FOMC_dates.csv, via fetch_fomc.py — auditable). Honest gauntlet:
  * event-window mean returns (relative trading day -5..+3) vs the unconditional mean;
  * SCHEDULED meetings only (drop emergency intermeeting cuts: they occur during crashes
    and would bias the pre-drift down) — min 20-day gap filter;
  * PLACEBO-DATE null: 2000 random date sets of the same size; is the real pre-window in
    the tail? (the key test that the effect is about FOMC, not just any days);
  * a tradable pre-FOMC long with costs, vs buy-&-hold time-in-market.

Run:  python examples/12_fomc_drift.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"


def tstat(a):
    a = np.asarray(a, float)
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 2 and a.std() > 0 else np.nan


def main():
    px = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"),
                     parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    ret = px.pct_change().dropna()
    idx = ret.index

    fomc = pd.read_csv(os.path.join(DATA_DIR, "FOMC_dates.csv"), parse_dates=["fomc_date"])["fomc_date"]
    fomc = fomc[(fomc >= idx.min()) & (fomc <= idx.max())].sort_values()
    # scheduled-only: drop a meeting <20 days after the previous kept one (emergency cuts)
    kept, last = [], None
    for d in fomc:
        if last is None or (d - last).days >= 20:
            kept.append(d); last = d
    sched = pd.DatetimeIndex(kept)
    print(f"NQ {idx.min().date()}..{idx.max().date()}; FOMC dates {len(fomc)} total, "
          f"{len(sched)} scheduled (after emergency filter)\n")

    # map each meeting to its trading-day position (next trading day if needed)
    loc = idx.get_indexer(sched, method="bfill")
    loc = [p for p in loc if 5 <= p < len(idx) - 3]
    uncond = ret.mean()

    print("=" * 72)
    print("EVENT WINDOW — mean NQ return by trading day relative to FOMC announcement")
    print("=" * 72)
    print(f"  unconditional daily mean = {uncond*1e4:+.1f} bps\n")
    print(f"  {'rel day':>8}{'mean bps':>10}{'t-stat':>9}{'vs uncond':>11}")
    windows = {}
    for k in range(-5, 4):
        vals = np.array([ret.iloc[p + k] for p in loc])
        windows[k] = vals
        tag = "  <- announce" if k == 0 else ("  <- pre-day" if k == -1 else "")
        print(f"  {k:>8}{vals.mean()*1e4:>9.1f}{tstat(vals):>9.1f}{(vals.mean()-uncond)*1e4:>+10.1f}{tag}")

    pre = windows[-1] + windows[0]           # 2-day pre/at-announcement drift
    print(f"\n  PRE-FOMC window (day -1 + day 0) mean = {pre.mean()*1e4:+.1f} bps "
          f"(t {tstat(pre):.1f}); ~{(pre.mean()-2*uncond)*1e4:+.1f} bps above 2x baseline")

    # ---- subperiod stability (this anomaly reportedly decayed after Lucca-Moench 2015) --
    print("\n  SUBPERIOD STABILITY (does it survive after the 2015 publication?)")
    ev_dates = pd.DatetimeIndex([idx[p] for p in loc])
    for lo, hi in [("1999", "2011"), ("2012", "2018"), ("2019", "2026")]:
        sel = [j for j, d in enumerate(ev_dates) if lo <= str(d.year) <= hi]
        sub = pre[sel]
        print(f"    {lo}-{hi}: {len(sub):>3} meetings  pre-window {sub.mean()*1e4:>+6.1f} bps  t {tstat(sub):>+4.1f}")

    # ---- placebo-date null ------------------------------------------------------
    rng = np.random.default_rng(7); N = 2000; K = len(loc)
    real = pre.mean()
    pool = np.arange(5, len(idx) - 3)
    placebo = []
    for _ in range(N):
        ps = rng.choice(pool, size=K, replace=False)
        placebo.append(np.mean([ret.iloc[p - 1] + ret.iloc[p] for p in ps]))
    placebo = np.array(placebo)
    pval = (placebo >= real).mean()
    print(f"\n  PLACEBO NULL: real pre-window {real*1e4:+.1f} bps vs placebo median "
          f"{np.median(placebo)*1e4:+.1f} bps -> p = {pval:.3f} "
          f"({'in tail, real' if pval < 0.05 else 'NOT distinguishable from random'})")

    # ---- tradable: long NQ only on day -1 and day 0, 1bp/side, vs always-in ------
    invested = pd.Series(0.0, index=idx)
    for p in loc:
        invested.iloc[p - 1] = 1.0; invested.iloc[p] = 1.0
    cost = 0.0001 * invested.diff().abs().fillna(0.0)
    strat = invested * ret - cost
    days_in = int(invested.sum())
    eq = float((1 + strat).prod())
    bh_same = float((1 + ret).prod())
    print("\n" + "=" * 72)
    print("TRADABLE — long NQ ONLY on day -1 & announcement day (else flat), 1bp/side")
    print("=" * 72)
    print(f"  in-market {days_in}/{len(idx)} days ({days_in/len(idx)*100:.0f}% of the time)")
    print(f"  strategy $1 -> {eq:.2f}x   |  buy&hold (always in) $1 -> {bh_same:.1f}x")
    print(f"  capture: {(eq-1)/(bh_same-1)*100:.0f}% of buy&hold's total gain in "
          f"{days_in/len(idx)*100:.0f}% of the days")
    print("\n  NOTE: daily data — cannot isolate the intraday pre-2pm run-up; this is the")
    print("  close-to-close proxy. Real (gross) drift before 2pm is likely larger; net of")
    print("  realistic costs and slippage the edge is small. Descriptive event study.")


if __name__ == "__main__":
    main()
