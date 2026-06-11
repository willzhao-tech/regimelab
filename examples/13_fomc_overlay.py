"""
Example 13 — harvest the pre-FOMC drift as an OVERLAY on the equity strategy.

12 showed a real, persistent pre-FOMC drift (+48 bps over day -1 & day 0, placebo p=0.005).
FOMC dates are SCHEDULED — known months ahead — so tilting into them is NOT look-ahead.
Here we add a FOMC-window leverage boost on top of the risk-managed equity overlay (vol-
target + trend) and check it improves return/Sharpe net of the extra turnover.

  base       : vol-target(63)+trend(200) overlay on NQ, 15% target, cap 3x  (the survivor)
  base+FOMC  : same, but +1.0x extra long on day -1 and announcement day (cap 4x)
  buy&hold   : raw NQ

Run:  python examples/13_fomc_overlay.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"
VOL_WIN, TREND, TARGET, MAXLEV, COST = 63, 200, 0.15, 3.0, 0.0005
FOMC_BOOST, CAP_TILT = 1.0, 4.0


def base_positions(ret, px):
    vol = ret.rolling(VOL_WIN).std() * np.sqrt(252)
    sig = (px > px.rolling(TREND).mean()).astype(float)
    desired = ((TARGET / vol).clip(upper=MAXLEV) * sig).shift(1).fillna(0.0)
    pos, cur, pm = [], 0.0, None
    for d, v in desired.items():
        if pm is None or d.month != pm.month:
            cur = v
        pos.append(cur); pm = d
    return pd.Series(pos, index=desired.index)


def fomc_window_mask(idx):
    """True on trading day -1 and day 0 of each scheduled FOMC announcement (causal: dates known ahead)."""
    f = pd.read_csv(os.path.join(DATA_DIR, "FOMC_dates.csv"), parse_dates=["fomc_date"])["fomc_date"]
    f = f[(f >= idx.min()) & (f <= idx.max())].sort_values()
    kept, last = [], None
    for d in f:                                    # scheduled-only (drop emergency intermeeting)
        if last is None or (d - last).days >= 20:
            kept.append(d); last = d
    mask = pd.Series(False, index=idx)
    locs = idx.get_indexer(pd.DatetimeIndex(kept), method="bfill")
    for p in locs:
        if 1 <= p < len(idx):
            mask.iloc[p - 1] = True; mask.iloc[p] = True
    return mask


def perf(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252),
            "maxDD": float((eq / eq.cummax() - 1).min()), "term": float(eq.iloc[-1])}


def main():
    px = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"),
                     parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    ret = px.pct_change().dropna()
    base = base_positions(ret, px).reindex(ret.index).fillna(0.0)
    win = fomc_window_mask(ret.index)
    boosted = base.where(~win, np.minimum(base + FOMC_BOOST, CAP_TILT))

    def net(pos):
        return pos * ret - COST * pos.diff().abs().fillna(0.0)

    results = {"buy&hold": perf(ret), "base overlay": perf(net(base)),
               "base + FOMC tilt": perf(net(boosted))}
    print(f"NQ {ret.index.min().date()}..{ret.index.max().date()}  "
          f"FOMC-window days: {int(win.sum())} ({win.mean()*100:.0f}% of time)\n")
    print(f"{'strategy':<20}{'CAGR':>8}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}{'terminal':>10}")
    for k, p in results.items():
        print(f"{k:<20}{p['CAGR']*100:>7.1f}%{p['vol']*100:>6.0f}%{p['Sharpe']:>8.2f}"
              f"{p['maxDD']*100:>7.0f}%{p['term']:>9.1f}x")

    # isolate the tilt's own contribution
    tilt_only = net(boosted) - net(base)
    eq = (1 + tilt_only).cumprod(); yrs = (ret.index[-1] - ret.index[0]).days / 365.25
    print(f"\n  FOMC tilt incremental: {(eq.iloc[-1]**(1/yrs)-1)*100:+.1f}%/yr, "
          f"t-stat {tilt_only.mean()/(tilt_only.std()/np.sqrt(len(tilt_only))):.1f}")
    base_p, boost_p = results["base overlay"], results["base + FOMC tilt"]
    print(f"  net effect: CAGR {base_p['CAGR']*100:.1f}% -> {boost_p['CAGR']*100:.1f}%, "
          f"Sharpe {base_p['Sharpe']:.2f} -> {boost_p['Sharpe']:.2f}, "
          f"maxDD {base_p['maxDD']*100:.0f}% -> {boost_p['maxDD']*100:.0f}%")
    print("\n  Scheduled FOMC dates are known ex-ante, so this tilt has NO look-ahead. The drift")
    print("  is real but small; as a tilt it nudges return for a little extra FOMC-day risk.")


if __name__ == "__main__":
    main()
