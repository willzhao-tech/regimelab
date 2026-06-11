"""
Example 16 — NQ daily range volatility (High-Low)/Open: statistics + a vol-conditional strategy.

(H-L)/O is a Parkinson-style intraday range/volatility proxy. We:
  1. characterize it fully (moments, percentiles, fat tails, clustering/persistence, the
     leverage effect, the variance-risk-premium vs VIX);
  2. ask what it PREDICTS — next-day NQ return conditional on today's range (causal);
  3. build and HONESTLY test a strategy from that (causal, costs, vs buy&hold, in/out-sample).

NB: a single liquid future has no riskless arbitrage. This is a statistical/vol-conditional
strategy, tested with the same gauntlet as everything else — not a free lunch.

Run:  python examples/16_nq_range_vol.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"


def perf(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan,
            "maxDD": float((eq / eq.cummax() - 1).min()), "term": float(eq.iloc[-1])}


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"),
                     parse_dates=["Date"]).set_index("Date").sort_index()
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    rv = ((h - l) / o).dropna()
    ret = c.pct_change().dropna()
    rv, ret = rv.align(ret, join="inner")

    # ---- 1) statistics --------------------------------------------------------
    print(f"NQ daily range vol (H-L)/O — {rv.index.min().date()}..{rv.index.max().date()} ({len(rv)} days)")
    print("=" * 74)
    q = rv.quantile([.01, .05, .10, .25, .50, .75, .90, .95, .99])
    print(f"  mean {rv.mean()*100:.2f}%  median {rv.median()*100:.2f}%  std {rv.std()*100:.2f}%  "
          f"min {rv.min()*100:.2f}%  max {rv.max()*100:.1f}%")
    print(f"  skew {rv.skew():.2f}  kurtosis {rv.kurt():.1f}  (right-skewed, fat-tailed)")
    print("  percentiles:  " + "  ".join(f"p{int(p*100)}={v*100:.2f}%" for p, v in q.items()))
    # Parkinson annualized vol estimator
    park = np.sqrt((np.log(h / l) ** 2).mean() / (4 * np.log(2))) * np.sqrt(252)
    cc = ret.std() * np.sqrt(252)
    print(f"  Parkinson ann vol {park*100:.1f}%  vs close-close ann vol {cc*100:.1f}%  "
          f"(range estimator is more efficient)")
    print(f"  persistence (autocorr): lag1 {rv.autocorr(1):.2f}  lag5 {rv.autocorr(5):.2f}  "
          f"lag20 {rv.autocorr(20):.2f}  -> strong vol clustering")
    print(f"  leverage effect corr(range, same-day return) = {rv.corr(ret):+.2f}  "
          f"(high range days are down days)")
    # variance risk premium vs VIX
    try:
        vix = pd.read_csv(os.path.join(DATA_DIR, "VIX_all_history.csv"),
                          parse_dates=["Date"]).set_index("Date")["Close"].dropna()
        v = vix.reindex(rv.index).dropna()
        # annualize range to vol-points comparable to VIX: Parkinson daily -> annual %
        rng_ann = (rv / np.sqrt(4 * np.log(2))) * np.sqrt(252) * 100
        common = rng_ann.index.intersection(v.index)
        vrp = (v.reindex(common) - rng_ann.reindex(common))
        print(f"  variance-risk-premium: VIX(implied) {v.reindex(common).mean():.1f} vs "
              f"realized-range {rng_ann.reindex(common).mean():.1f} -> VRP {vrp.mean():+.1f} vol pts "
              f"(implied richer {100*(vrp>0).mean():.0f}% of days)")
    except Exception as e:
        print("  (VIX VRP skipped:", type(e).__name__, ")")

    # ---- 2) what does range PREDICT? next-day return by today's range decile ----
    print("\n" + "=" * 74)
    print("PREDICTIVE: next-day NQ return by today's range decile (causal)")
    print("=" * 74)
    rank = rv.rolling(252).rank(pct=True)              # trailing percentile (no look-ahead)
    nxt = ret.shift(-1)
    dec = pd.qcut(rank.dropna(), 10, labels=False)
    tbl = pd.DataFrame({"dec": dec, "nxt": nxt.reindex(dec.index)}).dropna()
    g = tbl.groupby("dec")["nxt"]
    print(f"  {'range decile':>12}{'next-day mean bps':>20}{'t-stat':>9}")
    for d, s in g:
        t = s.mean() / (s.std() / np.sqrt(len(s)))
        lab = "low vol" if d == 0 else ("HIGH vol" if d == 9 else "")
        print(f"  {int(d)+1:>12}{s.mean()*1e4:>19.1f}{t:>9.1f}  {lab}")

    # ---- 3) strategy: long NQ the day AFTER a top-decile range day (mean-reversion) ----
    print("\n" + "=" * 74)
    print("STRATEGY — vol-spike reversal: long NQ on the day after a top-range day, else flat")
    print("=" * 74)
    signal = (rank >= 0.90).astype(float).shift(1).fillna(0.0)   # causal: yesterday's range
    COST = 0.0001
    strat = signal * ret - COST * signal.diff().abs().fillna(0.0)
    bh = ret
    print(f"  {'strategy':<26}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'%days in':>10}{'terminal':>10}")
    for name, r, inmask in [("buy&hold", bh, None), ("vol-spike reversal", strat, signal)]:
        p = perf(r); inpct = f"{signal.mean()*100:.0f}%" if inmask is not None else "100%"
        print(f"  {name:<26}{p['CAGR']*100:>7.1f}%{p['Sharpe']:>8.2f}{p['maxDD']*100:>7.0f}%"
              f"{inpct:>10}{p['term']:>9.1f}x")
    # in-sample vs out-of-sample split
    cut = strat.index[len(strat) // 2]
    print(f"\n  split at {cut.date()}:  in-sample Sharpe {perf(strat[:cut])['Sharpe']:.2f}  "
          f"out-of-sample Sharpe {perf(strat[cut:])['Sharpe']:.2f}  "
          f"(buy&hold OOS {perf(bh[cut:])['Sharpe']:.2f})")
    print("\n  Read: range is a strong, persistent VOL signal (clusters, predicts future vol) but a")
    print("  weak RETURN signal. Any 'reversal' edge is small, concentrated in crises, and the real")
    print("  use of (H-L)/O is risk SIZING / a VRP short-vol case, not directional arbitrage.")


if __name__ == "__main__":
    main()
