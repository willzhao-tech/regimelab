"""
Example 9 — transaction-cost & capacity stress on the SURVIVING strategy only.

The only thing that survived OOS cross-asset validation (08) was the equity-index
DRAWDOWN OVERLAY: vol-target + 200d-trend brake on equities (NQ, A50). Before it can
hold real money we must know (a) how fast costs eat it, and (b) how much AUM it can run.

  COST STRESS  : sweep 0 -> 50 bps per unit turnover; report net CAGR / Sharpe / maxDD /
                 terminal and the cost DRAG, plus the break-even cost where the overlay's
                 net return falls below plain buy-&-hold (i.e. you're overpaying for the
                 drawdown protection).
  CAPACITY     : from real traded VOLUME, estimate average-daily-volume notional, the
                 strategy's turnover, and the max AUM at 1% / 10% participation.

All causal, monthly leverage rebal, leverage capped 3x, 15% vol target.

Run:  python examples/09_cost_capacity.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"
# equity indices only (what survived OOS) + futures point-multipliers for notional sizing
EQUITY = {"NQ": ("NQ_F_all_history.csv", 20.0),    # e-mini NQ = $20 / index point
          "A50": ("A50_all_history.csv", 1.0)}     # SGX FTSE China A50 = US$1 / pt (approx)
TARGET, VOL_WIN, TREND, MAXLEV = 0.15, 63, 200, 3.0


def perf(r):
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return {"sharpe": np.nan, "cagr": np.nan, "maxdd": np.nan, "term": np.nan}
    eq = (1 + r).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    return {"sharpe": r.mean() / r.std() * np.sqrt(252), "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
            "maxdd": float((eq / eq.cummax() - 1).min()), "term": float(eq.iloc[-1])}


def positions(ret, price, vol_win=VOL_WIN, trend=TREND, target=TARGET, max_lev=MAXLEV):
    """Causal target position (leverage) series; monthly rebal."""
    vol = ret.rolling(vol_win).std() * np.sqrt(252)
    sig = (price > price.rolling(trend).mean()).astype(float)
    desired = ((target / vol).clip(upper=max_lev) * sig).shift(1).fillna(0.0)
    pos, cur, pm = [], 0.0, None
    for d, v in desired.items():
        if pm is None or d.month != pm.month:
            cur = v
        pos.append(cur); pm = d
    return pd.Series(pos, index=desired.index)


def net_returns(ret, pos, cost_bps):
    return pos * ret - (cost_bps / 1e4) * pos.diff().abs().fillna(0.0)


def main():
    for name, (fn, mult) in EQUITY.items():
        df = pd.read_csv(os.path.join(DATA_DIR, fn), parse_dates=["Date"]).set_index("Date").sort_index()
        px = df["Close"].dropna()
        ret = px.pct_change(fill_method=None).dropna()
        pos = positions(ret, px).reindex(ret.index).fillna(0.0)
        bh = perf(ret)
        ann_turnover = pos.diff().abs().sum() / ((ret.index[-1] - ret.index[0]).days / 365.25)

        print("=" * 96)
        print(f"{name}  equity drawdown-overlay (vol{VOL_WIN}+trend{TREND}, target {TARGET:.0%})  "
              f"buy&hold: CAGR {bh['cagr']*100:.1f}%  maxDD {bh['maxdd']*100:.0f}%  Sharpe {bh['sharpe']:.2f}")
        print("=" * 96)
        print("  COST STRESS")
        print(f"  {'cost(bps)':>10}{'net CAGR':>10}{'Sharpe':>9}{'maxDD':>8}{'terminal':>10}{'cost drag':>11}")
        gross_cagr = None; be = None; prev = None
        for c in [0, 1, 2, 5, 10, 20, 50]:
            p = perf(net_returns(ret, pos, c))
            if c == 0:
                gross_cagr = p["cagr"]
            drag = (gross_cagr - p["cagr"]) * 100
            print(f"  {c:>10}{p['cagr']*100:>9.1f}%{p['sharpe']:>9.2f}{p['maxdd']*100:>7.0f}%"
                  f"{p['term']:>9.2f}x{drag:>10.1f}%")
            # break-even vs buy&hold CAGR
            if prev is not None and be is None:
                c0, g0 = prev; g1 = p["cagr"]
                if (g0 - bh["cagr"]) * (g1 - bh["cagr"]) < 0:
                    be = c0 + (c - c0) * (g0 - bh["cagr"]) / (g0 - g1)
            prev = (c, p["cagr"])
        print(f"  annualized turnover: {ann_turnover:.1f}x   "
              f"break-even vs buy&hold CAGR: {'%.0f bps' % be if be else '>50 bps (or never beats B&H on CAGR)'}")

        # ---- capacity from real volume -----------------------------------------
        # the book only trades on ~monthly rebalances; size capacity off the actual
        # rebalance trades (nonzero position changes), executed in ONE day (conservative).
        vol_recent = df["Volume"].dropna().tail(252)
        px_recent = px.reindex(vol_recent.index)
        adv_notional = float((vol_recent * mult * px_recent).median())   # $/day median ADV
        chg = pos.diff().abs()
        trades = chg[chg > 1e-9]                                          # per-$1 rebalance trade sizes
        trade_p95 = float(trades.quantile(0.95)); trade_avg = float(trades.mean())
        print("\n  CAPACITY (from real traded volume; trade = a monthly rebalance, 1-day execution)")
        print(f"  median ADV notional ~ ${adv_notional/1e9:.1f}B/day   "
              f"rebalance trade size: avg {trade_avg*100:.0f}% of book, p95 {trade_p95*100:.0f}%")

        def fmt(x):
            return f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.0f}M"
        for part in (0.01, 0.10):
            cap = part * adv_notional / max(trade_p95, 1e-9)             # AUM so p95 trade <= part*ADV
            print(f"   at {part*100:>2.0f}% of ADV participation -> max AUM ~ {fmt(cap)}"
                  f"  (working the trade over the ~21-day month multiplies this ~20x)")
        print()


if __name__ == "__main__":
    main()
