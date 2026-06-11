"""
Example 10 — does MORE LEVERAGE buy more return on the equity overlay?

The overlay has a higher Sharpe than buy-&-hold, so levering it SHOULD beat buy-&-hold's
return at equal risk. But leverage is not free and not linear:
  * compound growth ~ arithmetic_return - variance/2, and variance scales with L^2, so
    realized CAGR is CONCAVE in leverage: it peaks at the growth-optimal (Kelly) leverage,
    then DECLINES. Past the peak, more leverage => less money AND far deeper drawdowns.
  * financing: the borrowed portion (L-1) costs ~the short rate; we charge rf on it.
  * the drawdown that was the whole point of the overlay scales ~linearly with L.

We sweep a leverage multiplier on the base 15%-vol overlay (NQ) and show the curve, the
growth-optimal point, and the drawdown cost — vs buy-&-hold.

Run:  python examples/10_leverage.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"
RF = 0.03                      # assumed financing / short rate on the borrowed portion
VOL_WIN, TREND, TARGET, MAXLEV, COST = 63, 200, 0.15, 3.0, 0.0005


def overlay_returns(px):
    ret = px.pct_change(fill_method=None).dropna()
    vol = ret.rolling(VOL_WIN).std() * np.sqrt(252)
    sig = (px > px.rolling(TREND).mean()).astype(float)
    desired = ((TARGET / vol).clip(upper=MAXLEV) * sig).shift(1).fillna(0.0)
    pos, cur, pm = [], 0.0, None
    for d, v in desired.items():
        if pm is None or d.month != pm.month:
            cur = v
        pos.append(cur); pm = d
    pos = pd.Series(pos, index=desired.index)
    return (pos * ret - COST * pos.diff().abs().fillna(0.0)).dropna(), ret


def stats(r):
    eq = (1 + r).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    return (eq.iloc[-1] ** (1 / yrs) - 1, r.std() * np.sqrt(252),
            r.mean() / r.std() * np.sqrt(252), float((eq / eq.cummax() - 1).min()), float(eq.iloc[-1]))


def main():
    px = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"),
                     parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    base, raw = overlay_returns(px)
    bh = stats(raw.loc[base.index])

    # growth-optimal (Kelly) multiple on the base overlay: f* = (mean - rf)/variance
    mu = base.mean() * 252; var = (base.std() * np.sqrt(252)) ** 2
    kelly = (mu - RF) / var

    print(f"NQ overlay (vol{VOL_WIN}+trend{TREND}, {TARGET:.0%} target), {base.index.min().date()}.."
          f"{base.index.max().date()}   financing rf={RF:.0%}")
    print(f"buy&hold NQ: CAGR {bh[0]*100:.1f}%  vol {bh[1]*100:.0f}%  Sharpe {bh[2]:.2f}  maxDD {bh[3]*100:.0f}%\n")
    print(f"{'leverage':>9}{'~peakLev':>10}{'CAGR':>8}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}"
          f"{'terminal':>10}{'CAGR no-fin':>13}")
    rows = []
    for L in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        lev = L * base - (L - 1) * RF / 252           # finance the borrowed part
        c, v, s, dd, term = stats(lev)
        c_nofin = stats(L * base)[0]
        rows.append((L, c, dd))
        print(f"{L:>9.1f}{L*MAXLEV:>9.1f}x{c*100:>7.1f}%{v*100:>6.0f}%{s:>8.2f}{dd*100:>7.0f}%"
              f"{term:>9.1f}x{c_nofin*100:>12.1f}%")

    best = max(rows, key=lambda r: r[1])
    print(f"\n  growth-optimal (Kelly) multiple ~ {kelly:.1f}x the base overlay  "
          f"(half-Kelly {kelly/2:.1f}x is the prudent choice)")
    print(f"  empirical CAGR peaks near L={best[0]:.1f} (CAGR {best[1]*100:.1f}%, maxDD {best[2]*100:.0f}%)")
    print( "  beyond the peak, leverage REDUCES compound return (vol drag) while drawdown keeps")
    print( "  scaling ~linearly toward ruin. Note maxDD here uses the 63d vol estimate, which LAGS")
    print( "  overnight crashes — real levered drawdowns in a gap (e.g. Mar-2020) would be worse.")


if __name__ == "__main__":
    main()
