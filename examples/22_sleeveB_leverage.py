"""
Example 22 — leverage on Strategy B (short-vol), where skew makes it lethal.

Sharpe says "lever it up"; skew says "you'll be ruined". Short-vol is NEGATIVELY skewed, so the
growth-optimal (Kelly) leverage is far LOWER than the Sharpe implies, and past a threshold a
SINGLE tail day takes equity to zero. We sweep leverage on Sleeve B at 10%-vol base, for the
TAIL-HEDGED book version and the UNHEDGED raw short-vol (to show why the wings matter for leverage).

Run:  python examples/22_sleeveB_leverage.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"
SLEEVE_VOL = 0.10


def short_vol(df, vxn, hedged):
    ret = df["Close"].pct_change(); idx = ret.index.intersection(vxn.index)
    ret, vxn = ret.loc[idx], vxn.loc[idx]
    iv = (vxn.shift(1) / 100) ** 2 / 252; rvar = ret ** 2
    if hedged:
        capv = 0.05 ** 2; wing = np.maximum(rvar - capv, 0).mean()
        pnl = iv - np.minimum(rvar, capv) - wing
    else:
        pnl = iv - rvar
    pnl = pnl.dropna()
    return pnl * (SLEEVE_VOL / (pnl.std() * np.sqrt(252)))      # scale to 10% vol base


def lever_stats(r, L):
    lr = L * r
    if lr.min() <= -1.0:                                         # a day wipes out the account
        first = lr[lr <= -1.0].index[0]
        return {"vol": lr.std() * np.sqrt(252), "CAGR": -1.0, "Sharpe": lr.mean()/lr.std()*np.sqrt(252),
                "maxDD": -1.0, "skew": float(lr.skew()), "worst": float(lr.min()),
                "term": 0.0, "status": f"RUINED {first.date()}"}
    eq = (1 + lr).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    return {"vol": lr.std() * np.sqrt(252), "CAGR": eq.iloc[-1] ** (1/yrs) - 1,
            "Sharpe": lr.mean()/lr.std()*np.sqrt(252), "maxDD": float((eq/eq.cummax()-1).min()),
            "skew": float(lr.skew()), "worst": float(lr.min()), "term": float(eq.iloc[-1]), "status": "ok"}


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
    vxn = pd.read_csv(os.path.join(DATA_DIR, "VXN_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    hedged = short_vol(df, vxn, True)
    unhedged = short_vol(df, vxn, False)

    for name, r in [("TAIL-HEDGED short-vol (the book's Sleeve B)", hedged),
                    ("UNHEDGED raw short-vol (no wings)", unhedged)]:
        # empirical growth-optimal (Kelly) leverage
        grid = np.arange(0.25, 12.01, 0.25)
        cagrs = [lever_stats(r, L)["CAGR"] for L in grid]
        kstar = grid[int(np.argmax(cagrs))]
        ruin_L = 1.0 / abs(r.min())                              # L where the worst day = -100%
        print("=" * 92)
        print(f"{name}   base 10% vol, Sharpe {r.mean()/r.std()*np.sqrt(252):.2f}, skew {r.skew():.1f}")
        print(f"  growth-optimal (Kelly) leverage ~ {kstar:.1f}x ;  RUIN at ~{ruin_L:.1f}x "
              f"(one day = -100%); worst base day {r.min()*100:.1f}%")
        print("=" * 92)
        print(f"  {'lever':>6}{'vol':>7}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>8}{'skew':>7}{'worstday':>10}{'terminal':>11}{'  status':<14}")
        for L in [0.75, 1, 2, 3, 5, 8]:
            s = lever_stats(r, L)
            term = f"{s['term']:.1f}x" if s['term'] > 0 else "0 (ruin)"
            print(f"  {L:>5.2f}x{s['vol']*100:>6.0f}%{s['CAGR']*100:>8.1f}%{s['Sharpe']:>8.2f}"
                  f"{s['maxDD']*100:>7.0f}%{s['skew']:>7.1f}{s['worst']*100:>9.1f}%{term:>11}  {s['status']:<14}")
        print()

    print("Read: short-vol Sharpe is high, so Sharpe-sizing screams 'lever up' — but the NEGATIVE")
    print("SKEW means the growth-optimal leverage is modest and a single tail day past the ruin")
    print("threshold zeroes the account. The book runs Sleeve B at ~0.75x (50% wt x 1.5x book) —")
    print("far below Kelly and far below ruin, which is the ONLY responsible way to size a")
    print("negatively-skewed sleeve. Unhedged, even low leverage courts ruin — hence the wings.")


if __name__ == "__main__":
    main()
