"""
Example 17 — Parkinson range vol as the risk estimator in the vol-targeting overlay.

The overlay (07-09) sizes positions off close-to-close rolling std. The Parkinson range
estimator (from H-L) is ~5x more statistically efficient, so for the SAME smoothness it can
use a shorter window -> faster de-risking when vol spikes. Test on NQ, equal everything else
(vol-target 15%, trend200, 3x cap, monthly rebal, 5bps), three estimators:
    close_63      : close-to-close std, 63d        (the incumbent)
    parkinson_63  : Parkinson range vol, 63d        (same window, cleaner)
    parkinson_21  : Parkinson range vol, 21d        (faster — efficiency 'spent' on speed)

Run:  python examples/17_parkinson_overlay.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"
TARGET, TREND, MAXLEV, COST = 0.15, 200, 3.0, 0.0005


def close_vol(df, win):
    return df["Close"].pct_change().rolling(win).std() * np.sqrt(252)


def parkinson_vol(df, win):
    hl2 = np.log(df["High"] / df["Low"]) ** 2
    return np.sqrt(hl2.rolling(win).mean() / (4 * np.log(2))) * np.sqrt(252)


def overlay(df, vol, target=TARGET):
    px = df["Close"]; ret = px.pct_change()
    sig = (px > px.rolling(TREND).mean()).astype(float)
    desired = ((target / vol).clip(upper=MAXLEV) * sig).shift(1).fillna(0.0)
    pos, cur, pm = [], 0.0, None
    for d, v in desired.items():
        if pm is None or d.month != pm.month:
            cur = v
        pos.append(cur); pm = d
    pos = pd.Series(pos, index=desired.index)
    r = (pos * ret - COST * pos.diff().abs().fillna(0.0)).dropna()
    return r, pos


def perf(r):
    eq = (1 + r).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252),
            "maxDD": float((eq / eq.cummax() - 1).min()), "term": float(eq.iloc[-1])}


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"),
                     parse_dates=["Date"]).set_index("Date").sort_index()
    bh = df["Close"].pct_change().dropna()
    ebh = (1 + bh).cumprod()
    print(f"NQ {df.index.min().date()}..{df.index.max().date()}")
    print(f"buy&hold: CAGR {(ebh.iloc[-1]**(365.25/(df.index[-1]-df.index[0]).days)-1)*100:.1f}%  "
          f"Sharpe {bh.mean()/bh.std()*np.sqrt(252):.2f}  maxDD {((ebh/ebh.cummax()-1).min())*100:.0f}%\n")
    configs = {"close_63": close_vol(df, 63), "parkinson_63": parkinson_vol(df, 63),
               "parkinson_21": parkinson_vol(df, 21)}
    print(f"{'estimator':<16}{'CAGR':>8}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}{'terminal':>10}{'turnover/yr':>13}")
    out = {}
    for name, vol in configs.items():
        r, pos = overlay(df, vol)
        p = perf(r); out[name] = (r, p)
        turn = pos.diff().abs().sum() / ((r.index[-1] - r.index[0]).days / 365.25)
        print(f"{name:<16}{p['CAGR']*100:>7.1f}%{p['vol']*100:>6.0f}%{p['Sharpe']:>8.2f}"
              f"{p['maxDD']*100:>7.0f}%{p['term']:>9.1f}x{turn:>12.1f}x")

    # crash-response check: average leverage going INTO the worst buy&hold months
    worst = bh.resample("ME").sum().nsmallest(6).index
    print("\n  Leverage held during the 6 worst buy&hold months (lower = de-risked faster):")
    for name, vol in configs.items():
        _, pos = overlay(df, vol)
        lev = np.mean([pos[pos.index.to_period("M") == w.to_period("M")].mean() for w in worst])
        print(f"    {name:<16}{lev:.2f}x avg leverage")
    print("\n  Read: Parkinson is a cleaner estimator; the short-window version reacts faster to")
    print("  vol spikes. Compare Sharpe/maxDD vs the close-63 incumbent — equal everything else.")


if __name__ == "__main__":
    main()
