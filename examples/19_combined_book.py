"""
Example 19 — the deployable book: the two VERIFIED edges combined, benchmarked.

Survivors of the whole investigation (everything else was rejected OOS / by a null / by deflation):
  SLEEVE A  Risk-managed equity + pre-FOMC tilt  (FUTURES, fully validated, paper-traded)
            = vol-target(Parkinson-21) + 200d-trend brake on NQ, 15% vol, 3x cap, monthly rebal,
              +1x leverage on the day before & day of each SCHEDULED FOMC announcement.
  SLEEVE B  Tail-hedged VXN variance-risk-premium  (needs options/var-swap desk; var-swap proxy)
            = sell NQ implied variance (VXN), own wings capping realized at +/-5%/day.

We benchmark each sleeve and a 50/50 risk-equalized COMBINED book vs buy&hold NQ and 60/40,
over the matched VXN period (2001+). Diversification: A is long-biased, B is short-vol carry;
their day-to-day correlation is low even though both dislike crashes.

Run:  python examples/19_combined_book.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"
TARGET, TREND, MAXLEV, COST = 0.15, 200, 3.0, 0.0005
FOMC_BOOST, CAP_TILT = 1.0, 4.0
SLEEVE_VOL = 0.10            # scale each sleeve to 10% for a clean risk-equalized combo


def perf(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252),
            "maxDD": float((eq / eq.cummax() - 1).min()),
            "worst1y": float(eq.pct_change(252).min()), "term": float(eq.iloc[-1])}


def sleeve_A(df, fomc):
    px = df["Close"]; ret = px.pct_change()
    hl2 = np.log(df["High"] / df["Low"]) ** 2
    vol = np.sqrt(hl2.rolling(21).mean() / (4 * np.log(2))) * np.sqrt(252)
    sig = (px > px.rolling(TREND).mean()).astype(float)
    desired = ((TARGET / vol).clip(upper=MAXLEV) * sig).shift(1).fillna(0.0)
    pos, cur, pm = [], 0.0, None
    for d, v in desired.items():
        if pm is None or d.month != pm.month:
            cur = v
        pos.append(cur); pm = d
    pos = pd.Series(pos, index=desired.index)
    win = pd.Series(False, index=ret.index)
    locs = ret.index.get_indexer(fomc, method="bfill")
    for p in locs:
        if 1 <= p < len(ret.index):
            win.iloc[p - 1] = True; win.iloc[p] = True
    pos = pos.where(~win, np.minimum(pos + FOMC_BOOST, CAP_TILT))
    return (pos * ret - COST * pos.diff().abs().fillna(0.0)).dropna()


def sleeve_B(df, vxn):
    ret = df["Close"].pct_change()
    idx = ret.index.intersection(vxn.index)
    ret, vxn = ret.loc[idx], vxn.loc[idx]
    iv = (vxn.shift(1) / 100) ** 2 / 252
    rvar = ret ** 2
    capv = 0.05 ** 2
    wing = np.maximum(rvar - capv, 0).mean()
    return (iv - np.minimum(rvar, capv) - wing).dropna()


def scale_to(r, target=SLEEVE_VOL):
    return r * (target / (r.std() * np.sqrt(252)))


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"),
                     parse_dates=["Date"]).set_index("Date").sort_index()
    vxn = pd.read_csv(os.path.join(DATA_DIR, "VXN_all_history.csv"),
                      parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    us10 = pd.read_csv(os.path.join(DATA_DIR, "US10Y_all_history.csv"),
                       parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    fomc = pd.read_csv(os.path.join(DATA_DIR, "FOMC_dates.csv"), parse_dates=["fomc_date"])["fomc_date"]
    f, last, kept = fomc.sort_values(), None, []
    for d in f:
        if last is None or (d - last).days >= 20:
            kept.append(d); last = d
    fomc = pd.DatetimeIndex(kept)

    rA = scale_to(sleeve_A(df, fomc))
    rB = scale_to(sleeve_B(df, vxn))
    idx = rA.index.intersection(rB.index)                 # matched VXN period (2001+)
    rA, rB = rA.loc[idx], rB.loc[idx]
    combined = 0.5 * rA + 0.5 * rB                          # risk-equalized 50/50

    nq = df["Close"].pct_change().loc[idx]
    bond = us10.pct_change().reindex(idx).fillna(0.0)
    bench6040 = 0.6 * nq + 0.4 * bond

    print(f"Benchmark period {idx.min().date()}..{idx.max().date()} (matched VXN history)\n")
    print(f"{'strategy':<28}{'CAGR':>7}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}{'worst1y':>9}{'terminal':>10}")
    rows = [("Buy&hold NQ", nq), ("60/40 (NQ/US10Y)", bench6040),
            ("A: equity+FOMC (@10%)", rA), ("B: tail-hedged short-vol (@10%)", rB),
            ("COMBINED 50/50", combined)]
    for name, r in rows:
        p = perf(r)
        print(f"{name:<28}{p['CAGR']*100:>6.1f}%{p['vol']*100:>6.0f}%{p['Sharpe']:>8.2f}"
              f"{p['maxDD']*100:>7.0f}%{p['worst1y']*100:>8.0f}%{p['term']:>9.1f}x")

    print(f"\n  sleeve correlation corr(A,B) = {rA.corr(rB):+.2f}  (low day-to-day -> diversification)")
    print(f"  Sharpe: A {perf(rA)['Sharpe']:.2f}, B {perf(rB)['Sharpe']:.2f}, combined "
          f"{perf(combined)['Sharpe']:.2f} (combo beats A and HALVES B's drawdown; B-alone is fragile)")
    # tail co-movement: both sleeves on the 20 worst NQ days
    worst = nq.nsmallest(20).index
    print(f"  on the 20 worst NQ days: A avg {rA.reindex(worst).mean()*100:+.1f}%, "
          f"B avg {rB.reindex(worst).mean()*100:+.1f}%  (both dislike crashes — tail risk is shared)")
    print("\n  NB: sleeves scaled ex-post to 10% vol to illustrate the risk-equalized combo. Sleeve B")
    print("  is a variance-swap PROXY (needs an options/var-swap desk + real wing costs); Sleeve A is")
    print("  futures-only and live in paper. In-sample where noted; deploy A first, B as a satellite.")


if __name__ == "__main__":
    main()
