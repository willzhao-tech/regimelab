"""
Example 21 — realistic dollar-accounting simulation of the book (FOMC increment fix).

Differences vs the return-series backtest (20):
  * FOMC tilt is a MEASURED +0.5x tactical overlay (not +1x doubling) — ~halves FOMC turnover/cost.
  * Real environment: $10M start, equity COMPOUNDS, Sleeve A held in INTEGER NQ contracts sized
    off current equity (so position scales as the book grows), daily mark-to-market in dollars,
    costs on actual contract changes.
  * Outputs the position allocation (contracts + $ exposure per sleeve), the equity curve, and
    the benchmark — all in dollars.

Sleeve B (tail-hedged VXN short-vol) is a modeled carry stream (variance-swap proxy), allocated a
fixed risk budget. Book = 1.5x * (50% A + 50% B), each sleeve risk-equalized to 10% vol.

Run:  python examples/21_real_sim.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"
TARGET, TREND, MAXLEV, COST = 0.15, 200, 3.0, 0.0005
FOMC_BOOST, CAP_TILT = 0.5, 4.0          # <-- measured +0.5x tactical overlay (the fix)
SLEEVE_VOL, BOOK_LEV = 0.10, 1.5
E0, MULT = 10_000_000, 20                 # $10M start, NQ $20/pt


def perf(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252),
            "maxDD": float((eq / eq.cummax() - 1).min())}


def sleeve_A(df, fomc, boost):
    px = df["Close"]; ret = px.pct_change()
    hl2 = np.log(df["High"] / df["Low"]) ** 2
    vol = np.sqrt(hl2.rolling(21).mean() / (4 * np.log(2))) * np.sqrt(252)
    sig = (px > px.rolling(TREND).mean()).astype(float)
    desired = ((TARGET / vol).clip(upper=MAXLEV) * sig).shift(1).fillna(0.0)
    base, cur, pm = [], 0.0, None
    for d, v in desired.items():
        if pm is None or d.month != pm.month:
            cur = v
        base.append(cur); pm = d
    base = pd.Series(base, index=desired.index)
    win = pd.Series(False, index=ret.index)
    for p in ret.index.get_indexer(fomc, method="bfill"):
        if 1 <= p < len(ret.index):
            win.iloc[p - 1] = True; win.iloc[p] = True
    pos = base.where(~win, np.minimum(base + boost, CAP_TILT))
    rA = (pos * ret - COST * pos.diff().abs().fillna(0.0)).dropna()
    return rA, pos


def sleeve_B(df, vxn):
    ret = df["Close"].pct_change(); idx = ret.index.intersection(vxn.index)
    ret, vxn = ret.loc[idx], vxn.loc[idx]
    iv = (vxn.shift(1) / 100) ** 2 / 252; rvar = ret ** 2; capv = 0.05 ** 2
    wing = np.maximum(rvar - capv, 0).mean()
    return (iv - np.minimum(rvar, capv) - wing).dropna()


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
    vxn = pd.read_csv(os.path.join(DATA_DIR, "VXN_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    us10 = pd.read_csv(os.path.join(DATA_DIR, "US10Y_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    fomc = pd.read_csv(os.path.join(DATA_DIR, "FOMC_dates.csv"), parse_dates=["fomc_date"])["fomc_date"].sort_values()
    kept, last = [], None
    for d in fomc:
        if last is None or (d - last).days >= 20:
            kept.append(d); last = d
    fomc = pd.DatetimeIndex(kept)

    rA_raw, posA = sleeve_A(df, fomc, FOMC_BOOST)
    fA = SLEEVE_VOL / (rA_raw.std() * np.sqrt(252))      # scale Sleeve A to 10% vol
    rB_raw = sleeve_B(df, vxn); fB = SLEEVE_VOL / (rB_raw.std() * np.sqrt(252))
    rA, rB = fA * rA_raw, fB * rB_raw
    idx = rA.index.intersection(rB.index)
    rA, rB = rA.loc[idx], rB.loc[idx]
    book_ret = BOOK_LEV * (0.5 * rA + 0.5 * rB)

    # ---- realistic dollar simulation (compounding equity, integer contracts) ----
    px = df["Close"].reindex(idx)
    E = E0 * (1 + book_ret).cumprod()                    # book equity ($)
    nq_lev = BOOK_LEV * 0.5 * fA * posA.reindex(idx)     # effective NQ leverage in the book
    contracts = (nq_lev * E / (px * MULT)).round()
    exposureA = contracts * px * MULT                    # $ NQ exposure (Sleeve A)
    fomc_days = posA.reindex(idx) > 0                     # flag (for annotation)

    nq_eq = E0 * (1 + df["Close"].pct_change().loc[idx]).cumprod()
    s6040 = 0.6 * df["Close"].pct_change().loc[idx] + 0.4 * us10.pct_change().reindex(idx).fillna(0)
    eq6040 = E0 * (1 + s6040).cumprod()

    print(f"REAL-ENVIRONMENT SIM  {idx.min().date()}..{idx.max().date()}  start ${E0/1e6:.0f}M  "
          f"(FOMC overlay +{FOMC_BOOST}x)\n")
    print(f"{'strategy':<14}{'CAGR':>7}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}{'end equity':>14}")
    for n, r, e in [("Book (1.5x)", book_ret, E), ("Buy&hold NQ", df['Close'].pct_change().loc[idx], nq_eq),
                    ("60/40", s6040, eq6040)]:
        p = perf(r)
        print(f"{n:<14}{p['CAGR']*100:>6.1f}%{p['vol']*100:>6.0f}%{p['Sharpe']:>8.2f}{p['maxDD']*100:>7.0f}%"
              f"   ${e.iloc[-1]/1e6:>9.1f}M")

    # cost actually paid (realistic, on compounding equity & integer contracts)
    notional_traded = (contracts.diff().abs().fillna(contracts.abs()) * px * MULT)
    cost_paid = notional_traded * COST
    yrs = (idx[-1] - idx[0]).days / 365.25
    print(f"\n  Sleeve-A realized trading cost: ${cost_paid.sum()/1e6:.2f}M over {yrs:.0f}y "
          f"= {cost_paid.sum()/E.mean()/yrs*100:.2f}%/yr of avg equity (was ~1.05%/yr at +1x FOMC)")
    print(f"  position now: {int(contracts.iloc[-1])} NQ contracts = ${exposureA.iloc[-1]/1e6:.1f}M "
          f"exposure on ${E.iloc[-1]/1e6:.1f}M equity ({nq_lev.iloc[-1]:.2f}x)")

    # ---- plots: equity ($), position allocation, drawdown ----
    fig, ax = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [3, 1.4, 1]}, sharex=True)
    ax[0].plot(E.index, E / 1e6, label="Book (1.5x)", lw=1.8, color="navy")
    ax[0].plot(nq_eq.index, nq_eq / 1e6, label="Buy&hold NQ", lw=1.0, color="firebrick")
    ax[0].plot(eq6040.index, eq6040 / 1e6, label="60/40", lw=1.0, color="darkorange")
    ax[0].set_yscale("log"); ax[0].set_ylabel("equity ($M, log)"); ax[0].legend(loc="upper left")
    ax[0].set_title("Real-environment simulation — equity curve ($10M start)")
    ax[1].plot(contracts.index, contracts.values, lw=0.7, color="teal")
    ax[1].set_ylabel("NQ contracts held"); ax[1].axhline(0, color="grey", lw=.5)
    dd = E / E.cummax() - 1
    ax[2].fill_between(dd.index, dd.values * 100, 0, color="firebrick", alpha=.5)
    ax[2].set_ylabel("Book DD %")
    fig.tight_layout()
    png = os.path.join(DATA_DIR, "real_sim.png"); fig.savefig(png, dpi=110); plt.close(fig)

    # save the daily allocation ledger
    led = pd.DataFrame({"equity_$": E.round(0), "NQ_contracts": contracts,
                        "NQ_exposure_$": exposureA.round(0), "NQ_leverage": nq_lev.round(3),
                        "book_ret": book_ret.round(5)})
    led.to_csv(os.path.join(DATA_DIR, "real_sim_ledger.csv"), index_label="Date")
    print(f"\n  equity + position allocation chart -> {png}")
    print(f"  daily allocation ledger -> {os.path.join(DATA_DIR, 'real_sim_ledger.csv')}")


if __name__ == "__main__":
    main()
