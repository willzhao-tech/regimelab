"""
Example 20 — full backtest of the execution plan: equity curve, dynamic benchmarks, trade log.

Book = 50/50 risk-equalized (each sleeve to 10% vol), levered to the plan's 1.5x (half-Kelly):
  Sleeve A: vol-target(Parkinson-21)+200d-trend brake on NQ, 15% tgt, 3x cap, monthly rebal,
            +1x leverage into each scheduled FOMC window (day -1 & day 0).
  Sleeve B: tail-hedged VXN short-variance (own +/-5% wings).  [variance-swap proxy, satellite]

Outputs:
  - equity_curve.png   : log-equity vs benchmarks + drawdown + rolling-1y Sharpe (dynamic benchmark)
  - book_trade_log.csv : every Sleeve-A rebalance on a $10M notional (date, trigger, contracts, cost)
  - printed: annual return table, rolling stats, trade-log summary + excerpt.

Run:  python examples/20_backtest_book.py
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
FOMC_BOOST, CAP_TILT = 1.0, 4.0
SLEEVE_VOL, BOOK_LEVERAGE = 0.10, 1.5
AUM, MULT = 10_000_000, 20            # $10M notional, NQ = $20/pt (for the trade log)


def perf(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252),
            "maxDD": float((eq / eq.cummax() - 1).min()),
            "worst1y": float(eq.pct_change(252).min()), "term": float(eq.iloc[-1])}


def sleeve_A(df, fomc):
    """Return (daily returns, daily leverage held, trade log DataFrame)."""
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
    pos = base.where(~win, np.minimum(base + FOMC_BOOST, CAP_TILT))
    rA = (pos * ret - COST * pos.diff().abs().fillna(0.0)).dropna()

    # trade log on a $10M notional
    contracts = (pos * AUM / (px * MULT)).round()
    chg = contracts.diff().fillna(contracts).abs()
    log = []
    prev_in_win = False
    for d in pos.index[1:]:
        dc = contracts[d] - contracts.shift(1)[d]
        if abs(dc) < 1:
            prev_in_win = bool(win[d]); continue
        if win[d] and not prev_in_win:
            trig = "FOMC_TILT_ON"
        elif prev_in_win and not win[d]:
            trig = "FOMC_TILT_OFF"
        elif base[d] == 0 and base.shift(1)[d] > 0:
            trig = "TREND_EXIT"
        elif base[d] > 0 and base.shift(1)[d] == 0:
            trig = "TREND_REENTER"
        else:
            trig = "MONTHLY_REBAL"
        log.append({"date": d.date(), "trigger": trig, "NQ_px": round(px[d], 1),
                    "leverage": round(pos[d], 2), "contracts": int(contracts[d]),
                    "trade": int(dc), "notional_$": int(abs(dc) * px[d] * MULT),
                    "cost_$": round(abs(dc) * px[d] * MULT * COST, 0)})
        prev_in_win = bool(win[d])
    return rA, pos, pd.DataFrame(log)


def sleeve_B(df, vxn):
    ret = df["Close"].pct_change()
    idx = ret.index.intersection(vxn.index); ret, vxn = ret.loc[idx], vxn.loc[idx]
    iv = (vxn.shift(1) / 100) ** 2 / 252; rvar = ret ** 2; capv = 0.05 ** 2
    wing = np.maximum(rvar - capv, 0).mean()
    return (iv - np.minimum(rvar, capv) - wing).dropna()


def scale_to(r, t=SLEEVE_VOL):
    return r * (t / (r.std() * np.sqrt(252)))


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

    rA_raw, posA, trades = sleeve_A(df, fomc)
    rA, rB = scale_to(rA_raw), scale_to(sleeve_B(df, vxn))
    idx = rA.index.intersection(rB.index)
    rA, rB = rA.loc[idx], rB.loc[idx]
    book = BOOK_LEVERAGE * (0.5 * rA + 0.5 * rB)
    nq = df["Close"].pct_change().loc[idx]
    b6040 = (0.6 * nq + 0.4 * us10.pct_change().reindex(idx).fillna(0.0))
    series = {"Book (1.5x)": book, "Sleeve A": rA, "Sleeve B": rB, "Buy&hold NQ": nq, "60/40": b6040}

    # ---- benchmark table ----
    print(f"EXECUTION-PLAN BACKTEST  {idx.min().date()}..{idx.max().date()}\n")
    print(f"{'strategy':<16}{'CAGR':>7}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}{'worst1y':>9}{'terminal':>10}")
    for n, r in series.items():
        p = perf(r)
        print(f"{n:<16}{p['CAGR']*100:>6.1f}%{p['vol']*100:>6.0f}%{p['Sharpe']:>8.2f}"
              f"{p['maxDD']*100:>7.0f}%{p['worst1y']*100:>8.0f}%{p['term']:>9.1f}x")

    # ---- annual returns (dynamic benchmark) ----
    print("\nANNUAL RETURNS (%)")
    ann = pd.DataFrame({n: (1 + r).groupby(r.index.year).prod() - 1 for n, r in
                        {"Book": book, "NQ": nq, "60/40": b6040}.items()}) * 100
    print(ann.round(1).to_string())

    # ---- equity curve + dynamic panels ----
    eqs = {n: (1 + r).cumprod() for n, r in series.items()}
    dd = eqs["Book (1.5x)"] / eqs["Book (1.5x)"].cummax() - 1
    roll_sh = book.rolling(252).mean() / book.rolling(252).std() * np.sqrt(252)
    fig, ax = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)
    for n, e in eqs.items():
        ax[0].plot(e.index, e.values, label=n, lw=1.6 if "Book" in n else 1.0)
    ax[0].set_yscale("log"); ax[0].set_ylabel("growth of $1 (log)"); ax[0].legend(loc="upper left")
    ax[0].set_title("Execution-plan book vs benchmarks — equity curve")
    ax[1].fill_between(dd.index, dd.values * 100, 0, color="firebrick", alpha=.5)
    ax[1].set_ylabel("Book DD %")
    ax[2].plot(roll_sh.index, roll_sh.values, color="navy"); ax[2].axhline(0, color="grey", lw=.5)
    ax[2].set_ylabel("rolling 1y Sharpe")
    fig.tight_layout()
    png = os.path.join(DATA_DIR, "equity_curve.png"); fig.savefig(png, dpi=110); plt.close(fig)
    print(f"\nequity curve + dynamic panels -> {png}")

    # ---- trade log (aligned to the book/backtest period) ----
    trades = trades[pd.to_datetime(trades["date"]) >= idx.min()].reset_index(drop=True)
    tl = os.path.join(DATA_DIR, "book_trade_log.csv"); trades.to_csv(tl, index=False)
    print(f"\nTRADE LOG (Sleeve A, $10M notional) -> {tl}  ({len(trades)} trades)")
    print("  by trigger:", trades["trigger"].value_counts().to_dict())
    print(f"  total cost ${int(trades['cost_$'].sum()):,} over {((idx[-1]-idx[0]).days/365.25):.0f}y "
          f"= ${int(trades['cost_$'].sum()/((idx[-1]-idx[0]).days/365.25)):,}/yr on $10M")
    print("\n  first 8 trades:"); print(trades.head(8).to_string(index=False))
    print("\n  a FOMC-tilt round-trip (example):")
    fomc_tr = trades[trades["trigger"].str.startswith("FOMC")].head(4)
    print(fomc_tr.to_string(index=False))
    print("\n  last 8 trades:"); print(trades.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
