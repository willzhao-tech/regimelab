"""
Example 4 — the 5-year horse-race: is risk-parity "efficient" in a client horizon?

Tests the thesis: within a reachable (<=5y) horizon, markets are concentrated/biased,
so risk-parity is not efficient. We run risk_parity vs all-equity vs 60/40 vs trend
over REAL long-history data (LONG-CORE NQ/US10Y/XAU, 1999-2026), all vol-targeted to
a common 10% so it is an apples-to-apples EFFICIENCY test (same risk budget). We also
show raw buy-&-hold NQ (no vol control) as the "ride the bias" reference.

Reports, per 5-year block: annualized Sharpe, max drawdown, and terminal wealth of $1.
Then, over many overlapping rolling 5y windows: how often each strategy WINS on each
metric — the direct test of "is risk-parity inefficient in the client horizon".

Run:  python examples/04_horse_race.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from regimelab.panel import Panel
from regimelab import strategies as strat

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"
FILES = {"NQ": "NQ_F_all_history.csv", "US10Y": "US10Y_all_history.csv",
         "XAU": "XAU_all_history.csv"}
CORE = ["NQ", "US10Y", "XAU"]                       # multi-asset core for RP / trend
SPECS = [                                           # (label, strategy, kwargs)
    ("risk_parity", "risk_parity", {}),
    ("all_equity(NQ)", "fixed", {"weights_map": {"NQ": 1.0}}),
    ("60/40", "fixed", {"weights_map": {"NQ": 0.6, "US10Y": 0.4}}),
    ("trend(126)", "trend", {"lookback": 126}),
]
TV, VW, RB = 0.10, 63, "M"                          # 10% vol target, 63d window, monthly rebal


def build_panel() -> Panel:
    closes = {n: pd.read_csv(os.path.join(DATA_DIR, f), parse_dates=["Date"])
              .set_index("Date")["Close"].sort_index() for n, f in FILES.items()}
    prices = pd.DataFrame(closes).sort_index()
    return Panel(returns=prices.pct_change(fill_method=None).iloc[1:], prices=prices)


def metrics(daily: pd.Series) -> tuple[float, float, float]:
    """Annualized Sharpe, max drawdown (<=0), terminal wealth of $1 over the window."""
    d = daily.dropna().values
    if len(d) < 30 or d.std() == 0:
        return float("nan"), float("nan"), float("nan")
    sharpe = d.mean() / d.std() * np.sqrt(252)
    eq = np.cumprod(1.0 + d)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return float(sharpe), mdd, float(eq[-1])


def run_strategy(name, kw, panel, start, end) -> pd.Series:
    return strat.run(strat.get(name, **kw), panel, instruments=CORE,
                     target_vol=TV, vol_win=VW, rebal=RB, start=start, end=end).returns


# ---- UNLEVERED (natural-vol) portfolios: gross exposure = 1, NO vol target -------
NATURAL = {
    "all_equity(NQ)": lambda m, i: np.array([1.0, 0.0, 0.0]),
    "60/40":          lambda m, i: np.array([0.6, 0.4, 0.0]),
    "equal_wt(1/3)":  lambda m, i: np.array([1/3, 1/3, 1/3]),
    "risk_parity":    lambda m, i: strat.inverse_vol_weights(m, i, VW),
}


def natural_portfolio(panel, wfun, start, end) -> pd.Series:
    """Monthly-rebalanced portfolio at NATURAL risk (weights sum to 1, no vol scaling)."""
    mat, cols, idx = panel.subset(CORE, start=start, end=end).to_matrix(CORE)
    n_inst, n_dates = mat.shape
    port = np.empty(n_dates - 1); w = None
    for i in range(1, n_dates):
        if w is None or idx[i].month != idx[i - 1].month:
            w = wfun(mat, i)
        port[i - 1] = float((w * mat[:, i]).sum())
    return pd.Series(port, index=idx[1:])


def metrics4(daily: pd.Series):
    """Sharpe, annualized vol, maxDD, terminal $1."""
    d = daily.dropna().values
    if len(d) < 30 or d.std() == 0:
        return (float("nan"),) * 4
    vol = d.std() * np.sqrt(252)
    sharpe = d.mean() / d.std() * np.sqrt(252)
    eq = np.cumprod(1.0 + d)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return float(sharpe), float(vol), mdd, float(eq[-1])


def main():
    panel = build_panel()
    idx = panel.common_dates(CORE)
    print(f"LONG-CORE {CORE}: {idx.min().date()} -> {idx.max().date()} ({len(idx)} days)\n")

    # ---- 1) consecutive non-overlapping ~5y blocks (human-readable) -------------
    blocks = [("2000-01-01", "2004-12-31"), ("2005-01-01", "2009-12-31"),
              ("2010-01-01", "2014-12-31"), ("2015-01-01", "2019-12-31"),
              ("2020-01-01", "2024-12-31"), ("2025-01-01", "2026-12-31")]
    print("=" * 100)
    print("FIVE-YEAR BLOCKS — Sharpe / maxDD / terminal $1 (all strategies vol-targeted to 10%)")
    print("  plus rawNQ = buy-&-hold NQ, NO vol control (the 'ride the bias' reference)")
    print("=" * 100)
    hdr = f"{'block':<22}" + "".join(f"{lab:>22}" for lab, _, _ in SPECS) + f"{'rawNQ($)':>12}"
    print(hdr)
    for s, e in blocks:
        tag = f"{s[:4]}-{e[:4]}" + ("*" if e.startswith("2026") else "")
        cells = []
        for lab, name, kw in SPECS:
            sh, dd, tw = metrics(run_strategy(name, kw, panel, s, e))
            cells.append(f"{sh:>5.2f}/{dd*100:>5.0f}%/{tw:>4.2f}x")
        rawnq = panel.subset(["NQ"], start=s, end=e).returns["NQ"]
        _, _, rawtw = metrics(rawnq)
        print(f"{tag:<22}" + "".join(f"{c:>22}" for c in cells) + f"{rawtw:>11.2f}x")

    # ---- 2) overlapping rolling 5y windows -> win-rates per metric --------------
    W = 5 * 252
    starts = range(0, len(idx) - W, 63)            # step ~quarterly
    rows = []
    for k in starts:
        s, e = str(idx[k].date()), str(idx[k + W].date())
        rec = {}
        for lab, name, kw in SPECS:
            rec[lab] = metrics(run_strategy(name, kw, panel, s, e))
        rows.append(rec)
    n = len(rows)
    labels = [lab for lab, _, _ in SPECS]
    print("\n" + "=" * 100)
    print(f"ROLLING 5-YEAR WINDOWS ({n} overlapping, quarterly step) — WIN-RATE per metric")
    print("=" * 100)
    print(f"{'strategy':<18}{'win% Sharpe':>14}{'win% lowDD':>14}{'win% terminal':>15}"
          f"{'med Sharpe':>13}{'med maxDD':>12}{'med term':>11}")
    # winners per window
    win_sh = {l: 0 for l in labels}; win_dd = {l: 0 for l in labels}; win_tw = {l: 0 for l in labels}
    for rec in rows:
        best_sh = max(labels, key=lambda l: (rec[l][0] if np.isfinite(rec[l][0]) else -1e9))
        best_dd = max(labels, key=lambda l: (rec[l][1] if np.isfinite(rec[l][1]) else -1e9))  # closest to 0
        best_tw = max(labels, key=lambda l: (rec[l][2] if np.isfinite(rec[l][2]) else -1e9))
        win_sh[best_sh] += 1; win_dd[best_dd] += 1; win_tw[best_tw] += 1
    for l in labels:
        shs = [r[l][0] for r in rows]; dds = [r[l][1] for r in rows]; tws = [r[l][2] for r in rows]
        print(f"{l:<18}{win_sh[l]/n*100:>13.0f}%{win_dd[l]/n*100:>13.0f}%{win_tw[l]/n*100:>14.0f}%"
              f"{np.nanmedian(shs):>13.2f}{np.nanmedian(dds)*100:>11.0f}%{np.nanmedian(tws):>10.2f}x")

    # ---- 3) UNLEVERED full-period: what a non-levering client actually gets -------
    print("\n" + "=" * 100)
    print("UNLEVERED (natural vol, gross=1, NO leverage) — full period 1999-2026")
    print("  this is the picture a client who will NOT lever actually faces")
    print("=" * 100)
    print(f"{'strategy':<18}{'ann vol':>10}{'Sharpe':>9}{'maxDD':>9}{'terminal $1':>14}{'CAGR':>9}")
    yrs = (idx.max() - idx.min()).days / 365.25
    for lab, wfun in NATURAL.items():
        sh, vol, mdd, tw = metrics4(natural_portfolio(panel, wfun, str(idx.min().date()), str(idx.max().date())))
        cagr = tw ** (1 / yrs) - 1
        print(f"{lab:<18}{vol*100:>9.1f}%{sh:>9.2f}{mdd*100:>8.0f}%{tw:>12.2f}x{cagr*100:>8.1f}%")


if __name__ == "__main__":
    main()
