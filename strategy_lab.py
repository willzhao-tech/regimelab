# -*- coding: utf-8 -*-
"""STRATEGY LAB — the new-strategy intake path. Define ONE signal function; the platform supplies
everything else: per-market walk-forward (1260/252), FULL L4 frictions, selection-free floor
weighting (invvol x coverage), equity curve vs the standard benchmarks, and the metrics table.

    from strategy_lab import run_lab
    def my_signal(ctx, p):            # ctx: prem, spread, rich, trend, rng, be, vi, ret, df
        s = pd.Series(0., index=ctx["vi"].index)
        s[ctx["vi"] < p[0]] = 1.      # +1 = SHORT vol tomorrow, -1 = LONG vol tomorrow
        return s
    run_lab("my_idea", [(my_signal, [(15,), (20,)])])     # list of (fn, param_grid) families

Rules of the lab (the platform enforces what it can; you enforce the rest):
  * signals computed on day t are applied on t+1 (the harness shifts) — ctx series are
    same-day-aligned like the established families; do NOT .shift(-1) anything;
  * the grid you pass IS a multiple-testing surface: it is walk-forward-selected per window
    (never globally), and the lab reports how many combos you searched;
  * a pretty equity curve here is an AUDITION, not a result. The bar is `python gauntlet.py`
    (point it at your families) + the honesty battery (ex 44/46/49 analogues).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bookopt_harness as H, bookopt_floor as F, bookopt_stats as S
from bookopt_runlog import log_run

OUT = H.OUT
SQ = H.SQ


def _stats(r):
    r = pd.Series(r).dropna()
    if len(r) < 100 or r.std() == 0:
        return dict(sh=np.nan, t=np.nan, sk=np.nan, cagr=np.nan, dd=np.nan, term=np.nan)
    e = (1 + r).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    return dict(sh=r.mean()/r.std()*SQ, t=r.mean()/(r.std()/np.sqrt(len(r))), sk=r.skew(),
                cagr=e.iloc[-1]**(1/yrs) - 1, dd=float((e/e.cummax() - 1).min()),
                term=float(e.iloc[-1]))


def run_lab(name, families, make_chart=True):
    """Backtest a custom strategy through the full platform. Returns dict with the book series,
    per-market sleeves, and headline stats. Writes <name>_lab.png + a runs.jsonl entry."""
    n_combos = sum(len(g) for _, g in families)
    print(f"STRATEGY LAB - '{name}'  ({len(families)} family(ies), {n_combos} grid combos searched "
          f"per walk-forward window)\n")

    sleeves, statics = {}, {}
    for mkt, _, _, _ in H.PAIRS:
        res = H.market(mkt, families=families)
        if res[0] is not None:
            sleeves[mkt], statics[mkt] = res
    if not sleeves:
        raise RuntimeError("no market produced a sleeve (insufficient data or empty signals)")

    # selection-free floor weighting, SPARSE-SAFE: trailing-63d risk floored at 30% of the
    # expanding to-date risk, so a signal re-entering after a long flat stretch cannot be
    # scaled to absurd size by a near-zero short-window std (the failure mode dense
    # production sleeves never hit). All inputs shifted -> causal.
    def _safe_invvol(p):
        sd63 = p.rolling(63, min_periods=40).std().shift(1)
        sdfl = 0.30 * p.expanding(min_periods=252).std().shift(1)
        sd = pd.concat([sd63, sdfl], axis=1).max(axis=1)
        return (1.0 / sd.where(sd > 0))
    W = {k: _safe_invvol(sleeves[k]) * F.coverage_gate(k).reindex(sleeves[k].index) for k in sleeves}
    book = H.book_of(sleeves, W, prescale=False)
    allidx = pd.DatetimeIndex(sorted(set().union(*[set(sleeves[k].index) for k in sleeves])))
    cal = book.reindex(allidx).fillna(0.0)

    # standard benchmarks: the production floor book, same-friction static, SPX buy&hold
    floor_book, floor_sleeves, _ = F.build()
    static_book = H.book_of(statics)

    lo, hi = book.index.min(), book.index.max()          # benchmark on the same date RANGE
    spx = pd.read_csv(os.path.join(OUT, "SPX_all_history.csv"), parse_dates=["Date"]
                      ).set_index("Date")["Close"].pct_change().loc[lo:hi].dropna()
    rows = [(f"NEW: {name}", book), ("floor book (production)", floor_book.loc[lo:hi]),
            ("static short-vol (same frictions)", static_book.loc[lo:hi]),
            ("SPX buy&hold", spx)]
    print(f"{'strategy':<36}{'Sharpe':>8}{'t':>7}{'skew':>7}{'CAGR':>8}{'maxDD':>8}{'$1->':>7}")
    for tag, r in rows:
        s = _stats(r)
        print(f"{tag:<36}{s['sh']:>8.2f}{s['t']:>7.1f}{s['sk']:>7.1f}{s['cagr']*100:>7.1f}%"
              f"{s['dd']*100:>7.0f}%{s['term']:>6.1f}x")
    print(f"\ncalendar-basis Sharpe (idle=0): NEW {S.sharpe_ann(cal):.2f}  "
          f"(deployed {len(book)/len(allidx)*100:.0f}% of days)")
    per_mkt = {k: round(S.sharpe_ann(sleeves[k]), 2) for k in sleeves}
    print("per-market sleeve Sharpe:", per_mkt)

    if make_chart:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        for tag, r in rows:
            e = (1 + r.dropna()).cumprod()
            ax[0].plot(e.index, e.values, lw=2.2 if tag.startswith("NEW") else 1.1,
                       label=f"{tag} (Sh {_stats(r)['sh']:.2f})")
        ax[0].set_yscale("log"); ax[0].legend(loc="upper left"); ax[0].set_ylabel("growth of $1 (log)")
        ax[0].set_title(f"strategy lab - {name} vs standard benchmarks (L4 frictions, walk-forward)")
        e = (1 + book).cumprod(); dd = e/e.cummax() - 1
        ax[1].fill_between(dd.index, dd.values*100, 0, color="firebrick", alpha=.5)
        ax[1].set_ylabel("DD %")
        fig.tight_layout()
        png = os.path.join(OUT, f"{name}_lab.png")
        fig.savefig(png, dpi=110); plt.close(fig)
        print(f"\nequity curve -> {png}")

    log_run("strategy_lab", {"name": name, "sharpe_active": round(S.sharpe_ann(book), 4),
                             "sharpe_calendar": round(S.sharpe_ann(cal), 4),
                             "grid_combos": n_combos, "per_market": per_mkt})
    print("\nNEXT (if the audition is promising): adversarial battery + gauntlet —")
    print("  spread sweep (mult=1.5/2.0), k-sweep, placebo, leave-one-out, THEN python gauntlet.py")
    return dict(book=book, cal=cal, sleeves=sleeves, statics=statics)
