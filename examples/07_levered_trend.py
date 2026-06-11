"""
Example 7 — the payoff strategy: VOL-TARGETED TREND-OVERLAY vs buy-&-hold.

06 showed trend-following is the one signal with provable tail value, but unlevered it
gives up return (it sits in cash ~30% of the time). Here we put it on an equal footing:
vol-target the trend-overlay to 15/20/25% annualized and ask the practical question —

  can we get EQUITY-LIKE return with roughly HALF the drawdown?

Three things compared, causal and net of 5bps/switch (monthly leverage rebalancing):
  buy&hold        : raw NQ (no control)              ~26.6% vol baseline
  voltarget_NQ    : NQ scaled to target vol           (isolates pure vol-targeting)
  voltarget_TREND : NQ-above-200dMA, scaled to target (vol-targeting + the trend brake)

The clean test is voltarget_TREND vs voltarget_NQ at the SAME target vol: does the trend
brake cut drawdown at equal risk? And vs buy&hold: does it match return with less tail?

Run:  python examples/07_levered_trend.py
"""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

t5 = _load("05_regime_timing")
perf, COST = t5.perf, t5.COST
VOL_WIN, MAX_LEV = 63, 3.0


def overlay(nq, signal, target, vol_win=VOL_WIN, max_lev=MAX_LEV, cost=COST):
    """Vol-target `signal`*NQ to `target` annualized vol; monthly leverage rebal; causal."""
    vol = nq.rolling(vol_win).std() * np.sqrt(252)
    desired = ((target / vol).clip(upper=max_lev) * signal).shift(1).fillna(0.0)  # causal
    pos, cur, pm = [], 0.0, None                       # hold the month-start position
    for d, v in desired.items():
        if pm is None or d.month != pm.month:
            cur = v
        pos.append(cur); pm = d
    pos = pd.Series(pos, index=desired.index)
    ret = pos * nq - cost * pos.diff().abs().fillna(0.0)
    return ret, pos


def main():
    panel = t5.load_panel()
    nq = panel.returns["NQ"].dropna()
    px = panel.prices["NQ"].dropna().reindex(nq.index)
    trend = (px > px.rolling(200).mean()).astype(float)        # 1 when above 200d MA
    ones = pd.Series(1.0, index=nq.index)

    bh = perf(nq)
    print(f"\nNQ {nq.index.min().date()} -> {nq.index.max().date()}\n")
    print("=" * 104)
    print("VOL-TARGETED TREND-OVERLAY vs buy-&-hold  (causal, 5bps/switch, monthly rebal, max 3x)")
    print("=" * 104)
    print(f"{'strategy':<22}{'target':>7}{'realVol':>9}{'CAGR':>8}{'Sharpe':>8}"
          f"{'maxDD':>8}{'worst1y':>9}{'terminal':>10}{'avgLev':>8}")
    print(f"{'buy&hold (raw NQ)':<22}{'-':>7}{bh['vol']*100:>8.0f}%{bh['CAGR']*100:>7.1f}%"
          f"{bh['Sharpe']:>8.2f}{bh['maxDD']*100:>7.0f}%{bh['worst1y']*100:>8.0f}%"
          f"{bh['terminal']:>9.2f}x{1.0:>8.2f}")
    for tgt in (0.15, 0.20, 0.25):
        for label, sig in [("voltarget_NQ", ones), ("voltarget_TREND", trend)]:
            ret, pos = overlay(nq, sig, tgt)
            p = perf(ret)
            print(f"{label:<22}{tgt*100:>6.0f}%{p['vol']*100:>8.0f}%{p['CAGR']*100:>7.1f}%"
                  f"{p['Sharpe']:>8.2f}{p['maxDD']*100:>7.0f}%{p['worst1y']*100:>8.0f}%"
                  f"{p['terminal']:>9.2f}x{pos[pos > 0].mean():>8.2f}")
        print("-" * 104)

    print("\nRead: compare voltarget_TREND vs voltarget_NQ at the SAME target (equal risk) — the")
    print("trend brake should cut maxDD/worst1y. vs raw buy&hold — match/beat CAGR with less tail.")
    print("avgLev = average leverage while invested (capped 3x). All net of cost.")


if __name__ == "__main__":
    main()
