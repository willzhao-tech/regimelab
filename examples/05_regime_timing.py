"""
Example 5 — testing the regime-timing thesis HONESTLY.

Thesis: "ride the dominant regime, step aside before the break" beats buy-&-hold on
tail-adjusted terms. This is a regime-TIMING bet. The danger is fooling ourselves
(look-ahead, reacting after the drop, or mistaking a low-exposure effect for skill).
So the harness is honest by construction:

  * CAUSAL signals only: regime labels use macro known at the time (VIX / drawdown /
    curve are real-time; CPI / Sahm-recession already lagged to release). Decisions
    act with a 1-day delay.
  * TRANSACTION COSTS on every switch (turnover is the timer's hidden tax).
  * Two null tests that kill most timing claims:
      (1) BLOCK-SHUFFLE NULL — reorder the regime *runs* (preserving each regime's
          total time AND persistence) and re-apply to the SAME returns. If real
          timing isn't in the tail of the shuffled distribution, the *timing* of
          regimes carries no information beyond their frequency/persistence.
      (2) DRAWDOWN-LEAD TEST — for each >20% buy-&-hold drawdown, does the timer's
          exit LEAD the peak (skill) or LAG it (just locking in losses)?

Run:  python examples/05_regime_timing.py
"""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from regimelab.regime import RichRuleBasedIdentifier

HERE = os.path.dirname(os.path.abspath(__file__))
COST = 0.0005          # 5 bps one-way per 100% turnover
ASSET = "NQ"
EXIT_REGIMES = {"Crash", "Risk-off", "Deflationary"}   # acute, real-time-detectable stress


def load_panel():
    """Reuse build_real_panel() from 03 (numeric filename -> load by path)."""
    spec = importlib.util.spec_from_file_location("rp03", os.path.join(HERE, "03_real_panel.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.build_real_panel()


def perf(daily: pd.Series) -> dict:
    d = daily.dropna()
    eq = (1 + d).cumprod()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    mdd = float((eq / eq.cummax() - 1).min())
    worst12 = float(eq.pct_change(252).min())          # worst rolling 1y
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "vol": d.std() * np.sqrt(252),
            "Sharpe": d.mean() / d.std() * np.sqrt(252), "maxDD": mdd,
            "worst1y": worst12, "terminal": float(eq.iloc[-1])}


def timer_returns(nq: pd.Series, invested: pd.Series, cost: float = COST) -> pd.Series:
    """invested in {0,1} (already causal). Apply to NQ returns, charge cost on switches."""
    sw = invested.diff().abs().fillna(0.0)
    return invested * nq - cost * sw


def runs_of(arr):
    out, cur, ln = [], arr[0], 1
    for v in arr[1:]:
        if v == cur:
            ln += 1
        else:
            out.append((cur, ln)); cur, ln = v, 1
    out.append((cur, ln))
    return out


def main():
    panel = load_panel()
    labels = RichRuleBasedIdentifier().label(panel)
    nq = panel.returns[ASSET].dropna()          # NQ starts 1999; drop the pre-history NaNs
    idx = nq.index.intersection(labels.dropna().index)
    nq = nq.loc[idx]; labels = labels.loc[idx]
    print(f"\n{ASSET}: {idx.min().date()} -> {idx.max().date()} ({len(idx)} days)")
    print(f"EXIT regimes (step aside): {sorted(EXIT_REGIMES)}\n")

    invested = (~labels.isin(EXIT_REGIMES)).astype(float).shift(1).fillna(1.0)   # causal
    timer = timer_returns(nq, invested)
    bh = nq.copy()

    # ---- 1) headline comparison -------------------------------------------------
    print("=" * 92)
    print("REGIME TIMER vs BUY-&-HOLD  (unlevered NQ; timer sits in cash during EXIT regimes)")
    print("=" * 92)
    pt, pb = perf(timer), perf(bh)
    print(f"{'metric':<12}{'buy&hold':>14}{'regime-timer':>16}")
    for k in ["CAGR", "vol", "Sharpe", "maxDD", "worst1y", "terminal"]:
        fb = f"{pb[k]:.2f}x" if k == "terminal" else (f"{pb[k]*100:.1f}%" if k != "Sharpe" else f"{pb[k]:.2f}")
        ft = f"{pt[k]:.2f}x" if k == "terminal" else (f"{pt[k]*100:.1f}%" if k != "Sharpe" else f"{pt[k]:.2f}")
        print(f"{k:<12}{fb:>14}{ft:>16}")
    n_switch = int(invested.diff().abs().sum())
    print(f"\n  % time invested: {invested.mean()*100:.0f}%   switches: {n_switch}   "
          f"cost drag total: {COST*invested.diff().abs().sum()*100:.1f}%")

    # ---- 2) BLOCK-SHUFFLE NULL --------------------------------------------------
    print("\n" + "=" * 92)
    print("BLOCK-SHUFFLE NULL — reorder regime runs (same diet & persistence), re-apply to NQ")
    print("=" * 92)
    base_runs = runs_of(labels.values)
    rng = np.random.default_rng(7)
    N = 500
    sh_term, sh_mdd = [], []
    for _ in range(N):
        order = rng.permutation(len(base_runs))
        seq = []
        for j in order:
            lab, ln = base_runs[j]; seq.extend([lab] * ln)
        sl = pd.Series(seq, index=labels.index)
        inv_s = (~sl.isin(EXIT_REGIMES)).astype(float).shift(1).fillna(1.0)
        p = perf(timer_returns(nq, inv_s))
        sh_term.append(p["terminal"]); sh_mdd.append(p["maxDD"])
    sh_term, sh_mdd = np.array(sh_term), np.array(sh_mdd)
    pct_term = (sh_term < pt["terminal"]).mean() * 100
    pct_mdd = (sh_mdd < pt["maxDD"]).mean() * 100      # frac of shuffles with DEEPER DD than real
    print(f"  real timer terminal {pt['terminal']:.2f}x  vs shuffled median {np.median(sh_term):.2f}x"
          f"  -> real beats {pct_term:.0f}% of shuffles")
    print(f"  real timer maxDD {pt['maxDD']*100:.1f}%  vs shuffled median {np.median(sh_mdd)*100:.1f}%"
          f"  -> real shallower than {pct_mdd:.0f}% of shuffles")
    print("  (if real is NOT in the top few %, the TIMING of regimes adds nothing beyond their mix.)")

    # ---- 3) DRAWDOWN-LEAD TEST --------------------------------------------------
    print("\n" + "=" * 92)
    print("DRAWDOWN-LEAD TEST — does the exit LEAD the break (skill) or LAG it (too late)?")
    print("=" * 92)
    eq = (1 + bh).cumprod()
    # find >20% peak-to-trough episodes
    episodes = []
    peak = eq.iloc[0]; peak_d = eq.index[0]; trough = eq.iloc[0]; trough_d = eq.index[0]
    for d, v in eq.items():
        if v >= peak:
            if (peak - trough) / peak >= 0.20:
                episodes.append((peak_d, trough_d, trough / peak - 1))
            peak, peak_d, trough, trough_d = v, d, v, d
        elif v < trough:
            trough, trough_d = v, d
    if (peak - trough) / peak >= 0.20:
        episodes.append((peak_d, trough_d, trough / peak - 1))

    exits = invested[(invested.shift(1) == 1) & (invested == 0)].index   # 1->0 transitions
    print(f"  {'peak':>12}{'trough':>12}{'depth':>8}{'first exit':>14}{'lead vs peak':>14}")
    for pk, tr, depth in episodes:
        in_win = [e for e in exits if pk - pd.Timedelta(days=120) <= e <= tr]
        if in_win:
            e0 = min(in_win); lead = (pk - e0).days
            tag = f"{lead:+d}d {'(LEAD)' if lead > 0 else '(late)'}"
            ex = str(e0.date())
        else:
            tag, ex = "no exit", "-"
        print(f"  {str(pk.date()):>12}{str(tr.date()):>12}{depth*100:>7.0f}%{ex:>14}{tag:>14}")
    print("\n  Positive lead = exited BEFORE the peak (skill). Negative = exited after the drop")
    print("  had already started (reactive — locks in losses, may miss rebound).")


if __name__ == "__main__":
    main()
