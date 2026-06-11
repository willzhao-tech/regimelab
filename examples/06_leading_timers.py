"""
Example 6 — do LEADING regime signals clear the bar? (with multiple-testing deflation)

05 showed the reactive "stress-exit" timer fails: it sells the bottom. Here we test
signals that could fire BEFORE the break, each as a de-risk overlay on NQ (invested,
else cash), through the same honest gauntlet:
  * causal signals, 1-day execution delay, transaction costs;
  * vs buy-&-hold, judged on TAIL metrics (maxDD, worst-1y) not just Sharpe;
  * block-shuffle null (does the TIMING beat random reorderings of the same in/out diet?);
  * drawdown-lead (does the exit lead or lag each >20% break?);
  * deflated Sharpe across ALL timers tried (multiple-testing honesty).

Timers:
  reactive    : exit in {Crash, Risk-off, Deflationary}     (from 05, reference)
  curve_inv   : exit when 10y-2y curve < 0 (inverted)        (leading, very early)
  sahm        : exit when Sahm recession flag is on          (leading/coincident)
  trend_200d  : exit when NQ < its 200-day moving average     (trend-following)
  trend_12m   : exit when trailing 12-month NQ return < 0     (trend-following)

Run:  python examples/06_leading_timers.py
"""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from regimelab.regime import RichRuleBasedIdentifier
from regimelab.evaluation import deflated_sharpe

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

t5 = _load("05_regime_timing")            # reuse perf / timer_returns / runs_of / load_panel
perf, timer_returns, runs_of = t5.perf, t5.timer_returns, t5.runs_of
COST = t5.COST


def block_shuffle_pctile(nq, invested, n=400, seed=7):
    """Fraction of block-shuffled in/out schedules the REAL timer beats (terminal, maxDD)."""
    real = perf(timer_returns(nq, invested))
    runs = runs_of(invested.values)
    rng = np.random.default_rng(seed)
    term, mdd = [], []
    for _ in range(n):
        order = rng.permutation(len(runs))
        seq = []
        for j in order:
            v, ln = runs[j]; seq.extend([v] * ln)
        p = perf(timer_returns(nq, pd.Series(seq, index=invested.index)))
        term.append(p["terminal"]); mdd.append(p["maxDD"])
    term, mdd = np.array(term), np.array(mdd)
    return ((term < real["terminal"]).mean() * 100,    # % of shuffles real beats on terminal
            (mdd < real["maxDD"]).mean() * 100)         # % with DEEPER dd than real (real shallower)


def episodes_of(nq, thr=0.20):
    eq = (1 + nq).cumprod()
    out = []; peak = eq.iloc[0]; pk = eq.index[0]; tr = eq.iloc[0]; trd = eq.index[0]
    for d, v in eq.items():
        if v >= peak:
            if (peak - tr) / peak >= thr:
                out.append((pk, trd))
            peak, pk, tr, trd = v, d, v, d
        elif v < tr:
            tr, trd = v, d
    if (peak - tr) / peak >= thr:
        out.append((pk, trd))
    return out


def lead_summary(invested, episodes):
    """Median lead (days, +ve = exit before peak) across episodes; % of episodes led."""
    exits = invested[(invested.shift(1) == 1) & (invested == 0)].index
    leads = []
    for pk, tr in episodes:
        win = [e for e in exits if pk - pd.Timedelta(days=120) <= e <= tr]
        if win:
            leads.append((pk - min(win)).days)
    if not leads:
        return float("nan"), 0
    return float(np.median(leads)), int(np.mean([l > 0 for l in leads]) * 100)


def main():
    panel = t5.load_panel()
    nq = panel.returns["NQ"].dropna()
    px = panel.prices["NQ"].dropna()
    m = panel.macro
    labels = RichRuleBasedIdentifier().label(panel)

    def causal(raw):                       # bool 'invested' -> causal float, default in-market
        return raw.reindex(nq.index).shift(1).fillna(1.0).astype(float)

    timers = {
        "reactive":   causal(~labels.isin({"Crash", "Risk-off", "Deflationary"})),
        "curve_inv":  causal(m["curve"].reindex(nq.index) >= 0),
        "sahm":       causal(m["recession"].reindex(nq.index) < 0.5),
        "trend_200d": causal(px > px.rolling(200).mean()),
        "trend_12m":  causal(px.pct_change(252) > 0),
    }
    eps = episodes_of(nq)
    bh = perf(nq)

    print(f"\nNQ {nq.index.min().date()} -> {nq.index.max().date()} ({len(nq)} days), "
          f"{len(eps)} drawdowns >20%\n")
    print("=" * 116)
    print("LEADING-TIMER GAUNTLET (unlevered NQ overlay; all causal, 5bps/switch)")
    print("=" * 116)
    print(f"{'timer':<12}{'CAGR':>7}{'Sharpe':>8}{'maxDD':>8}{'worst1y':>9}{'term':>8}"
          f"{'%inv':>6}{'sw':>5}{'beats_shuf_term':>16}{'shallower_dd':>14}{'medLead':>9}{'%led':>6}")
    print(f"{'buy&hold':<12}{bh['CAGR']*100:>6.1f}%{bh['Sharpe']:>8.2f}{bh['maxDD']*100:>7.0f}%"
          f"{bh['worst1y']*100:>8.0f}%{bh['terminal']:>7.2f}x{'100':>6}{'0':>5}{'-':>16}{'-':>14}{'-':>9}{'-':>6}")
    results = {}
    for name, inv in timers.items():
        p = perf(timer_returns(nq, inv))
        bt, bd = block_shuffle_pctile(nq, inv)
        ml, pl = lead_summary(inv, eps)
        results[name] = (p, bt, bd)
        sw = int(inv.diff().abs().sum())
        print(f"{name:<12}{p['CAGR']*100:>6.1f}%{p['Sharpe']:>8.2f}{p['maxDD']*100:>7.0f}%"
              f"{p['worst1y']*100:>8.0f}%{p['terminal']:>7.2f}x{inv.mean()*100:>5.0f}%{sw:>5}"
              f"{bt:>15.0f}%{bd:>13.0f}%{ml:>8.0f}d{pl:>5.0f}%")

    # ---- multiple-testing deflation on the best Sharpe -------------------------
    n_trials = len(timers)
    best = max(results, key=lambda k: results[k][0]["Sharpe"])
    bp = results[best][0]
    dsr = deflated_sharpe(bp["Sharpe"], n_periods=len(nq), n_trials=n_trials)
    print("\n" + "=" * 116)
    print(f"MULTIPLE-TESTING DEFLATION:  {n_trials} timers tried; best by Sharpe = '{best}' "
          f"(Sharpe {bp['Sharpe']:.2f})")
    print(f"  Deflated Sharpe (vs buy&hold-relevant hurdle of {n_trials} trials) = {dsr:.2f}  "
          f"(-> 1 survives; buy&hold Sharpe = {bh['Sharpe']:.2f})")
    print("=" * 116)
    print("\nReading: a timer 'works' only if it beats buy&hold on a TAIL metric net of cost,")
    print("sits in the top few % of its block-shuffle null, AND shows positive drawdown-lead.")
    print("'beats_shuf_term' / 'shallower_dd' = % of random reorderings the real timing beats;")
    print(">~90% = informative timing. medLead>0 = exits before the peak.")


if __name__ == "__main__":
    main()
