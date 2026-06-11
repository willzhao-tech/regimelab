# -*- coding: utf-8 -*-
"""
APPROACH: cost_coverage_gate

CAUSAL COST-COVERAGE INCLUSION
==============================
Idea: a short-vol sleeve only deserves to be in the book on days when it is
EMPIRICALLY paying for its own frictions. The harness exposes, for each market
and each day, these causal series in ctx:
    prem[t]   = gross option premium captured per trade (the credit collected)
    spread[t] = per-trade spread cost RATE (already L4-stress-widened, EEM-bumped)
    ret[t]    = underlying return (the realized move the short-vol leg pays out)

CALIBRATION FINDING (see header of run output): the *spread* itself is tiny
(2.4-4.6 bp) relative to the premium (~80-100 bp), so "gross premium vs spread
cost" is ALWAYS covered ~25-40x and never gates anything. The friction that
actually kills the book is the REALIZED MOVE the short-vol leg must pay against
the premium. So the economically-correct cost-coverage quantity is the sleeve's
realized NET per-unit short-vol payoff:

    net_unit[t] = prem[t-1] - |ret[t] - eps[t]| - spread[t-1]*prem[t-1]

This is exactly the per-traded-unit PnL the harness books (modulo the long-leg
2x and 1.5bp floor, which only make costs larger -> our gate is conservative).
It is the empirical answer to "is this market paying for ALL its frictions?".

We INCLUDE the sleeve on day t (gate=1) only when its TRAILING (252d) mean
net_unit, shifted 1 day so only the past is used, clears a margin M (in bp):

    trailing_edge[t] = mean(net_unit over past 252d, as of t-1)
    gate[t] = 1  iff  trailing_edge[t] >= M_bp
            = 0  otherwise  (sleeve muted -- it is currently NOT paying its way)

So the book simply trades the markets that are currently paying for ALL their
realized frictions, decided purely on trailing data.

WHY THIS IS NOT POST-HOC (the cardinal rule)
---------------------------------------------
- The gate at day t uses ONLY data up to t-1 (252d trailing window, .shift(1)).
- No market is hand-picked, dropped, or weighted using full-sample / LOO results.
- The SAME rule and the SAME single threshold M are applied to ALL 8 markets.
- If the rule ends up mostly keeping US sleeves and muting HSI, that is an
  EMERGENT, causal consequence of trailing economics -- not us typing "drop HSI".
  Compare: "drop HSI because LOO says so" is FORBIDDEN; "mute any sleeve whose
  trailing capture stops covering its trailing spread" is ALLOWED.
- The book weighting (book_of) is untouched / equal-risk; the only intervention
  is the causal on/off inclusion gate.
"""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
import bookopt_harness as H

SQ = H.SQ
NAMES = [p[0] for p in H.PAIRS]
WIN = 252          # trailing window (1 trading year)
MARGIN_BP = 0.0    # require trailing mean net-of-friction per-unit edge >= this many bp


def _eps(idx):
    # same discrete-strike +-0.125% alternating offset market() uses
    return pd.Series(0.00125 * np.where(np.arange(len(idx)) % 2, 1, -1), index=idx)


def _ctx_premspread(n):
    """Rebuild the causal prem, spread, ret series market() uses, for market n."""
    H._load()
    from math import erf
    Nrm = lambda x: 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))
    df, ret, vi, sp0 = H._DATA[n]
    idx = ret.index
    prem = pd.Series(2 * (2 * Nrm(0.5 * (H.K * vi.values / 100) * np.sqrt(H.DT)) - 1), index=idx)
    spread = (pd.Series(sp0, index=idx) *
              (1 + (vi / vi.rolling(63).median().shift(1) - 1).clip(lower=0)).fillna(1.)).fillna(sp0)
    if n == "EEM":
        spread = spread + 0.005
    return prem, spread, ret, idx


def coverage_indicator(n, margin_bp=MARGIN_BP, win=WIN):
    """CAUSAL trailing cost-coverage indicator (1=paying its way, 0=not) for market n.

    net_unit[t] = prem[t-1] - |ret[t]-eps[t]| - spread[t-1]*prem[t-1]   (realized
    per-traded-unit short-vol payoff: premium minus realized move minus spread).
    coverage[t] = 1 iff mean(net_unit over the past `win` days, as known at t-1)
                       >= margin_bp ; else 0.

    Strictly causal: the trailing mean is rolling-then-shift(1), so the indicator
    for day t depends only on net_unit up to t-1 (day t's own move never enters).
    The returned series is intended to be used as a book-level weight and is
    shifted ONE MORE day by the caller before going into book_of (belt & braces).
    """
    m = margin_bp * 1e-4
    prem, spread, ret, idx = _ctx_premspread(n)
    eps = _eps(idx)
    net_unit = prem.shift(1) - (ret - eps).abs() - spread.shift(1) * prem.shift(1)
    trailing_edge = net_unit.rolling(win).mean().shift(1)
    cov = (trailing_edge >= m).astype(float)
    cov = cov.where(trailing_edge.notna(), 1.0)   # neutral prior pre-window: include
    return cov


def build_book(margin_bp):
    """Equal-risk book, but each sleeve carries a CAUSAL cost-coverage weight in {0,1}.

    Sleeves are computed UNGATED (clean per-sleeve vol scaling). The cost-coverage
    decision enters only as book_of(weights=...): on days a sleeve is not paying
    its trailing frictions it gets weight 0 and book_of drops it from that day's
    cross-sectional average (W.where(P.notna()).sum). This is exactly the
    harness-sanctioned 'down-weight a sleeve by its trailing cost-coverage' path."""
    sleeves, weights, active = {}, {}, {}
    for n in NAMES:
        gp, _ = H.market(n)                       # UNGATED sleeve
        if gp is None:
            continue
        sleeves[n] = gp
        # caller-side extra .shift(1): book_of weights must be causal Series
        weights[n] = coverage_indicator(n, margin_bp=margin_bp).shift(1)
        active[n] = float(weights[n].reindex(gp.index).fillna(0).mean())
    bk = H.book_of(sleeves, weights=weights)
    return bk, sleeves, active


def fraction_on(margin_bp):
    """Diagnostic: fraction of days each sleeve's coverage weight is ON (causal)."""
    return {n: float(coverage_indicator(n, margin_bp=margin_bp).mean()) for n in NAMES}


def main():
    # ---- baseline (no gate) ----
    base = {n: H.market(n)[0] for n in NAMES}
    base = {k: v for k, v in base.items() if v is not None}
    bbk = H.book_of(base)
    bs, bd = H.sharpe(bbk), H.maxdd(bbk)
    bh1, bh2 = H.split_halves(bbk)
    print("=== BASELINE (no gate, 8-market L4) ===")
    print(f"Sharpe {bs:.3f}  maxDD {bd:.3f}  halves {bh1:.3f}/{bh2:.3f}")

    # ---- cost-coverage gated book across margins (one fixed rule, swept threshold) ----
    print("\n=== COST-COVERAGE GATE (causal trailing 252d net-of-friction edge) ===")
    results = {}
    for margin in (0.0, 1.0, 2.0, 3.0):
        bk, sleeves, active = build_book(margin)
        s, d = H.sharpe(bk), H.maxdd(bk)
        h1, h2 = H.split_halves(bk)
        results[margin] = (s, d, h1, h2, active)
        on = fraction_on(margin)
        on_s = "  ".join(f"{n}:{on[n]*100:4.0f}%" for n in NAMES if n in on)
        print(f"\nmargin={margin:>4.1f}bp: Sharpe {s:.3f}  maxDD {d:.3f}  halves {h1:.3f}/{h2:.3f}")
        print(f"   gate-ON fraction:  {on_s}")
    return bs, bd, bh1, bh2, results


if __name__ == "__main__":
    main()
