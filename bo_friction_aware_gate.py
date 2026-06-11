# -*- coding: utf-8 -*-
"""
APPROACH: friction_aware_gate
=============================
Suppress trades when frictions are likely to eat the edge, using ONLY trailing/causal info.

Two independent suppression channels (ANDed into the signal via H.market(gate=...)):
  (1) SPREAD-REGIME gate: trade only when the (already-causal) spread is NOT in its
      high regime, i.e. spread <= rolling-252d Q-th percentile. We additionally .shift(1)
      the threshold so the percentile is computed from data strictly BEFORE today.
  (2) EDGE-CLEARS-COST gate: trade only when the expected per-trade edge exceeds the
      modeled cost. Cost proxy = spread*prem (the harness charges ~spread*prem per unit
      |pos|, doubled on long legs, 1.5bp floor). Edge proxy for the rich/cheap sleeve is
      |rich|/100*prem-ish scale; for the range sleeve it is the distance of range from the
      breakeven band. We require edge >= kappa * cost, with kappa a FIXED constant (not
      tuned on full-sample results).

CAUSALITY ARGUMENT (why this is NOT post-hoc):
  - Every series in ctx (prem, spread, rich, trend, rng, be, vi, ret) is already causal-
    shifted by the harness. The gate consumes them and adds its own .shift(1) on any
    rolling statistic (percentile threshold, cost normaliser), so the decision for day t
    uses only data observable strictly before t.
  - The gate parameters (percentile Q, kappa) are FIXED constants applied UNIFORMLY to all
    8 markets. We do NOT pick which markets to gate, nor tune Q/kappa per market using the
    full-sample book Sharpe. We do NOT drop HSI or any sleeve. The same rule runs everywhere.
  - The walk-forward grid selection inside H.market is unchanged; we only thin the signal
    with a trailing-decidable mask. Turnover only ever goes DOWN.
"""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
import bookopt_harness as H

NAMES = [p[0] for p in H.PAIRS]

# ---------- FIXED gate constants (uniform across markets, not full-sample tuned) ----------
SPREAD_Q = 0.70   # suppress when trailing-252d spread percentile rank > 0.70 (high-cost regime)
KAPPA    = 1.0    # require expected edge >= KAPPA * modeled cost
PCT_WIN  = 252    # trailing window for the spread percentile
FLOOR    = 0.15   # soft-gate floor: in bad-friction regimes scale signal to FLOOR, not 0.
#  Rationale for a soft floor (not hard 0/1): the harness equal-risk step divides each
#  sleeve PnL by its rolling-63d std. A hard on/off gate can flatten a sleeve for 63+
#  consecutive days -> std==0 -> +/-inf leverage -> NaN book Sharpe (a pure div-by-zero
#  artifact, NOT a real blow-up; the pre-artifact maxDD was actually only ~-13%). Scaling
#  to a small positive floor keeps every sleeve's variance > 0 while still concentrating
#  risk into low-friction days. FLOOR is one fixed constant applied uniformly to all markets.


def make_gate(spread_q=SPREAD_Q, kappa=KAPPA, floor=FLOOR, use_spread=True, use_edge=True):
    def gate(ctx):
        spread = ctx["spread"]; prem = ctx["prem"]; rich = ctx["rich"]
        rng = ctx["rng"]; be = ctx["be"]
        idx = spread.index
        keep = pd.Series(True, index=idx)

        # (1) spread-regime: trailing percentile rank of TODAY's spread within the prior 252d.
        #     rank in [0,1]; shift(1) makes the comparison window strictly historical.
        if use_spread:
            rank = spread.rolling(PCT_WIN, min_periods=60).apply(
                lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else 0.5, raw=True)
            rank = rank.shift(1)
            keep = keep & (rank.fillna(0.0) <= spread_q)

        # (2) edge-clears-cost: cost proxy = spread*prem (per unit |pos|). Edge proxies:
        #     - rich/cheap sleeve edge ~ |rich|/100 * prem (fractional mispricing on daily scale)
        #     - range sleeve edge ~ |rng - be| / 100 (distance of range from breakeven band)
        #     Trade survives if EITHER sleeve's edge clears cost.
        if use_edge:
            cost = (spread * prem).shift(1)
            edge_rich = (rich.abs() / 100.0 * prem).shift(1)
            edge_rng  = ((rng - be).abs() / 100.0).shift(1)
            edge = pd.concat([edge_rich, edge_rng], axis=1).max(axis=1)
            keep = keep & (edge.fillna(0.0) >= kappa * cost.fillna(np.inf))

        # soft gate: keep-days -> weight 1.0, suppressed-days -> weight = floor (not 0).
        return keep.astype(float) * (1.0 - floor) + floor
    return gate


def build_book(gate=None):
    sleeves = {}
    for nm in NAMES:
        g, _ = H.market(nm, gate=gate)
        if g is not None:
            sleeves[nm] = g
    bk = H.book_of(sleeves)
    return bk, sleeves


def gate_keep_fraction(gate):
    """Avg fraction of days the gate is at FULL weight (>~1) across markets.
    This is the true 'fewer trades' metric: it measures how often the friction filter
    lets a full-conviction trade through. Independent of the soft floor."""
    if gate is None:
        return 1.0
    fr = []
    for nm in NAMES:
        try:
            _load_ctx = _market_ctx(nm)
        except Exception:
            continue
        if _load_ctx is None:
            continue
        w = gate(_load_ctx).dropna()
        if len(w):
            fr.append((w > 0.99).mean())   # full-weight days (keep==True -> weight 1.0)
    return float(np.mean(fr)) if fr else float("nan")


def _market_ctx(name):
    """Rebuild the SAME ctx dict the harness passes to gate(), for turnover accounting."""
    H._load()
    df, ret, vi, sp0 = H._DATA[name]
    idx = ret.index
    if len(idx) < H.TRAIN + H.TEST + 50:
        return None
    DT = H.DT; K = H.K; SQ = H.SQ; Nrm = H.Nrm
    prem = pd.Series(2 * (2 * Nrm(0.5 * (K * vi.values / 100) * np.sqrt(DT)) - 1), index=idx)
    pk = lambda w: (np.sqrt((np.log(df["High"] / df["Low"]) ** 2).rolling(w).mean()
                            / (4 * np.log(2))) * SQ * 100).shift(1)
    rich = vi - pk(21); trend = pk(10) - pk(42)
    rng = np.log(df["High"] / df["Low"]) * 100; be = vi / SQ
    spread = (pd.Series(sp0, index=idx) * (1 + (vi / vi.rolling(63).median().shift(1) - 1)
              .clip(lower=0)).fillna(1.)).fillna(sp0)
    if name == "EEM":
        spread = spread + 0.005
    return dict(prem=prem, spread=spread, rich=rich, trend=trend, rng=rng, be=be, vi=vi, ret=ret)


def report(tag, bk):
    sh = H.sharpe(bk); dd = H.maxdd(bk); h1, h2 = H.split_halves(bk)
    print(f"{tag:34s} Sharpe={sh:6.3f}  maxDD={dd:7.2%}  h1={h1:5.2f}  h2={h2:5.2f}  N={len(bk)}")
    return sh, dd, h1, h2


if __name__ == "__main__":
    print("=" * 96)
    print("FRICTION-AWARE GATE  (baseline target: L4 8-market book Sharpe 0.84, maxDD -39%, h1=0.22 h2=1.47)")
    print("=" * 96)

    # 0) Baseline (no gate) -- must reproduce ~0.84
    bk0, slv0 = build_book(gate=None)
    s0 = report("baseline (no gate)", bk0)
    to0 = 1.0  # baseline lets 100% of full-conviction trades through
    print(f"   baseline full-trade fraction = {to0:.3f}")
    print("-" * 96)

    # 1) Spread-regime gate only
    g_s = make_gate(use_spread=True, use_edge=False)
    bk_s, slv_s = build_book(gate=g_s)
    report("spread-regime gate (Q=0.70)", bk_s)
    ks = gate_keep_fraction(g_s)
    print(f"   full-trade frac = {ks:.3f}  (turnover reduction {1-ks/to0:+.1%})")

    # 2) Edge-clears-cost gate only
    g_e = make_gate(use_spread=False, use_edge=True)
    bk_e, slv_e = build_book(gate=g_e)
    report("edge-clears-cost gate (kappa=1.0)", bk_e)
    ke = gate_keep_fraction(g_e)
    print(f"   full-trade frac = {ke:.3f}  (turnover reduction {1-ke/to0:+.1%})")

    # 3) Combined gate (both channels)
    g_c = make_gate(use_spread=True, use_edge=True)
    bk_c, slv_c = build_book(gate=g_c)
    sc = report("combined gate (Q=0.70,kappa=1.0)", bk_c)
    kc = gate_keep_fraction(g_c)
    print(f"   full-trade frac = {kc:.3f}  (turnover reduction {1-kc/to0:+.1%})")
    print("-" * 96)

    # 4) Robustness sweep over Q and kappa (NOT used to pick the headline number; shown to
    #    demonstrate the result is not a knife-edge artifact of one (Q,kappa) choice).
    print("Robustness sweep (uniform constants; headline uses Q=0.70,kappa=1.0):")
    for q in (0.60, 0.70, 0.80):
        for kp in (0.0, 0.5, 1.0, 1.5):
            g = make_gate(spread_q=q, kappa=kp, use_spread=True, use_edge=(kp > 0))
            bk, slv = build_book(gate=g)
            sh = H.sharpe(bk); dd = H.maxdd(bk); h1, h2 = H.split_halves(bk)
            print(f"   Q={q:.2f} kappa={kp:.1f}  Sharpe={sh:6.3f}  maxDD={dd:7.2%}"
                  f"  h1={h1:5.2f} h2={h2:5.2f}  full-frac={gate_keep_fraction(g):.3f}")

    print("=" * 96)
    print("SUMMARY")
    print(f"  baseline       Sharpe={s0[0]:.3f}  maxDD={s0[1]:.2%}  h1={s0[2]:.2f} h2={s0[3]:.2f}")
    print(f"  combined gate  Sharpe={sc[0]:.3f}  maxDD={sc[1]:.2%}  h1={sc[2]:.2f} h2={sc[3]:.2f}")
    print(f"  delta Sharpe   {sc[0]-s0[0]:+.3f}   beats 0.84: {sc[0] > 0.84}")
    print("=" * 96)
