# -*- coding: utf-8 -*-
"""
bo_best.py  -- COMPOSE ONLY the honest, causal, non-post-hoc enhancements.

KEEP (REAL + causal, verified):
  (A) causal_riskweight   : per-sleeve weight = trailing realized Sharpe (252d),
                            floored >=0 +eps, .shift(1).  Continuous down-weight.
  (B) cost_coverage_gate  : per-sleeve {0,1} inclusion when trailing 252d mean
                            net-of-friction per-unit edge >= margin, .shift(1).

DISCARD (ARTIFACT or post-hoc, NOT composed):
  - friction_aware_gate (post-hoc Q-pick, never beats baseline, OOS-destroys)
  - longleg_redesign    (honest NEGATIVE result, 0.47 < baseline, long legs help)
  - vix1d_scaling       (robustness check that DEGRADES; dies under spread bump)
  - vol_target_tune     (vol-shrinkage artifact, fails random-delever placebo)

Both kept enhancements are sleeve-level causal weights fed to book_of(weights=...).
(A) is a *continuous* trailing-Sharpe weight; (B) is a *binary* trailing-edge gate.
We combine them MULTIPLICATIVELY: weight[t] = riskweight[t] * coverage_gate[t].
Both are .shift(1)-causal, so the product is causal. We then report the composed
book and stress it. We do NOT cherry-pick the margin: margin=0.0 (the
selection-free 'is it paying its way at all' threshold) is the headline; a small
sweep is shown only for transparency.

Cardinal-rule compliance:
  - All 8 markets ALWAYS enter; none dropped/picked via full-sample or LOO info.
  - Every per-sleeve quantity uses a trailing rolling window then .shift(1).
  - Same single rule + same params applied identically to all 8 markets.
"""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
from math import erf
import bookopt_harness as H

SQ = H.SQ
NAMES = [p[0] for p in H.PAIRS]
WIN = 252                      # one trailing year, used by both (A) and (B)

# ---------------------------------------------------------------------------
# Build the 8 honest UNGATED L4 sleeves once.
# ---------------------------------------------------------------------------
SLEEVES = {}
for n in NAMES:
    gp, _ = H.market(n)
    if gp is not None:
        SLEEVES[n] = gp

# ---------------------------------------------------------------------------
# (A) causal trailing-Sharpe weight  (continuous, >=eps, .shift(1))
# ---------------------------------------------------------------------------
def riskweight(pnl, win=WIN, eps=0.05):
    mu = pnl.rolling(win).mean()
    sd = pnl.rolling(win).std()
    sh = (mu / sd * SQ).replace([np.inf, -np.inf], np.nan)
    return (sh.clip(lower=0.0) + eps).shift(1)

# ---------------------------------------------------------------------------
# (B) causal cost-coverage gate  (binary, trailing net-of-friction edge, .shift(1))
# ---------------------------------------------------------------------------
Nrm = lambda x: 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))
def _eps_off(idx):
    return pd.Series(0.00125 * np.where(np.arange(len(idx)) % 2, 1, -1), index=idx)

def coverage_gate(n, margin_bp=0.0, win=WIN):
    H._load()
    df, ret, vi, sp0 = H._DATA[n]
    idx = ret.index
    prem = pd.Series(2 * (2 * Nrm(0.5 * (H.K * vi.values / 100) * np.sqrt(H.DT)) - 1), index=idx)
    spread = (pd.Series(sp0, index=idx) *
              (1 + (vi / vi.rolling(63).median().shift(1) - 1).clip(lower=0)).fillna(1.)).fillna(sp0)
    if n == "EEM":
        spread = spread + 0.005
    epsoff = _eps_off(idx)
    net_unit = prem.shift(1) - (ret - epsoff).abs() - spread.shift(1) * prem.shift(1)
    trailing_edge = net_unit.rolling(win).mean().shift(1)
    cov = (trailing_edge >= margin_bp * 1e-4).astype(float)
    cov = cov.where(trailing_edge.notna(), 1.0)      # neutral pre-window: include
    return cov.shift(1)                              # belt-and-braces extra shift

# ---------------------------------------------------------------------------
# Weight builders
# ---------------------------------------------------------------------------
def w_riskonly():
    return {n: riskweight(SLEEVES[n]) for n in SLEEVES}

def w_covonly(margin_bp=0.0):
    return {n: coverage_gate(n, margin_bp) for n in SLEEVES}

def w_composed(margin_bp=0.0):
    rw = w_riskonly(); cv = w_covonly(margin_bp)
    return {n: rw[n] * cv[n].reindex(rw[n].index) for n in SLEEVES}

# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def report(label, weights):
    bk = H.book_of(SLEEVES, weights=weights)
    s = H.sharpe(bk); dd = H.maxdd(bk); h1, h2 = H.split_halves(bk)
    cal = (s and dd) and (abs(s * 0 + (bk.add(1).prod()**(252/len(bk)) - 1) / abs(dd)))
    # annualized return for Calmar
    ann = (1 + bk).prod() ** (252.0 / len(bk)) - 1.0
    calmar = ann / abs(dd) if dd != 0 else float("nan")
    print(f"{label:30s} Sharpe={s:.3f}  maxDD={dd:.3f}  Calmar={calmar:.3f}  "
          f"halves=({h1:.3f},{h2:.3f})  n={len(bk)}")
    return dict(label=label, bk=bk, s=s, dd=dd, calmar=calmar, h1=h1, h2=h2, ann=ann)

def book_at_mult(weights_fn, mult, margin_bp=0.0):
    """Rebuild sleeves AND weights at a stressed spread mult, then book."""
    sl = {}
    for n in NAMES:
        gp, _ = H.market(n, mult=mult)
        if gp is not None:
            sl[n] = gp
    # weights must be rebuilt on the stressed sleeves / stressed spread for coverage
    if weights_fn == "risk":
        w = {n: riskweight(sl[n]) for n in sl}
    elif weights_fn == "cov":
        w = {n: _coverage_gate_mult(n, mult, margin_bp) for n in sl}
    elif weights_fn == "composed":
        cv = {n: _coverage_gate_mult(n, mult, margin_bp) for n in sl}
        w = {n: riskweight(sl[n]) * cv[n].reindex(sl[n].index) for n in sl}
    else:
        w = None
    bk = H.book_of(sl, weights=w)
    return H.sharpe(bk), H.split_halves(bk)

def _coverage_gate_mult(n, mult, margin_bp=0.0, win=WIN):
    """Coverage gate recomputed with the stressed spread*mult, matching market(mult)."""
    H._load()
    df, ret, vi, sp0 = H._DATA[n]
    idx = ret.index
    prem = pd.Series(2 * (2 * Nrm(0.5 * (H.K * vi.values / 100) * np.sqrt(H.DT)) - 1), index=idx)
    spread = (pd.Series(sp0, index=idx) *
              (1 + (vi / vi.rolling(63).median().shift(1) - 1).clip(lower=0)).fillna(1.)).fillna(sp0)
    if n == "EEM":
        spread = spread + 0.005
    spread = spread * mult
    epsoff = _eps_off(idx)
    net_unit = prem.shift(1) - (ret - epsoff).abs() - spread.shift(1) * prem.shift(1)
    trailing_edge = net_unit.rolling(win).mean().shift(1)
    cov = (trailing_edge >= margin_bp * 1e-4).astype(float)
    cov = cov.where(trailing_edge.notna(), 1.0)
    return cov.shift(1)

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
print("=== SLEEVES:", list(SLEEVES.keys()))
base = report("BASELINE equal-risk", None)
ronly = report("(A) riskweight only", w_riskonly())
conly = report("(B) coverage gate only m=0", w_covonly(0.0))
comp = report("(A)x(B) COMPOSED m=0", w_composed(0.0))
# transparency-only margin sweep on the composed book (m=0 is the headline)
print("\n-- composed margin sweep (headline = m=0; others shown for honesty) --")
for m in (0.0, 1.0, 2.0):
    report(f"composed m={m}", w_composed(m))

# ---- pick the composed m=0 as the DEFENSIBLE book ----
BEST = comp
print("\n================= SPREAD-SENSITIVITY (where does it break?) =================")
print(f"{'mult':>5s} | {'baseline':>9s} (h1,h2) | {'risk':>6s} | {'cov':>6s} | {'composed':>8s} (h1,h2)")
for mult in (1.0, 1.25, 1.5, 1.75, 2.0):
    bs, (bh1, bh2) = book_at_mult("base", mult)
    rs, _ = book_at_mult("risk", mult)
    cs, _ = book_at_mult("cov", mult, 0.0)
    ks, (kh1, kh2) = book_at_mult("composed", mult, 0.0)
    print(f"{mult:5.2f} | {bs:9.3f} ({bh1:.2f},{bh2:.2f}) | {rs:6.3f} | {cs:6.3f} | "
          f"{ks:8.3f} ({kh1:.2f},{kh2:.2f})")

# ---------------------------------------------------------------------------
# LEAVE-ONE-OUT concentration on the COMPOSED book (each market removed; the
# remaining sleeves keep their OWN trailing weights -- still fully causal).
# This is a DIAGNOSTIC of concentration, NOT a selection step.
# ---------------------------------------------------------------------------
print("\n================= LEAVE-ONE-OUT (composed m=0) =================")
print(f"full composed Sharpe = {BEST['s']:.3f}")
loo = {}
for drop in SLEEVES:
    sub = {n: SLEEVES[n] for n in SLEEVES if n != drop}
    rw = {n: riskweight(sub[n]) for n in sub}
    cv = {n: coverage_gate(n, 0.0) for n in sub}
    w = {n: rw[n] * cv[n].reindex(rw[n].index) for n in sub}
    bk = H.book_of(sub, weights=w)
    loo[drop] = H.sharpe(bk)
for k, v in sorted(loo.items(), key=lambda kv: kv[1]):
    print(f"  drop {k:6s} -> {v:.3f}   (delta {v-BEST['s']:+.3f})")

# US-only vs ex-US, on composed mechanics, for the 'US-centric?' question
def subset_book(keep, margin_bp=0.0):
    sub = {n: SLEEVES[n] for n in keep if n in SLEEVES}
    rw = {n: riskweight(sub[n]) for n in sub}
    cv = {n: coverage_gate(n, margin_bp) for n in sub}
    w = {n: rw[n] * cv[n].reindex(rw[n].index) for n in sub}
    return H.sharpe(H.book_of(sub, weights=w))
print("\n-- subset books under composed mechanics --")
print(f"  US-only (SPX,NQ,EEM)         -> {subset_book(['SPX','NQ','EEM']):.3f}")
print(f"  ex-US (drop SPX,NQ,EEM)      -> {subset_book(['DAX','SX5E','N225','HSI','NIFTY']):.3f}")
print(f"  no-SPX (other 7)             -> {subset_book(['NQ','EEM','DAX','SX5E','N225','HSI','NIFTY']):.3f}")

# redundancy check: are (A) and (B) independent, or is the composition just one
# of them? corr of the two weight matrices, and incremental Sharpe.
print("\n-- redundancy of (A) vs (B) --")
print(f"  baseline {base['s']:.3f} | +A {ronly['s']:.3f} | +B {conly['s']:.3f} | +A+B {comp['s']:.3f}")
print(f"  incremental of B given A : {comp['s']-ronly['s']:+.3f}")
print(f"  incremental of A given B : {comp['s']-conly['s']:+.3f}")
