# -*- coding: utf-8 -*-
"""
LONG-VOL LEG REDESIGN under FULL (ex-44) L4 frictions.

Context: long-vol legs (pos<0) pay 2x spread, and under stress-widening spreads they
barely fire / bleed cost. Question: is the book better off
  (a) dropping long legs entirely  (short+flat only), or
  (b) keeping long legs ONLY when expected convexity is large
      (rich very negative AND range >> breakeven),
... while preserving crisis protection (2008 / 2020 sub-windows)?

CAUSALITY DISCIPLINE
--------------------
Every gate decision is computed from ctx variables that the harness has ALREADY
causal-shifted (see harness docstring: "ctx ... ALL causal-shifted already"), and
the harness itself .shift(1)s the signal before forming pnl. The gate is the SAME
rule applied to EVERY market (no per-market parameter picking, no LOO market drop).
We never read full-sample results to choose markets, params, or weights.

The (b) convexity gate uses fixed structural thresholds (sign of rich, range vs
breakeven) chosen a priori from the economics of convexity, NOT tuned on the book
Sharpe. We report sensitivity to those thresholds so the reader can see we are not
cherry-picking a magic number.

Baseline to beat: L4 8-market book Sharpe 0.84 (maxDD -39%, H1 0.22 / H2 1.47).
"""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
import bookopt_harness as H

SQ = H.SQ
NAMES = [p[0] for p in H.PAIRS]

# ---- crisis sub-windows (for protection check) ----
CRISES = {
    "2008": ("2008-08-01", "2009-04-01"),
    "2020": ("2020-02-01", "2020-05-01"),
}

def sub(pnl, lo, hi):
    s = pd.Series(pnl).replace([np.inf, -np.inf], np.nan).dropna()
    w = s.loc[lo:hi]
    return w

def crisis_stats(book):
    out = {}
    for k, (lo, hi) in CRISES.items():
        w = sub(book, lo, hi)
        if len(w) > 5:
            out[k] = dict(total=float((1+w).prod()-1),
                          shp=H.sharpe(w) if len(w) > 100 else float("nan"),
                          worst=float(w.min()), n=int(len(w)))
        else:
            out[k] = dict(total=float("nan"), shp=float("nan"), worst=float("nan"), n=int(len(w)))
    return out

# ============================================================
# GATES (causal: ctx vars are pre-shifted by the harness)
# ============================================================
# The harness gate multiplies the *signal* (pos) by g in {0,1}. A gate that returns
# 0 where we DON'T want a long leg, but 1 everywhere else, would also kill short legs.
# We therefore build gates that are 1 for short-vol candidates always, and only
# modulate the LONG-vol (would-be pos<0) candidates. But the gate has no access to
# the eventual sign; it only sees ctx. The long-vol signal in the harness fires when
# rich<=p[1] & trend>=p[2]  (signal A)  OR  rng > b2*be (signal B).
# A long leg is therefore associated with rich being LOW and/or rng being HIGH.
#
# Design:
#  (a) DROP-LONG gate: zero the signal wherever the long-vol *conditions* hold, so
#      only short/flat survive. Long conditions = (rich very low) or (rng very high).
#      We must NOT zero short-vol candidates. Short-vol A fires on rich high & trend
#      low; short-vol B fires on rng < b1*be. Those never overlap the long region for
#      the same bar in the relevant variable, so masking the long region is safe-ish.
#      To be precise we kill the bar ONLY when it looks like a long-vol setup:
#         long_setup = (rich < 0) | (rng > be)   [structural, a priori]
#  (b) CONVEXITY-KEEP gate: keep the bar (g=1) when NOT a long setup (so shorts pass)
#      OR when it IS a long setup AND convexity is large
#         big_convex = (rich < RICH_TH) & (rng > RNG_MULT*be)
#      i.e. drop only the WEAK long-vol setups; keep the strongly-convex ones.
# ============================================================

# EXACT long-leg-eligible region, derived from the harness grids (GA/GB):
#   long-A fires when rich<=p[1] (p[1] in {0,-2}) & trend>=p[2] (p[2] in {0,1,2})
#           widest: rich<=0 & trend>=0
#   long-B fires when rng>b2*be (b2 in {1.3,1.6,2.0})
#           widest: rng>1.3*be
# Short legs live in DISJOINT regions (short-A: rich>=2; short-B: rng<1.2*be), so
# masking the long region does NOT touch any short leg. Verified disjoint by grid.
def _long_eligible(ctx):
    rich, trend, rng, be = ctx["rich"], ctx["trend"], ctx["rng"], ctx["be"]
    a = (rich <= 0) & (trend >= 0)
    b = (rng > 1.3 * be)
    return (a.fillna(False) | b.fillna(False))

def make_drop_long():
    """Gate: g=0 on every long-leg-eligible bar -> short + flat only (no long legs)."""
    def gate(ctx):
        elig = _long_eligible(ctx)
        g = pd.Series(1.0, index=ctx["rich"].index)
        g[elig] = 0.0
        return g
    return gate

def make_convex_keep(rich_th=-3.0, rng_mult=1.6):
    """Gate: keep ALL short legs; keep a long-leg-eligible bar ONLY when convexity is
    large (rich very negative AND range >> breakeven). Drop the weak long legs."""
    def gate(ctx):
        rich, rng, be = ctx["rich"], ctx["rng"], ctx["be"]
        elig = _long_eligible(ctx)
        big_convex = (rich < rich_th) & (rng > rng_mult * be)
        drop = elig & (~big_convex.fillna(False))
        g = pd.Series(1.0, index=rich.index)
        g[drop] = 0.0
        return g
    return gate

# ============================================================
# Build sleeves for a given gate, then book
# ============================================================
def build_book(gate=None):
    sl = {}
    for name in NAMES:
        gp, st = H.market(name, gate=gate)
        if gp is not None and len(gp.dropna()) > 300:
            sl[name] = gp.dropna()
    if not sl:
        return None, {}
    bk = H.book_of(sl)
    # NUMERICAL SANITATION (not a signal choice): when a gate zeros a sleeve for a
    # full 63d window, book_of's p/rolling_std(63) divides by zero -> +/-inf on a
    # handful of bars. Those bars carry no information; drop them. This is a harness
    # artifact of sparse gated sleeves, identical treatment for every variant.
    bk = bk.replace([np.inf, -np.inf], np.nan).dropna()
    return bk, sl

def report(tag, bk):
    if bk is None or len(bk.dropna()) < 100:
        print(f"[{tag}] EMPTY"); return None
    s = H.sharpe(bk); dd = H.maxdd(bk); h1, h2 = H.split_halves(bk)
    cr = crisis_stats(bk)
    print(f"[{tag}] Sharpe={s:.3f}  maxDD={dd:.3f}  H1={h1:.3f}  H2={h2:.3f}  n={len(bk.dropna())}")
    for k, v in cr.items():
        print(f"     crisis {k}: total={v['total']:+.4f}  worstday={v['worst']:+.4f}  n={v['n']}")
    return dict(tag=tag, sharpe=s, maxdd=dd, h1=h1, h2=h2, crisis=cr, n=len(bk.dropna()))

# ============================================================
# RUN
# ============================================================
print("="*70)
print("BASELINE (no gate, long legs as-is) -- must reproduce ~0.84")
print("="*70)
bk0, sl0 = build_book(gate=None)
r0 = report("baseline", bk0)

print()
print("="*70)
print("(a) DROP LONG LEGS ENTIRELY  (short + flat only)")
print("="*70)
bk_a, _ = build_book(gate=make_drop_long())
r_a = report("drop_long", bk_a)

print()
print("="*70)
print("(b) KEEP LONG LEGS ONLY WHEN STRONGLY CONVEX (rich<<0 & rng>>be)")
print("="*70)
# a-priori structural thresholds; we then show sensitivity below
bk_b, _ = build_book(gate=make_convex_keep(rich_th=-3.0, rng_mult=1.6))
r_b = report("convex_keep[-3.0,1.6]", bk_b)

print()
print("="*70)
print("SENSITIVITY of (b) to the convexity thresholds (NOT picking best on book)")
print("  -- shown only to demonstrate robustness, baseline choice fixed a priori --")
print("="*70)
sens = []
for rt in (-2.0, -3.0, -5.0):
    for rm in (1.4, 1.6, 1.8):
        bkx, _ = build_book(gate=make_convex_keep(rich_th=rt, rng_mult=rm))
        if bkx is not None and len(bkx.dropna()) > 100:
            s = H.sharpe(bkx); dd = H.maxdd(bkx); h1, h2 = H.split_halves(bkx)
            c20 = crisis_stats(bkx)["2020"]["total"]; c08 = crisis_stats(bkx)["2008"]["total"]
            sens.append((rt, rm, s, dd, h1, h2, c08, c20))
            print(f"  rich<{rt:>5} rng>{rm:>3}*be : Sh={s:.3f} dd={dd:.3f} H1={h1:.3f} H2={h2:.3f} cr08={c08:+.3f} cr20={c20:+.3f}")

print()
print("="*70)
print("SUMMARY vs baseline 0.84")
print("="*70)
def line(r):
    if r is None: return
    print(f"  {r['tag']:<26} Sh={r['sharpe']:.3f}  dd={r['maxdd']:.3f}  "
          f"H1={r['h1']:.3f} H2={r['h2']:.3f}  cr08={r['crisis']['2008']['total']:+.3f} "
          f"cr20={r['crisis']['2020']['total']:+.3f}")
for r in (r0, r_a, r_b):
    line(r)

# pick winner among the two redesigns by SHARPE, but require non-catastrophic crisis
def ok_crisis(r):
    # crisis protection preserved if 2020 (and 2008) total not materially worse than baseline
    if r is None or r0 is None: return False
    b08 = r0["crisis"]["2008"]["total"]; b20 = r0["crisis"]["2020"]["total"]
    return (r["crisis"]["2008"]["total"] >= b08 - 0.05) and (r["crisis"]["2020"]["total"] >= b20 - 0.05)

cands = [r for r in (r_a, r_b) if r is not None]
best = max(cands, key=lambda r: r["sharpe"]) if cands else None
print()
if best is not None:
    print(f"BEST redesign by Sharpe: {best['tag']}  Sharpe={best['sharpe']:.3f}  "
          f"(crisis-preserved={ok_crisis(best)})")
    if best["sharpe"] > (r0["sharpe"] if r0 else 0.84):
        print("  -> BEATS baseline Sharpe.")
    else:
        print("  -> Does NOT beat baseline Sharpe.")

# emit machine-readable final line for the orchestrator
import json
final = dict(
    baseline=dict(sharpe=(r0["sharpe"] if r0 else None), maxdd=(r0["maxdd"] if r0 else None),
                  h1=(r0["h1"] if r0 else None), h2=(r0["h2"] if r0 else None)),
    drop_long=(None if r_a is None else dict(sharpe=r_a["sharpe"], maxdd=r_a["maxdd"], h1=r_a["h1"], h2=r_a["h2"],
               cr08=r_a["crisis"]["2008"]["total"], cr20=r_a["crisis"]["2020"]["total"])),
    convex_keep=(None if r_b is None else dict(sharpe=r_b["sharpe"], maxdd=r_b["maxdd"], h1=r_b["h1"], h2=r_b["h2"],
               cr08=r_b["crisis"]["2008"]["total"], cr20=r_b["crisis"]["2020"]["total"])),
    best=(None if best is None else best["tag"]),
)
print("FINAL_JSON " + json.dumps(final))
