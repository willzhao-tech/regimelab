# -*- coding: utf-8 -*-
"""
wf_best.py -- COMBINED tail-safe short-vol config.

Stacks the two AUDITED-HONEST overlays that each add REAL OOS value on top of the
realistic hedged sleeve (smile=1.5), all causal, all WF-selected:

  (A) INVERSE-RISK DYNAMIC SIZING (from wf_dyn_size): continuous exposure in [0,1] that
      de-gears when yesterday's VXN level OR trailing vol-of-vol is elevated. This is the
      one overlay shown to beat its OWN static sleeve OOS (2.308 vs 1.037), i.e. the sizing
      logic adds value beyond the loss cap.
  (B) LEADING-STRESS DE-RISK (from wf_stress_derisk): a hard 0/1 gate that goes FLAT when a
      leading stress trigger fires (VXN-change trailing percentile, NQ<200dMA, NQ 20d dd).
      Audited uplift over always-on ~ +0.33 Sharpe (marginal, CI straddles 0, but causal).

Combined exposure_t = continuous_inverse_risk_size_t * stress_gate_t  (both pre-shifted).

Everything is causal:
  - VXN level / vol-of-vol use vxn.shift(1) (and trailing diff std .shift(1)).
  - stress signals are trailing (252d pctile, 200d MA, 20d dd) and .shift(1).
  - hedged_pnl: iv=(vxn.shift(1))^2 sold; loss=min(rvar,cap^2); wing=smile*trailing-mean(tail).shift(1).
  - smile FIXED >= 1.5 (realistic markup) -- NOT walk-forwarded (the load-bearing honesty knob).
  - causal_scale: rolling(63) std .shift(1), 8x leverage cap.
  - walk_forward picks the whole param set on a trailing-TRAIN Sharpe, applies OOS, rolls.
    The grid INCLUDES "static" (no sizing) and "always-on" (no gate) so WF can decline either
    overlay if it does not help in-train -- honest competition, no forced complexity.

Reports OOS Sharpe / maxDD / worst-day and compares to the realistic static baseline (1.394).
"""
import sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import shortvol_harness as H

SQ = H.SQ
SMILE = 1.5          # realistic markup, fixed (never tuned)

df, ret, vxn = H.load()
close = df["Close"]

# ----------------------------------------------------------------------------------
# (A) Causal inverse-risk continuous sizing in [0,1].
# de-gear if EITHER yesterday's VXN level is high OR trailing vol-of-vol is high.
# ----------------------------------------------------------------------------------
vxn_lag = vxn.shift(1)
dvxn = vxn.diff()

def inv_risk_size(ref_level, ref_vv, pwr=1.0, vv_win=21):
    e_lvl = (ref_level / vxn_lag).clip(lower=0.0, upper=1.0)
    volvol = dvxn.rolling(vv_win, min_periods=10).std().shift(1)
    e_vv = (ref_vv / volvol).clip(lower=0.0, upper=1.0)
    return (np.minimum(e_lvl, e_vv) ** pwr)

# ----------------------------------------------------------------------------------
# (B) Causal leading-stress 0/1 gate (go FLAT on a leading trigger).
# ----------------------------------------------------------------------------------
vchg = vxn.diff()
def trailing_pctile(s, win=252, minp=60):
    return s.rolling(win, min_periods=minp).apply(
        lambda w: (w[:-1] < w[-1]).mean() if np.isfinite(w[-1]) else np.nan, raw=True)
vchg_pct = trailing_pctile(vchg, 252, 60).reindex(ret.index)
ma200 = close.rolling(200, min_periods=100).mean()
below_ma = (close / ma200).reindex(ret.index)              # <1 => below 200d MA
roll_max20 = close.rolling(20, min_periods=10).max()
dd20 = (close / roll_max20 - 1.0).reindex(ret.index)       # <=0

def stress_gate(use_vxn, use_ma, use_dd, X, Y):
    trig = pd.Series(False, index=ret.index)
    if use_vxn: trig = trig | (vchg_pct >= X)
    if use_ma:  trig = trig | (below_ma < 1.0)
    if use_dd:  trig = trig | (dd20 <= -Y)
    trig = trig.fillna(False)
    return (~trig).astype(float)   # 1=invested, 0=flat  (caller shifts)

# ----------------------------------------------------------------------------------
# build_fn(params) -> causal full-history pnl.
# exposure = size(A) * gate(B), then shift(1) ONCE for strict causality, passed to hedged_pnl.
# ----------------------------------------------------------------------------------
def build_fn(p):
    # (A) sizing component
    if p["size"] == "static":
        size = pd.Series(1.0, index=ret.index)
    else:
        size = inv_risk_size(p["ref"], p["ref_vv"], p.get("pwr", 1.0)).reindex(ret.index)
    # (B) gate component
    if p["gate"] == "off":
        gate = pd.Series(1.0, index=ret.index)
    else:
        gate = stress_gate(p["use_vxn"], p["use_ma"], p["use_dd"], p["X"], p["Y"])
    expo = (size.fillna(0.0) * gate.fillna(0.0)).shift(1)   # CAUSAL: day-t uses t-1 info
    return H.hedged_pnl(ret, vxn, cap=p["cap"], smile=SMILE, exposure=expo)

# ----------------------------------------------------------------------------------
# GRID. WF picks the whole tuple per block on trailing-train Sharpe.
# Includes pure static / gate-off so the overlays are NOT forced.
# cap ceiling 0.06 (tail bound); no wider wing offered.
# ----------------------------------------------------------------------------------
size_opts = [dict(size="static")]
for ref in [16, 18, 20, 22]:
    for ref_vv in [1.3, 1.6, 2.0]:
        for pwr in [1.0, 1.5]:
            size_opts.append(dict(size="blend", ref=ref, ref_vv=ref_vv, pwr=pwr))

gate_opts = [dict(gate="off", use_vxn=False, use_ma=False, use_dd=False, X=0.95, Y=0.08)]
X_levels = [0.95, 0.975]
Y_levels = [0.08, 0.12]
for X in X_levels:
    gate_opts.append(dict(gate="on", use_vxn=True, use_ma=False, use_dd=False, X=X, Y=0.08))
gate_opts.append(dict(gate="on", use_vxn=False, use_ma=True, use_dd=False, X=0.95, Y=0.08))
for Y in Y_levels:
    gate_opts.append(dict(gate="on", use_vxn=False, use_ma=False, use_dd=True, X=0.95, Y=Y))
for X, Y in itertools.product(X_levels, Y_levels):
    gate_opts.append(dict(gate="on", use_vxn=True, use_ma=True, use_dd=True, X=X, Y=Y))

caps = [0.04, 0.05, 0.06]
grid = []
for cap in caps:
    for s in size_opts:
        for g in gate_opts:
            grid.append({**s, **g, "cap": cap})

print(f"grid size = {len(grid)}")

# ----------------------------------------------------------------------------------
# Walk-forward OOS, then causal vol-scale to 10%.
# ----------------------------------------------------------------------------------
oos = H.walk_forward(build_fn, grid, train=1260, test=252)
oos_scaled = H.causal_scale(oos, target=0.10)
m = H.metrics(oos_scaled)
worst = float(oos_scaled.min())
picks = oos.attrs.get("picks", [])

# Realistic static baseline.
base = H.causal_scale(H.hedged_pnl(ret, vxn, cap=0.05, smile=SMILE))
bm = H.metrics(base)

# Reference single-overlay OOS (rebuild quickly): dyn_size-only and stress-only,
# using the same cap grid but with the other overlay disabled, for honest attribution.
def wf_subset(grid_sub):
    o = H.walk_forward(build_fn, grid_sub, train=1260, test=252)
    os_ = H.causal_scale(o, target=0.10)
    return H.metrics(os_), float(os_.min())

dyn_only_grid = [g for g in grid if g["gate"] == "off"]
gate_only_grid = [g for g in grid if g["size"] == "static"]
m_dyn, w_dyn = wf_subset(dyn_only_grid)
m_gate, w_gate = wf_subset(gate_only_grid)

# Summarize picks.
from collections import Counter
def lab(p):
    s = "static" if p["size"] == "static" else f"size[ref{p['ref']}/vv{p['ref_vv']}/pwr{p['pwr']}]"
    if p["gate"] == "off":
        g = "gate-off"
    else:
        parts = []
        if p["use_vxn"]: parts.append(f"VXN>=p{int(p['X']*100)}")
        if p["use_ma"]: parts.append("NQ<200dMA")
        if p["use_dd"]: parts.append(f"dd20<=-{int(p['Y']*100)}")
        g = "+".join(parts)
    return f"cap{p['cap']}|{s}|{g}"
pc = Counter(lab(p) for p in picks)

print("=" * 74)
print("COMBINED tail-safe short-vol (inverse-risk sizing x leading-stress gate)")
print("  smile=1.5 fixed, walk-forward OOS, causal-scaled to 10% vol")
print("=" * 74)
print("OOS metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
print(f"OOS worst single day : {worst:.5f} ({worst*100:.2f}%)")
print("-" * 74)
print(f"REALISTIC STATIC BASELINE (smile=1.5, cap=0.05): Sharpe {bm['sharpe']:.4f} "
      f"maxDD {bm['maxdd']:.4f} worst {bm['worst']*100:.2f}%")
print(f"Beats baseline Sharpe?    {m['sharpe'] > bm['sharpe']}  ({m['sharpe']:.3f} vs {bm['sharpe']:.3f})")
print(f"Improves worst day?       {worst > bm['worst']}  ({worst*100:.2f}% vs {bm['worst']*100:.2f}%)")
print(f"Improves maxDD?           {m['maxdd'] > bm['maxdd']}  ({m['maxdd']:.3f} vs {bm['maxdd']:.3f})")
print("-" * 74)
print(f"ATTRIBUTION (same harness, one overlay at a time, OOS):")
print(f"  dyn-size only  : Sharpe {m_dyn['sharpe']:.4f}  worst {w_dyn*100:.2f}%  maxDD {m_dyn['maxdd']:.3f}")
print(f"  stress only    : Sharpe {m_gate['sharpe']:.4f}  worst {w_gate*100:.2f}%  maxDD {m_gate['maxdd']:.3f}")
print(f"  combined       : Sharpe {m['sharpe']:.4f}  worst {worst*100:.2f}%  maxDD {m['maxdd']:.3f}")
print("-" * 74)
print(f"WF picks (top), {len(picks)} blocks:")
for k, c in pc.most_common(10):
    print(f"   {c:3d}  {k}")
print("=" * 74)

# Cost-stress: re-run combined at higher smile to show fragility of the edge to wing price.
# build_smile rebuilds with an explicit smile so we don't rely on rebinding the global.
def build_smile(p, sm):
    if p["size"] == "static":
        size = pd.Series(1.0, index=ret.index)
    else:
        size = inv_risk_size(p["ref"], p["ref_vv"], p.get("pwr", 1.0)).reindex(ret.index)
    if p["gate"] == "off":
        gate = pd.Series(1.0, index=ret.index)
    else:
        gate = stress_gate(p["use_vxn"], p["use_ma"], p["use_dd"], p["X"], p["Y"])
    expo = (size.fillna(0.0) * gate.fillna(0.0)).shift(1)
    return H.hedged_pnl(ret, vxn, cap=p["cap"], smile=sm, exposure=expo)

print("SMILE STRESS (combined config, WF re-selected at each smile):")
for sm in [1.5, 2.0, 2.5, 3.0]:
    o = H.walk_forward(lambda p, _sm=sm: build_smile(p, _sm), grid, train=1260, test=252)
    os_ = H.causal_scale(o, target=0.10)
    mm = H.metrics(os_)
    print(f"   smile={sm}: OOS Sharpe {mm['sharpe']:.4f}  worst {float(os_.min())*100:.2f}%")
print("=" * 74)
