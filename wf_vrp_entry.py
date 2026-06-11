# -*- coding: utf-8 -*-
"""
Strategy B optimization: VRP-gated short-vol entry to PREVENT EXTREME LOSS.

Idea: only put on short-vol when implied is meaningfully above realized.
    exposure = 1  when (VXN - trailing_realized_vol) > k   else 0
    trailing_realized = ret.rolling(21).std()*sqrt(252)*100   (causal, shifted)
The threshold k in {0,1,2,3,4} vol pts is chosen OUT-OF-SAMPLE via H.walk_forward.

All signals are causal:
  - VXN is .shift(1) (yesterday's close implied, known before today's move).
  - trailing_realized uses ret.rolling(21).std() then .shift(1).
  - exposure is the AND of those, then .shift(1) once more so the gate decision
    uses only information strictly before t (belt-and-suspenders causality).
  - H.hedged_pnl multiplies pnl by this causal exposure; its wing cost is a
    trailing rolling mean with smile=1.5 markup; cap=0.05 wing strike.
  - H.causal_scale uses trailing rolling std .shift(1).
  - H.walk_forward picks k on a trailing TRAIN window, applies to the next
    OOS TEST block, and rolls -> no full-sample parameter selection.
"""
import sys
import numpy as np
import pandas as pd

HARNESS = r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab"
sys.path.insert(0, HARNESS)
import shortvol_harness as H

SQ = np.sqrt(252.0)

df, ret, vxn = H.load()

# ---- Causal VRP signal --------------------------------------------------
# trailing realized vol, annualized, in vol points (e.g. 18.0); shifted so it
# uses returns strictly before t.
trailing_rv = ret.rolling(21).std() * SQ * 100.0
trailing_rv = trailing_rv.shift(1)

# implied (VXN) known at yesterday's close -> shift(1).
vxn_lag = vxn.shift(1)

# variance risk premium proxy in vol points (all causal already).
vrp = (vxn_lag - trailing_rv)


def build(params):
    """Return causal daily short-vol P&L for a given VRP entry threshold k."""
    k = params["k"]
    # gate: only short vol when richly paid. Shift once more for strict causality.
    expo = (vrp > k).astype(float)
    expo = expo.shift(1)  # decision uses info strictly before t
    pnl = H.hedged_pnl(ret, vxn, cap=0.05, smile=1.5, exposure=expo)
    return pnl


GRID = [{"k": k} for k in [0, 1, 2, 3, 4]]

# ---- Walk-forward: parameters chosen OUT-OF-SAMPLE -----------------------
oos = H.walk_forward(build, GRID, train=1260, test=252)
picks = oos.attrs.get("picks", [])

# scale the OOS pnl to 10% vol with causal (trailing, shifted) leverage.
oos_scaled = H.causal_scale(oos, target=0.10)

m = H.metrics(oos_scaled)
worst_day = float(oos_scaled.min())

# baseline: static (always-on) smile=1.5 hedged, full history, scaled.
base = H.causal_scale(H.hedged_pnl(ret, vxn, cap=0.05, smile=1.5))
mb = H.metrics(base)

# average OOS exposure (fraction of days short-vol is on) for context.
# rebuild the actually-selected exposure path over the OOS index.
ks = pd.Series([p["k"] for p in picks])
pick_counts = ks.value_counts().sort_index().to_dict()

print("=" * 64)
print("Strategy B -- VRP-gated short-vol entry (walk-forward, OOS)")
print("=" * 64)
print(f"OOS days: {m['n']}")
print(f"OOS Sharpe      : {m['sharpe']:.4f}")
print(f"OOS maxDD       : {m['maxdd']:.4f}")
print(f"OOS worst day   : {worst_day*100:.4f} %")
print(f"OOS skew        : {m['skew']:.4f}")
print(f"OOS CAGR        : {m['cagr']:.4f}")
print("-" * 64)
print(f"Baseline (static smile=1.5) Sharpe: {mb['sharpe']:.4f}")
print(f"Beats realistic baseline (1.394)?  {m['sharpe'] > mb['sharpe']}")
print("-" * 64)
print(f"Walk-forward k picks (k -> #test-blocks chosen): {pick_counts}")
print(f"Number of test blocks: {len(picks)}")
print(f"Grid searched: k in {[g['k'] for g in GRID]} vol pts")
print("=" * 64)
