# -*- coding: utf-8 -*-
"""
Strategy B optimized to PREVENT EXTREME LOSS via CAUSAL leading-stress de-risking.

Idea: short-vol (hedged_pnl, smile=1.5) but GO FLAT (exposure=0) the day a leading
stress signal fires, to dodge vol-spike days. Three causal leading signals:
  (1) VXN 1-day change above its TRAILING-1y X-percentile   (vol-of-vol spike)
  (2) NQ below its 200d moving average                       (trend break)
  (3) NQ trailing-20d drawdown < -Y%                          (momentum stress)
Exposure = 0 if ANY enabled trigger fires, else 1.

EVERYTHING is causal:
  - all signals .shift(1) so day-t exposure uses only info strictly before t
  - the VXN percentile threshold is an EXPANDING/ROLLING trailing-1y quantile, .shift(1)
    (never a full-sample quantile)
  - the 200d MA and 20d drawdown use only past prices, .shift(1)
  - which signals + thresholds (X, Y) are chosen OUT-OF-SAMPLE by H.walk_forward
    on a trailing-train window, applied to the next test block, rolled.
  - wing/hedge cost is the harness trailing+smile(1.5) estimate
  - final OOS pnl scaled to 10% vol with H.causal_scale (trailing std, shifted)

We report OUT-OF-SAMPLE metrics from walk_forward, never in-sample.
"""
import sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import shortvol_harness as H

df, ret, vxn = H.load()
close = df["Close"]

# -----------------------------------------------------------------------------
# Build CAUSAL leading-stress component signals (all values known strictly < t).
# We compute the raw indicators on full history (that is fine: each value at date d
# uses only prices/vxn up to d), then .shift(1) so the exposure for day t reads the
# signal as of t-1. Thresholds X/Y are NOT chosen here -- walk_forward chooses them.
# -----------------------------------------------------------------------------

# (1) VXN 1-day change vs its TRAILING-1y percentile.
#     vchg_d = level change of VXN from d-1 to d. trailing percentile rank of vchg_d
#     within the prior ~252 obs (expanding-capped rolling window), computed causally.
vchg = vxn.diff()
# rolling trailing-1y rank of the latest change as a percentile in [0,1].
# rank of last element within the window / window_size  -> strictly uses the window
# that ENDS at d (includes d itself); we .shift(1) the resulting boolean below so the
# decision for day t only sees ranks up to t-1.
def trailing_pctile(s, win=252, minp=60):
    # percentile rank of the CURRENT value within the trailing window ending at it
    return s.rolling(win, min_periods=minp).apply(
        lambda w: (w[:-1] < w[-1]).mean() if np.isfinite(w[-1]) else np.nan, raw=True)
vchg_pct = trailing_pctile(vchg, 252, 60)   # in [0,1], causal (window ends at d)

# (2) NQ below its 200d MA: ma200_d uses Close up to d. ratio<1 => below.
ma200 = close.rolling(200, min_periods=100).mean()
below_ma_ratio = close / ma200          # <1 means below 200d MA

# (3) NQ trailing-20d drawdown: (close / trailing-20d-rolling-max) - 1, <= -Y triggers.
roll_max20 = close.rolling(20, min_periods=10).max()
dd20 = close / roll_max20 - 1.0         # <=0, more negative = deeper recent drawdown

# Align all to the ret index (the harness pnl index).
vchg_pct       = vchg_pct.reindex(ret.index)
below_ma_ratio = below_ma_ratio.reindex(ret.index)
dd20           = dd20.reindex(ret.index)

# -----------------------------------------------------------------------------
# Parameter grid for WALK-FORWARD. Each param dict enables a subset of signals and
# sets thresholds X (VXN pctile) and Y (20d drawdown). 'none' baseline = always-on
# short-vol (exposure=1) so walk_forward can DECLINE to de-risk if it doesn't help OOS.
# -----------------------------------------------------------------------------
X_levels = [0.90, 0.95, 0.975]     # VXN-change trailing percentile trigger
Y_levels = [0.05, 0.08, 0.12]      # 20d drawdown depth trigger (as positive frac)

grid = []
# always-on (no de-risk) reference point inside the grid
grid.append(dict(use_vxn=False, use_ma=False, use_dd=False, X=0.95, Y=0.08))
# single-signal variants
for X in X_levels:
    grid.append(dict(use_vxn=True, use_ma=False, use_dd=False, X=X, Y=0.08))
grid.append(dict(use_vxn=False, use_ma=True, use_dd=False, X=0.95, Y=0.08))
for Y in Y_levels:
    grid.append(dict(use_vxn=False, use_ma=False, use_dd=True, X=0.95, Y=Y))
# pairwise + triple combos (OR logic) at representative thresholds
for X, Y in itertools.product(X_levels, Y_levels):
    grid.append(dict(use_vxn=True,  use_ma=True,  use_dd=False, X=X, Y=Y))   # vxn|ma
    grid.append(dict(use_vxn=True,  use_ma=False, use_dd=True,  X=X, Y=Y))   # vxn|dd
    grid.append(dict(use_vxn=False, use_ma=True,  use_dd=True,  X=X, Y=Y))   # ma|dd
    grid.append(dict(use_vxn=True,  use_ma=True,  use_dd=True,  X=X, Y=Y))   # all three

def build_fn(p):
    """Return a CAUSAL daily pnl series for the given params. exposure in {0,1},
    .shift(1) applied so day-t exposure uses only signals as of t-1."""
    trig = pd.Series(False, index=ret.index)
    if p["use_vxn"]:
        trig = trig | (vchg_pct >= p["X"])
    if p["use_ma"]:
        trig = trig | (below_ma_ratio < 1.0)
    if p["use_dd"]:
        trig = trig | (dd20 <= -p["Y"])
    # NaN warmup -> treat as "no trigger" (stay invested) but those early rows are
    # dropped by hedged_pnl/walk_forward warmup anyway.
    trig = trig.fillna(False)
    exposure = (~trig).astype(float)          # 1 = invested, 0 = flat
    exposure = exposure.shift(1)              # CAUSAL: decision for t uses t-1 signals
    pnl = H.hedged_pnl(ret, vxn, cap=0.05, smile=1.5, exposure=exposure)
    return pnl

# -----------------------------------------------------------------------------
# WALK-FORWARD: pick params on trailing-train Sharpe, apply OOS, roll.
# -----------------------------------------------------------------------------
oos = H.walk_forward(build_fn, grid, train=1260, test=252)
oos_scaled = H.causal_scale(oos, target=0.10)

m = H.metrics(oos_scaled)
worst_day = float(oos_scaled.min())

# Realistic baseline for comparison (static hedged, smile=1.5, scaled to 10% vol).
base = H.hedged_pnl(ret, vxn, cap=0.05, smile=1.5)
base_scaled = H.causal_scale(base, target=0.10)
bm = H.metrics(base_scaled)

# Summarize the walk-forward picks.
picks = oos.attrs.get("picks", [])
def pick_label(p):
    parts = []
    if p["use_vxn"]: parts.append(f"VXN>=p{int(p['X']*100)}")
    if p["use_ma"]:  parts.append("NQ<200dMA")
    if p["use_dd"]:  parts.append(f"dd20<=-{int(p['Y']*100)}%")
    return "+".join(parts) if parts else "always-on"
from collections import Counter
pick_counts = Counter(pick_label(p) for p in picks)

print("="*70)
print("WALK-FORWARD OUT-OF-SAMPLE  (stress de-risk short-vol, smile=1.5)")
print("="*70)
print("grid size:", len(grid), " train=1260 test=252  n_test_blocks:", len(picks))
print("OOS metrics:", {k: round(v,4) if isinstance(v,float) else v for k,v in m.items()})
print("OOS worst single-day return: {:.4%}".format(worst_day))
print("-"*70)
print("REALISTIC BASELINE (static smile=1.5):",
      {k: round(v,4) if isinstance(v,float) else v for k,v in bm.items()})
print("beats baseline (Sharpe):", bool(m["sharpe"] > bm["sharpe"]),
      f"({m['sharpe']:.3f} vs {bm['sharpe']:.3f})")
print("-"*70)
print("WALK-FORWARD PICKS (config -> # of 252-day test blocks chosen):")
for lab, c in pick_counts.most_common():
    print(f"   {lab:>30s} : {c}")
print("="*70)
