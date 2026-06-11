# wf_dyn_size.py
# Short-vol Strategy B optimized to PREVENT EXTREME LOSS via INVERSE-RISK DYNAMIC SIZING.
#
# Idea: continuously scale exposure inversely to *current* risk so we sell LESS vol when
# vol is already high. Two causal forms, both fully shifted so day-t exposure uses only
# data strictly before t:
#   form="level": exposure = clip(ref_level / VXN_{t-1}, 0, 1)
#       -> when yesterday's VXN > ref_level, we de-gear; otherwise full size.
#   form="volvol": exposure = clip(ref_vv / volvol_{t-1}, 0, 1)
#       -> volvol = trailing std of daily VXN changes (the "vol of vol"), shifted.
#
# The reference level / form / wing-cap are ALL chosen OUT-OF-SAMPLE by H.walk_forward
# (best trailing-train Sharpe per block, applied to the next test block, rolled).
#
# Anti-look-ahead: every signal below is trailing + .shift(1). See SELF-AUDIT at bottom.

import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import shortvol_harness as H
import numpy as np
import pandas as pd

df, ret, vxn = H.load()

# ----------------------------------------------------------------------------------
# Causal exposure builders. All inputs to exposure_t are known strictly before t.
# ----------------------------------------------------------------------------------

def exposure_level(ref_level):
    # VXN_{t-1}: yesterday's close (shift(1)) -> known before t.
    vxn_lag = vxn.shift(1)
    expo = (ref_level / vxn_lag).clip(lower=0.0, upper=1.0)
    return expo  # already lagged; one value per day, indexed like vxn

def exposure_volvol(ref_vv, vv_win=21):
    # vol-of-vol = trailing std of daily VXN *changes*.
    # diff() then rolling std -> uses up to and including yesterday; .shift(1) makes it
    # strictly pre-t (exposure for day t must not see VXN_t).
    dvxn = vxn.diff()
    volvol = dvxn.rolling(vv_win, min_periods=10).std().shift(1)
    expo = (ref_vv / volvol).clip(lower=0.0, upper=1.0)
    return expo

def exposure_blend(ref_level, ref_vv, pwr=1.0, vv_win=21):
    # De-gear if EITHER the *level* is elevated OR the *vol-of-vol* is elevated.
    # Take the min of the two inverse-risk ratios (whichever screams 'risk' wins),
    # optionally raised to a power>1 for more aggressive de-gearing. Fully causal:
    # both legs use only VXN data through yesterday (shift(1)).
    vxn_lag = vxn.shift(1)
    e_lvl = (ref_level / vxn_lag).clip(lower=0.0, upper=1.0)
    dvxn = vxn.diff()
    volvol = dvxn.rolling(vv_win, min_periods=10).std().shift(1)
    e_vv = (ref_vv / volvol).clip(lower=0.0, upper=1.0)
    expo = np.minimum(e_lvl, e_vv) ** pwr
    return expo

# ----------------------------------------------------------------------------------
# build_fn(params) -> causal daily pnl over full history.
# hedged_pnl already applies the trailing+smile wing cost and lags iv via vxn.shift(1).
# exposure is passed pre-shifted (we built it lagged above), so day-t pnl uses only
# pre-t exposure. smile fixed at 1.5 per spec.
# ----------------------------------------------------------------------------------

def build_fn(params):
    form = params["form"]
    cap = params["cap"]
    if form == "level":
        expo = exposure_level(params["ref"])
    elif form == "volvol":
        expo = exposure_volvol(params["ref"], params.get("vv_win", 21))
    elif form == "blend":
        expo = exposure_blend(params["ref"], params["ref_vv"], params.get("pwr", 1.0),
                              params.get("vv_win", 21))
    elif form == "static":
        expo = pd.Series(1.0, index=vxn.index)  # plain hedged sleeve, no dynamic sizing
    else:
        raise ValueError(form)
    pnl = H.hedged_pnl(ret, vxn, cap=cap, smile=1.5, exposure=expo)
    return pnl

# ----------------------------------------------------------------------------------
# Parameter grid -- walk_forward picks among ALL of these per test block (OOS).
# ref levels span the VXN distribution (~10..80). volvol refs span typical trailing
# std of daily VXN changes. We also include a plain static sleeve so WF can decline
# to de-gear when that wins in-train (honest competition).
# ----------------------------------------------------------------------------------

grid = []
for cap in [0.04, 0.05, 0.06]:
    for ref in [16, 18, 20, 22, 25, 28, 32]:
        grid.append({"form": "level", "ref": ref, "cap": cap})
    for ref in [1.0, 1.3, 1.6, 2.0, 2.5, 3.0]:
        grid.append({"form": "volvol", "ref": ref, "vv_win": 21, "cap": cap})
    # blended inverse-risk (level AND vol-of-vol), with optional power for harder de-gear
    for ref in [16, 18, 20]:
        for ref_vv in [1.3, 1.6, 2.0]:
            for pwr in [1.0, 1.5]:
                grid.append({"form": "blend", "ref": ref, "ref_vv": ref_vv,
                             "pwr": pwr, "vv_win": 21, "cap": cap})
    grid.append({"form": "static", "cap": cap})

# ----------------------------------------------------------------------------------
# Walk-forward: OOS pnl, then scale OOS series to 10% vol causally.
# ----------------------------------------------------------------------------------

oos = H.walk_forward(build_fn, grid, train=1260, test=252)
oos_scaled = H.causal_scale(oos, target=0.10)

m = H.metrics(oos_scaled)
worst = float(oos_scaled.min())

# Baseline: realistic static hedged smile=1.5, scaled the same way (full-sample sleeve,
# but params not tuned -- this is the published ~1.394 reference).
base = H.hedged_pnl(ret, vxn, cap=0.05, smile=1.5)
base_scaled = H.causal_scale(base, target=0.10)
bm = H.metrics(base_scaled)

# Summarize WF picks.
picks = oos.attrs.get("picks", [])
from collections import Counter
def pick_key(p):
    if p["form"] == "static":
        return f"static/cap{p['cap']}"
    if p["form"] == "blend":
        return f"blend/ref{p['ref']}/vv{p['ref_vv']}/pwr{p['pwr']}/cap{p['cap']}"
    return f"{p['form']}/ref{p['ref']}/cap{p['cap']}"
pick_counts = Counter(pick_key(p) for p in picks)

print("=" * 70)
print("INVERSE-RISK DYNAMIC SIZING  (OOS, walk-forward, scaled to 10% vol)")
print("=" * 70)
print("OOS metrics:", m)
print("OOS worst single-day return: %.6f (%.3f%%)" % (worst, worst * 100))
print()
print("REALISTIC BASELINE (static smile=1.5, scaled):")
print("baseline metrics:", bm)
print("baseline worst day: %.6f (%.3f%%)" % (base_scaled.min(), base_scaled.min() * 100))
print()
print("Beats baseline Sharpe (~1.394)?", m["sharpe"] > bm["sharpe"])
print("Improves worst day vs baseline?", worst > base_scaled.min())
print("Improves maxDD vs baseline?", m["maxdd"] > bm["maxdd"])
print()
print("WF picks (count per param, %d blocks):" % len(picks))
for k, c in pick_counts.most_common():
    print("   %3d  %s" % (c, k))
print()
print("OOS span:", oos_scaled.index[0].date(), "->", oos_scaled.index[-1].date(), "n=", len(oos_scaled))
