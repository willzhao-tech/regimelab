"""
Strategy B optimization: VXN-richness mean-reversion short-vol sleeve.

Idea: Sell vol proportional to how RICH VXN is vs its OWN trailing level.
    exposure_t = clip( (VXN_{t-1} - trailing_median_VXN_{t-1}) / scale , 0, 1 )
Sell more when VXN is elevated above its own trailing median (more premium to
harvest + mean-reversion tailwind); flat (exposure->0) when VXN is low /
complacent (little premium, and elevated crash-into-low-vol risk).

ALL signals causal:
  - VXN_{t-1} uses yesterday's close (shift(1)).
  - trailing_median_VXN is a rolling median over a trailing window, then .shift(1)
    so day t only sees medians computed from data strictly before t.
  - scale (the richness normalizer) is itself a trailing dispersion estimate
    (rolling MAD-like std of VXN), .shift(1). A 'scale_mult' grid knob tunes it.
  - The hedged P&L (H.hedged_pnl) already uses trailing IV (shift), trailing
    wing cost (rolling mean shift) with smile=1.5, and a daily-move cap.
  - exposure is shifted ONE more day before being handed to hedged_pnl, so the
    position on day t is decided using only info available at t-1.

Parameters (window for the median/scale, scale_mult, cap) are chosen
OUT-OF-SAMPLE by H.walk_forward (best trailing-train Sharpe -> next test block).
Final OOS pnl is scaled to 10% vol by H.causal_scale (trailing std, shifted).
"""
import sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import shortvol_harness as H

SMILE = 1.5  # OTM tail wing markup, per anti-look-ahead rules (>=1.5)

df, ret, vxn = H.load()


def build_fn(params):
    """Return a CAUSAL daily P&L Series over full history for given params.

    params: dict(window=int, scale_mult=float, cap=float)
    """
    window = params["window"]
    scale_mult = params["scale_mult"]
    cap = params["cap"]

    # --- Causal VXN richness signal -------------------------------------
    # Yesterday's VXN level (strictly past).
    vxn_lag = vxn.shift(1)

    # Trailing median of VXN over a rolling window, then shift(1): day t uses
    # only medians computed from VXN values dated <= t-1 (no t info).
    trail_med = vxn.rolling(window, min_periods=max(60, window // 4)).median().shift(1)

    # Trailing dispersion (scale) of VXN: rolling std over the same window,
    # shifted. This normalizes "richness" into a 0..1 ramp. scale_mult tunes
    # how many trailing-std's above the median saturate exposure to 1.
    trail_disp = vxn.rolling(window, min_periods=max(60, window // 4)).std().shift(1)
    scale = (scale_mult * trail_disp).replace(0.0, np.nan)

    richness = (vxn_lag - trail_med) / scale
    exposure = richness.clip(lower=0.0, upper=1.0)

    # Shift exposure one MORE day: position for day t is set from info at t-1.
    exposure = exposure.shift(1)

    pnl = H.hedged_pnl(ret, vxn, cap=cap, smile=SMILE, exposure=exposure)
    return pnl


# --- Parameter grid (chosen OOS by walk_forward) ------------------------
windows = [126, 189, 252, 504]
scale_mults = [0.5, 1.0, 1.5, 2.0]
caps = [0.04, 0.05, 0.06]
grid = [dict(window=w, scale_mult=sm, cap=c)
        for w, sm, c in itertools.product(windows, scale_mults, caps)]

# --- Walk-forward: parameters picked out-of-sample -----------------------
oos = H.walk_forward(build_fn, grid, train=1260, test=252)

# --- Scale OOS pnl to 10% annualized vol, causally -----------------------
oos_scaled = H.causal_scale(oos, target=0.10)

m = H.metrics(oos_scaled)
worst = float(oos_scaled.min())

# --- Baseline (realistic static hedged, smile=1.5) for comparison --------
base = H.causal_scale(H.hedged_pnl(ret, vxn, cap=0.05, smile=SMILE))
mb = H.metrics(base)

picks = oos.attrs.get("picks", [])
# Summarize pick frequency.
from collections import Counter
pick_counts = Counter((p["window"], p["scale_mult"], p["cap"]) for p in picks)

print("=" * 70)
print("VXN-RICHNESS MEAN-REVERSION  (OUT-OF-SAMPLE, walk-forward)")
print("=" * 70)
print(f"OOS n days          : {m['n']}")
print(f"OOS Sharpe          : {m['sharpe']:.4f}")
print(f"OOS CAGR            : {m['cagr']:.4f}")
print(f"OOS maxDD           : {m['maxdd']:.4f}")
print(f"OOS skew            : {m['skew']:.4f}")
print(f"OOS worst day       : {worst:.5f}  ({worst*100:.3f}%)")
print("-" * 70)
print(f"BASELINE Sharpe     : {mb['sharpe']:.4f}")
print(f"BASELINE maxDD      : {mb['maxdd']:.4f}")
print(f"BASELINE worst day  : {mb['worst']:.5f}  ({mb['worst']*100:.3f}%)")
print(f"BASELINE skew       : {mb['skew']:.4f}")
print("-" * 70)
beats = (m["sharpe"] > mb["sharpe"])
print(f"BEATS BASELINE (Sharpe): {beats}")
print(f"Number of WF test blocks (picks): {len(picks)}")
print("Pick frequency (window, scale_mult, cap) -> count:")
for k, c in pick_counts.most_common():
    print(f"   window={k[0]}, scale_mult={k[1]}, cap={k[2]}  ->  {c}")
print("=" * 70)
