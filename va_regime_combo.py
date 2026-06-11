# -*- coding: utf-8 -*-
"""
va_regime_combo.py — LONG-SHORT vol-arb on NQ/VXN, family: regime_combo.

Idea: combine the two strongest causal facts:
  (1) richness  = VXN_t - park21 trailing forecast (forecast uses data through t-1)
  (2) vol-trend = park10 fcast - park42 fcast (short-window range vol vs long-window;
                  both trailing, shifted by the harness fcast_vol itself)

Rules (s in [-1,+1]; +1 = SHORT vol, -1 = LONG vol, 0 = flat):
  SHORT vol only when rich  (richness >= r_hi) AND vol falling (trend <= -d)
  LONG  vol only when cheap (richness <= r_lo) AND vol rising  (trend >= +d)
  else flat.

Causality: s_t uses VXN close at t plus fcast_vol values at t (which themselves
use data only through t-1). The harness backtest() additionally shifts s by one
day, so the position on day t is s_{t-1}. No caps, no winsorizing, no tail edits.
Params picked ONLY by H.walk_forward (trailing 1260d train -> next 252d OOS).
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import volarb_harness as H

df, ret, vxn = H.load()

# ---- causal building blocks (computed ONCE on full data; each value at t uses <= t) ----
fc21 = H.fcast_vol(df, ret, "park21")   # value at t uses data through t-1
fc10 = H.fcast_vol(df, ret, "park10")
fc42 = H.fcast_vol(df, ret, "park42")

richness = vxn - fc21                   # VXN close t (known at t) minus trailing fcast
trend    = fc10 - fc42                  # >0: range vol rising; <0: falling


def make_signal(g):
    r_hi, r_lo, d = g
    s = pd.Series(0.0, index=ret.index)
    short_mask = (richness >= r_hi) & (trend <= -d)
    long_mask  = (richness <= r_lo) & (trend >= d)
    s[short_mask.fillna(False)] = 1.0
    s[long_mask.fillna(False)] = -1.0
    return s


def build_fn(g):
    return H.backtest(make_signal(g), ret, vxn)


# ---- small grid over thresholds ----
R_HI = [2.0, 4.0, 6.0]        # richness (vol pts) required to short vol
R_LO = [0.0, -2.0]            # cheapness (vol pts) required to go long vol
D    = [0.0, 1.0, 2.0]        # vol-trend buffer (vol pts)
grid = [(a, b, c) for a in R_HI for b in R_LO for c in D]
print(f"combos tried: {len(grid)}")

# ---- walk-forward (trailing-train pick, next-block OOS) ----
oos = H.walk_forward(build_fn, grid)
picks = oos.attrs["picks"]

print("\n=== OOS metrics (regime_combo, walk-forward) ===")
m = H.metrics(oos)
for k, v in m.items():
    print(f"  {k}: {v}")

# ---- baseline: static s=+1 on the SAME OOS dates ----
static_pnl = H.backtest(pd.Series(1.0, index=ret.index), ret, vxn)
base = static_pnl.reindex(oos.index).dropna()
print("\n=== baseline static s=+1 on SAME OOS dates ===")
mb = H.metrics(base)
for k, v in mb.items():
    print(f"  {k}: {v}")

# ---- reconstruct OOS positions to report long/short/flat fractions ----
# replicate walk_forward's block structure exactly (train=1260, test=252) on the
# pnl index, then take the picked param's HARNESS position (s shifted by 1).
pnl_idx = build_fn(grid[0]).index
train, test = 1260, 252
pos_parts = []
start, b = train, 0
while start + test <= len(pnl_idx):
    te = pnl_idx[start:start + test]
    g = picks[b]
    pos = make_signal(g).reindex(ret.index).clip(-1, 1).shift(1).fillna(0.0)
    pos_parts.append(pos.reindex(te))
    start += test
    b += 1
oos_pos = pd.concat(pos_parts).dropna()
n = len(oos_pos)
pct_short = float((oos_pos > 0).sum()) / n * 100.0
pct_long  = float((oos_pos < 0).sum()) / n * 100.0
pct_flat  = float((oos_pos == 0).sum()) / n * 100.0
print(f"\nOOS position days: n={n}  short-vol: {pct_short:.2f}%  "
      f"long-vol: {pct_long:.2f}%  flat: {pct_flat:.2f}%")

# ---- timing-alpha regression: oos = a + b * static (same dates) ----
yx = pd.concat([oos, base], axis=1, keys=["y", "x"]).dropna()
x = yx["x"].values
y = yx["y"].values
X = np.column_stack([np.ones(len(x)), x])
beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
dof = len(y) - 2
sigma2 = float(resid @ resid) / dof
covb = sigma2 * np.linalg.inv(X.T @ X)
t_alpha = beta[0] / np.sqrt(covb[0, 0])
print(f"\ntiming-alpha regression vs static-short (same OOS dates): "
      f"alpha={beta[0]:.3e} (daily var units), t(alpha)={t_alpha:.2f}, beta={beta[1]:.3f}")

# ---- picks summary ----
from collections import Counter
cnt = Counter(picks)
print("\nwalk-forward picks (r_hi, r_lo, d) -> blocks:")
for g, c in sorted(cnt.items(), key=lambda kv: -kv[1]):
    print(f"  {g}: {c}")
print(f"blocks: {len(picks)}")

# ---- look-ahead self-audit (programmatic truncation test) ----
# Rebuild the signal using ONLY data up to a cutoff and check the value at the
# cutoff date equals the full-sample value. Any look-ahead would break equality.
print("\n=== look-ahead self-audit ===")
ok_all = True
test_params = [grid[0], grid[len(grid) // 2], grid[-1]]
for k in [2000, 4000, len(ret) - 10]:
    df2, ret2, vxn2 = df.iloc[:k], ret.iloc[:k], vxn.iloc[:k]
    f21 = H.fcast_vol(df2, ret2, "park21")
    f10 = H.fcast_vol(df2, ret2, "park10")
    f42 = H.fcast_vol(df2, ret2, "park42")
    rich2 = vxn2 - f21
    tr2 = f10 - f42
    for g in test_params:
        r_hi, r_lo, d = g
        s2 = pd.Series(0.0, index=ret2.index)
        s2[((rich2 >= r_hi) & (tr2 <= -d)).fillna(False)] = 1.0
        s2[((rich2 <= r_lo) & (tr2 >= d)).fillna(False)] = -1.0
        full = make_signal(g)
        same = bool((s2.iloc[-5:] == full.reindex(s2.index).iloc[-5:]).all())
        ok_all &= same
print(f"truncation test (signal at t unchanged when future data removed): "
      f"{'PASS' if ok_all else 'FAIL'}")
print("audit notes:")
print("  - fcast_vol is harness-shifted (uses data through t-1); VXN used at close t;")
print("    harness backtest() shifts s again, so trade is on t+1 information-wise.")
print("  - thresholds are a fixed grid; selection ONLY via H.walk_forward trailing train.")
print("  - NO caps/clips/winsorization of P&L; uncapped variance-swap P&L incl. 2008-10-13.")
print("  - costs: harness default 0.5 volpt per unit turnover, charged on position changes.")
