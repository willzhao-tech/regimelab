"""
va_vrp_threshold.py -- LONG-SHORT vol-arb on NQ/VXN, family: vrp_threshold.

Discrete VRP gate:
    vrp_t = VXN_t - fcast_t          (fcast = H.fcast_vol, already shifted: uses data thru t-1)
    s_t = +1 (short vol)  if vrp_t >  k1
    s_t = -1 (long  vol)  if vrp_t < -k2
    s_t =  0              otherwise (incl. fcast warm-up NaNs)

Grid: kind in {park21, ewma94, cc21} x k1 in {1,2,4} x k2 in {1,2,4}  -> 27 combos.
Param selection: H.walk_forward ONLY (trailing 1260d train pick, next 252d OOS).

CAUSALITY AUDIT (in-code):
  - fcast_vol returns v.shift(1): value at t uses OHLC/ret data through t-1.
  - VXN_t is the close of day t -> s_t uses data through close t ONLY.
  - H.backtest shifts s again (pos_t = s_{t-1}) before applying to (iv - rvar)_t.
  - No caps / clips / winsorization anywhere: harness P&L is uncapped variance P&L.
  - No full-sample statistics enter the signal; thresholds are a fixed a-priori grid.
"""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
from collections import Counter
import volarb_harness as H

df, ret, vxn = H.load()

KINDS = ("park21", "ewma94", "cc21")
K1S = (1.0, 2.0, 4.0)
K2S = (1.0, 2.0, 4.0)
GRID = [(k, k1, k2) for k in KINDS for k1 in K1S for k2 in K2S]
print("combos tried:", len(GRID))

fcasts = {k: H.fcast_vol(df, ret, k) for k in KINDS}

def make_signal(g):
    kind, k1, k2 = g
    vrp = vxn - fcasts[kind]            # data through close t only (fcast pre-shifted)
    s = pd.Series(0.0, index=ret.index)
    s[vrp > k1] = 1.0                   # rich implied vol -> short vol
    s[vrp < -k2] = -1.0                 # cheap implied vol -> long vol
    s[vrp.isna()] = 0.0                 # warm-up: stay flat
    return s

signals = {g: make_signal(g) for g in GRID}
pnls = {g: H.backtest(signals[g], ret, vxn) for g in GRID}

def build_fn(g):
    return pnls[g]

oos = H.walk_forward(build_fn, GRID)    # trailing-train pick, next-block OOS
picks = oos.attrs["picks"]

# ---- reconstruct the stitched OOS signal (same block logic as walk_forward) ----
TRAIN, TEST = 1260, 252
idx = pnls[GRID[0]].index
stitched = []
start, b = TRAIN, 0
while start + TEST <= len(idx):
    te = idx[start:start + TEST]
    stitched.append(signals[picks[b]].reindex(te))
    start += TEST
    b += 1
s_oos = pd.concat(stitched).reindex(oos.index)

# ---- baseline: static s=+1 (always short vol) on the SAME OOS dates ----
s_static = pd.Series(1.0, index=ret.index)
pnl_static_full = H.backtest(s_static, ret, vxn)
base = pnl_static_full.reindex(oos.index).dropna()

m_oos = H.metrics(oos)
m_base = H.metrics(base)

print("\nOOS period:", oos.index[0].date(), "->", oos.index[-1].date(), " n =", len(oos))
print("\n== OOS metrics (vrp_threshold walk-forward) ==")
for k, v in m_oos.items():
    print(f"  {k:12s} {v: .4f}" if isinstance(v, float) else f"  {k:12s} {v}")
print("\n== baseline static s=+1 on SAME OOS dates ==")
for k, v in m_base.items():
    print(f"  {k:12s} {v: .4f}" if isinstance(v, float) else f"  {k:12s} {v}")

pct_short = float((s_oos == 1.0).mean() * 100)
pct_long = float((s_oos == -1.0).mean() * 100)
pct_flat = float((s_oos == 0.0).mean() * 100)
print(f"\nOOS days short-vol (s=+1): {pct_short:.1f}%")
print(f"OOS days long-vol  (s=-1): {pct_long:.1f}%")
print(f"OOS days flat      (s= 0): {pct_flat:.1f}%")

# ---- timing alpha vs the static-short stream (OLS: oos = a + b*static) ----
x = base.values
y = oos.reindex(base.index).values
X = np.column_stack([np.ones_like(x), x])
beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
e = y - X @ beta
dof = len(y) - 2
s2 = float(e @ e) / dof
covb = s2 * np.linalg.inv(X.T @ X)
a, bcoef = beta
ta = a / np.sqrt(covb[0, 0])
print(f"\ntiming-alpha regression vs static short:  alpha/day = {a:.3e}  "
      f"t(alpha) = {ta:.2f}   beta = {bcoef:.3f}")
print(f"annualized alpha (units of variance pnl): {a*252:.4e}")

print("\n== walk-forward picks per 252d block ==")
for i, p in enumerate(picks):
    print(f"  block {i+1:2d}: kind={p[0]:7s} k1={p[1]:.0f} k2={p[2]:.0f}")
print("\npick frequencies:", dict(Counter(picks)))

print("\nLOOK-AHEAD SELF-AUDIT:")
print("  1. fcast_vol is shift(1) inside harness -> fcast_t uses data thru t-1.")
print("  2. s_t = f(VXN_close_t, fcast_t): data thru close t only; harness shifts s")
print("     again, so day-t P&L uses position decided at close t-1. No look-ahead.")
print("  3. Thresholds (k1,k2) are a fixed a-priori grid; (kind,k1,k2) chosen per")
print("     block by H.walk_forward on TRAILING train Sharpe only.")
print("  4. No caps/clips/winsorization: uncapped variance P&L incl. 2008-10-13 etc.")
print("  5. No full-sample quantities (costs, scales, caps) enter the signal.")
