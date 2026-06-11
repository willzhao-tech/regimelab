# va_ratio_continuous.py
# LONG-SHORT vol-arb on NQ/VXN, family: ratio_continuous.
# Signal: s_t = clip((VXN_t / fcast_t - 1) / w, -1, +1)
#   fcast_t = H.fcast_vol(kind) -> already shifted, uses data through t-1 (causal)
#   VXN_t   = close of day t (known at close t)
#   => s_t uses data through close t only; harness backtest() shifts s once more.
# Params picked ONLY via H.walk_forward (trailing 1260d train -> next 252d OOS).
# NO caps / winsorization / tail deletion anywhere: raw uncapped variance P&L.

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import volarb_harness as H

df, ret, vxn = H.load()

KINDS = ["park10", "park21", "park42", "ewma94"]
WS = [0.15, 0.25, 0.40]
GRID = [(k, w) for k in KINDS for w in WS]
print("combos tried:", len(GRID))

# --- build causal signals and full-sample pnl streams once ---
fcasts = {k: H.fcast_vol(df, ret, k) for k in KINDS}

def signal(g):
    kind, w = g
    s = ((vxn / fcasts[kind] - 1.0) / w).clip(-1.0, 1.0)
    return s

sigs = {g: signal(g) for g in GRID}
pnls = {g: H.backtest(sigs[g], ret, vxn) for g in GRID}

# --- walk-forward (param selection on trailing train only) ---
TRAIN, TEST = 1260, 252
oos = H.walk_forward(lambda g: pnls[g], GRID, train=TRAIN, test=TEST)
picks = oos.attrs["picks"]

print("\nOOS period:", oos.index[0].date(), "->", oos.index[-1].date(), " n =", len(oos))
print("\n=== OOS metrics (ratio_continuous, walk-forward) ===")
m = H.metrics(oos)
for k, v in m.items():
    print(f"  {k:12s} {v:.4f}" if isinstance(v, float) else f"  {k:12s} {v}")

# --- static always-short baseline on the SAME OOS dates ---
s_static = pd.Series(1.0, index=ret.index)
pnl_static_full = H.backtest(s_static, ret, vxn)
pnl_static = pnl_static_full.reindex(oos.index).dropna()
print("\n=== Baseline static s=+1 on SAME OOS dates ===")
mb = H.metrics(pnl_static)
for k, v in mb.items():
    print(f"  {k:12s} {v:.4f}" if isinstance(v, float) else f"  {k:12s} {v}")

# --- reconstruct held positions over OOS blocks (same loop as walk_forward) ---
idx = pnls[GRID[0]].index
pos_parts = []
start, k = TRAIN, 0
while start + TEST <= len(idx):
    te = idx[start:start + TEST]
    g = picks[k]
    pos = sigs[g].reindex(ret.index).clip(-1, 1).shift(1).fillna(0.0)  # exactly as harness holds it
    pos_parts.append(pos.reindex(te))
    start += TEST
    k += 1
pos_oos = pd.concat(pos_parts).reindex(oos.index)

pct_short = float((pos_oos > 0).mean())   # short-vol days (s>0: VXN rich)
pct_long = float((pos_oos < 0).mean())    # long-vol days  (s<0: VXN cheap)
pct_flat = float((pos_oos == 0).mean())
print("\n=== Position mix on OOS days (held position) ===")
print(f"  %short-vol  {pct_short*100:.1f}%")
print(f"  %long-vol   {pct_long*100:.1f}%")
print(f"  %flat       {pct_flat*100:.1f}%")
print(f"  mean pos    {float(pos_oos.mean()):.3f}   mean |pos| {float(pos_oos.abs().mean()):.3f}")

# --- timing alpha: regress OOS pnl on static-short pnl (same dates) ---
common = oos.index.intersection(pnl_static.index)
y = oos.reindex(common).values
x = pnl_static.reindex(common).values
X = np.column_stack([np.ones_like(x), x])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
dof = len(y) - 2
covb = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
alpha, b = beta
alpha_t = alpha / np.sqrt(covb[0, 0])
# Newey-West (10 lags) on the alpha as robustness
L = 10
u = resid
S = np.zeros((2, 2))
Xu = X * u[:, None]
S += Xu.T @ Xu
for l in range(1, L + 1):
    G = Xu[l:].T @ Xu[:-l]
    S += (1 - l / (L + 1)) * (G + G.T)
XtXi = np.linalg.inv(X.T @ X)
cov_nw = XtXi @ S @ XtXi
alpha_t_nw = alpha / np.sqrt(cov_nw[0, 0])
print("\n=== Timing-alpha regression: oos_pnl = a + b * static_pnl ===")
print(f"  beta        {b:.3f}")
print(f"  alpha(ann)  {alpha*252:.6f} var-units/yr")
print(f"  alpha tstat {alpha_t:.2f}  (OLS)   {alpha_t_nw:.2f}  (NW-10)")

# --- picks summary ---
print("\n=== Walk-forward picks per 252d block ===")
from collections import Counter
for i, g in enumerate(picks):
    blk_start = idx[TRAIN + i * TEST]
    print(f"  block {i+1:2d} starting {blk_start.date()}: kind={g[0]:7s} w={g[1]}")
cnt = Counter(picks)
print("  pick counts:", dict(cnt))

# --- look-ahead self-audit ---
print("""
=== Look-ahead self-audit ===
1. fcast_vol returns v.shift(1): forecast at t uses OHLC/ret through t-1. PASS
2. VXN_t is the day-t close; signal s_t = f(VXN_t, fcast_t) uses nothing past close t. PASS
3. backtest() shifts s again (pos_t = s_{t-1}) before applying to day-t variance P&L. PASS
4. Grid values (kind, w) are fixed a priori; selection is H.walk_forward trailing-train
   Sharpe only; reported series is concatenated next-block OOS. PASS
5. No P&L capping/winsorizing/tail deletion; clip is on the SIGNAL in [-1,1] (position
   sizing), the variance P&L itself is uncapped -- 2008-10-13-type days hit in full. PASS
6. No full-sample statistics (costs, scales, thresholds) enter the signal; w is a fixed
   constant per combo, not fitted. Cost model is the harness default (trailing VXN). PASS
""")
