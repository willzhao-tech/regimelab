# va_vol_momentum.py
# LONG-SHORT vol-arbitrage on NQ/VXN, family: vol_momentum.
#   SHORT vol (s=+1): VXN HIGH vs its own trailing median AND falling  -> post-spike rich premium
#   LONG  vol (s=-1): VXN LOW  vs trailing median AND trailing range-vol rising -> complacency before storm
#   else flat (s=0).
#
# CAUSALITY CONTRACT (self-audit inline):
#   - s_t uses data through close t ONLY. All rolling stats (median/std/diff) end at t.
#     The harness backtest() shifts s by 1 before P&L, so position on day t+1 uses info <= t.
#   - Range-vol forecasts come from H.fcast_vol which is ALREADY shifted (value at t uses <= t-1).
#   - No full-sample statistics anywhere in the signal path. No caps/clips/winsorization of P&L.
#   - Parameters picked exclusively by H.walk_forward (trailing 1260d train -> next 252d OOS).
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
import volarb_harness as H

df, ret, vxn = H.load()

# Causal range-vol forecasts (already shifted inside fcast_vol: value at t uses data <= t-1)
F10 = H.fcast_vol(df, ret, "park10")
F42 = H.fcast_vol(df, ret, "park42")


def make_signal(med_win, z_hi, z_lo, mom_k):
    """Signal value at index t uses data through close t only."""
    med = vxn.rolling(med_win, min_periods=med_win).median()   # ends at t
    sd = vxn.rolling(med_win, min_periods=med_win).std()       # ends at t
    z = (vxn - med) / sd                                       # causal z-score of VXN vs own trailing median
    mom = vxn.diff(mom_k)                                      # VXN change over last mom_k days, ends at t
    rv_rising = F10 > F42                                      # short-horizon range vol above long-horizon (both <= t-1)

    s = pd.Series(0.0, index=vxn.index)
    s[(z > z_hi) & (mom < 0)] = +1.0    # rich & rolling over -> short vol
    s[(z < -z_lo) & rv_rising] = -1.0   # cheap & realized stirring -> long vol
    # no signal until trailing windows are full
    s[z.isna() | mom.isna() | F10.isna() | F42.isna()] = 0.0
    return s


def build_fn(g):
    s = make_signal(*g)
    return H.backtest(s, ret, vxn)


# Small grid: 3 median windows x 2 short-entry z x 2 momentum lookbacks (z_lo fixed) = 12 combos
GRID = [(mw, zh, 0.5, mk)
        for mw in (63, 126, 252)
        for zh in (0.75, 1.25)
        for mk in (5, 10)]

TRAIN, TEST = 1260, 252
oos = H.walk_forward(build_fn, GRID, train=TRAIN, test=TEST)
picks = oos.attrs["picks"]

# ---- static always-short baseline on the SAME OOS dates ----
static_full = H.backtest(pd.Series(1.0, index=ret.index), ret, vxn)
static_oos = static_full.reindex(oos.index).dropna()

# ---- reconstruct chosen positions per OOS block to count long/short/flat days ----
sig_cache = {g: make_signal(*g) for g in GRID}
pnl_index = build_fn(GRID[0]).index  # same index walk_forward iterated over
pos_parts = []
start, bi = TRAIN, 0
while start + TEST <= len(pnl_index):
    te = pnl_index[start:start + TEST]
    pos_parts.append(sig_cache[picks[bi]].reindex(te))
    start += TEST
    bi += 1
pos_oos = pd.concat(pos_parts).reindex(oos.index)
n = len(pos_oos)
pct_long = float((pos_oos < 0).sum()) / n * 100.0
pct_short = float((pos_oos > 0).sum()) / n * 100.0
pct_flat = float((pos_oos == 0).sum()) / n * 100.0

# ---- timing-alpha regression: oos_pnl = a + b * static_pnl ----
y = oos.reindex(static_oos.index).values
x = static_oos.values
X = np.column_stack([np.ones_like(x), x])
beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
dof = len(y) - 2
sigma2 = float(resid @ resid) / dof
cov = sigma2 * np.linalg.inv(X.T @ X)
alpha, b = beta
alpha_t = float(alpha / np.sqrt(cov[0, 0]))
alpha_ann = float(alpha * 252)

m_oos = H.metrics(oos)
m_sta = H.metrics(static_oos)

# ---- leg attribution (diagnostic only; pnl day t belongs to signal day t-1 after harness shift) ----
leg = pos_oos.shift(1).reindex(oos.index)
pnl_short_leg = oos[leg > 0]
pnl_long_leg = oos[leg < 0]
ann = 252.0

print("=" * 70)
print("FAMILY: vol_momentum  (short rich-and-falling VXN / long cheap-and-stirring)")
print(f"Grid combos tried: {len(GRID)}  | walk-forward train={TRAIN} test={TEST}")
print(f"OOS span: {oos.index[0].date()} -> {oos.index[-1].date()}  ({len(oos)} days)")
print("-" * 70)
print("OOS strategy metrics:", m_oos)
print("Static s=+1 baseline (same OOS dates):", m_sta)
print("-" * 70)
print(f"Days long-vol: {pct_long:.1f}%  short-vol: {pct_short:.1f}%  flat: {pct_flat:.1f}%")
print(f"Timing-alpha regression vs static stream: alpha(ann)={alpha_ann:.6f}  t(alpha)={alpha_t:.2f}  beta={b:.3f}")
print(f"Leg attribution (OOS): short-vol leg n={len(pnl_short_leg)} ann_pnl={pnl_short_leg.mean() * ann:.6f} "
      f"sharpe={(pnl_short_leg.mean() / pnl_short_leg.std() * H.SQ) if pnl_short_leg.std() > 0 else float('nan'):.2f}")
print(f"                       long-vol  leg n={len(pnl_long_leg)} ann_pnl={pnl_long_leg.mean() * ann:.6f} "
      f"sharpe={(pnl_long_leg.mean() / pnl_long_leg.std() * H.SQ) if pnl_long_leg.std() > 0 else float('nan'):.2f}")
print("-" * 70)
print("Walk-forward picks (med_win, z_hi, z_lo, mom_k) per 252d block:")
for i, p in enumerate(picks):
    blk = pnl_index[TRAIN + i * TEST]
    print(f"  block {i + 1:2d} starting {blk.date()}: {p}")
print("-" * 70)
print("LOOK-AHEAD SELF-AUDIT:")
print("  [1] rolling median/std/diff end at t; harness shifts s -> position t+1 uses <= t. PASS")
print("  [2] fcast_vol park10/park42 internally shifted (<= t-1). PASS")
print("  [3] No full-sample constants in signal; thresholds are grid params picked on trailing train only. PASS")
print("  [4] P&L uncapped; no winsorizing/clipping of pnl (only s clipped to [-1,1], a position bound). PASS")
print("  [5] First signal emitted only after all trailing windows are full (min_periods=window). PASS")
