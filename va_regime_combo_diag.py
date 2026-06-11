# -*- coding: utf-8 -*-
"""Diagnostics for va_regime_combo: tail concentration, crash-day behavior, yearly stability."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import volarb_harness as H

df, ret, vxn = H.load()
fc21 = H.fcast_vol(df, ret, "park21")
fc10 = H.fcast_vol(df, ret, "park10")
fc42 = H.fcast_vol(df, ret, "park42")
richness = vxn - fc21
trend = fc10 - fc42

def make_signal(g):
    r_hi, r_lo, d = g
    s = pd.Series(0.0, index=ret.index)
    s[((richness >= r_hi) & (trend <= -d)).fillna(False)] = 1.0
    s[((richness <= r_lo) & (trend >= d)).fillna(False)] = -1.0
    return s

def build_fn(g):
    return H.backtest(make_signal(g), ret, vxn)

R_HI = [2.0, 4.0, 6.0]; R_LO = [0.0, -2.0]; D = [0.0, 1.0, 2.0]
grid = [(a, b, c) for a in R_HI for b in R_LO for c in D]
oos = H.walk_forward(build_fn, grid)
picks = oos.attrs["picks"]

# reconstruct OOS positions
pnl_idx = build_fn(grid[0]).index
train, test = 1260, 252
pos_parts = []; start, b = train, 0
while start + test <= len(pnl_idx):
    te = pnl_idx[start:start + test]
    pos = make_signal(picks[b]).clip(-1, 1).shift(1).fillna(0.0)
    pos_parts.append(pos.reindex(te)); start += test; b += 1
oos_pos = pd.concat(pos_parts).dropna()

static_pnl = H.backtest(pd.Series(1.0, index=ret.index), ret, vxn)
base = static_pnl.reindex(oos.index).dropna()

print("=== top 8 OOS pnl days (strategy) ===")
top = oos.sort_values(ascending=False).head(8)
for dt, v in top.items():
    print(f"  {dt.date()}  pnl={v:+.3e}  pos={oos_pos.loc[dt]:+.0f}  ret={ret.loc[dt]*100:+.2f}%  vxn={vxn.loc[dt]:.1f}")
print("=== bottom 8 OOS pnl days (strategy) ===")
bot = oos.sort_values().head(8)
for dt, v in bot.items():
    print(f"  {dt.date()}  pnl={v:+.3e}  pos={oos_pos.loc[dt]:+.0f}  ret={ret.loc[dt]*100:+.2f}%  vxn={vxn.loc[dt]:.1f}")

print("\n=== strategy position on static-short's 8 worst days ===")
for dt, v in base.sort_values().head(8).items():
    print(f"  {dt.date()}  static_pnl={v:+.3e}  strat_pos={oos_pos.loc[dt]:+.0f}  strat_pnl={oos.loc[dt]:+.3e}")

print("\n=== tail-sensitivity (diagnostic only; reported headline stays uncapped) ===")
for k in [1, 3, 5, 10]:
    drop = oos.sort_values(ascending=False).head(k).index
    sub = oos.drop(drop)
    m = H.metrics(sub)
    print(f"  excl top {k:>2} positive days: sharpe={m['sharpe']:.3f}  tstat={m['tstat']:.2f}  skew={m['skew']:.2f}")
m_all = H.metrics(oos)
print(f"  full OOS:                 sharpe={m_all['sharpe']:.3f}  tstat={m_all['tstat']:.2f}  skew={m_all['skew']:.2f}")

print("\n=== yearly sharpe (strategy vs static, same dates) ===")
for y, grp in oos.groupby(oos.index.year):
    bs = base.reindex(grp.index)
    shs = grp.mean() / grp.std() * H.SQ if grp.std() > 0 else float('nan')
    shb = bs.mean() / bs.std() * H.SQ if bs.std() > 0 else float('nan')
    print(f"  {y}: strat={shs:+.2f}  static={shb:+.2f}  (strat mean={grp.mean():+.2e})")

print("\n=== long-vol leg vs short-vol leg contribution (OOS) ===")
short_days = oos[oos_pos > 0]; long_days = oos[oos_pos < 0]
print(f"  short-vol leg: n={len(short_days)} total={short_days.sum():+.3e} mean={short_days.mean():+.3e}")
print(f"  long-vol leg:  n={len(long_days)} total={long_days.sum():+.3e} mean={long_days.mean():+.3e}")
ml = H.metrics(long_days) if len(long_days) > 60 else None
ms = H.metrics(short_days)
print(f"  short-leg sharpe (active days only): {ms['sharpe']:.2f}")
if ml: print(f"  long-leg sharpe (active days only): {ml['sharpe']:.2f}")

# robustness: re-run walk-forward with extra signal lag (s shifted one MORE day)
def build_fn_lag(g):
    return H.backtest(make_signal(g).shift(1), ret, vxn)
oos_lag = H.walk_forward(build_fn_lag, grid)
m_lag = H.metrics(oos_lag)
print(f"\n=== extra-lag robustness (signal delayed one MORE day) ===")
print(f"  sharpe={m_lag['sharpe']:.3f}  tstat={m_lag['tstat']:.2f}  skew={m_lag['skew']:.2f}")
