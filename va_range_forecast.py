# va_range_forecast.py
# LONG-SHORT vol-arb on NQ/VXN, family: range_forecast.
#
# Idea: today's intraday range predicts tomorrow's |ret| (corr ~0.47).
# The variance-swap strike for day t+1 is set at VXN close of day t (harness iv uses
# vxn.shift(1)), so the implied daily breakeven move for tomorrow's position is
#     be_t = VXN_t / sqrt(252)   (in %)
# Signal (computed with data through close t only; harness shifts it):
#     s_t = +1 (short vol)  if range_t < b1 * be_t   (tape quieter than implied)
#     s_t = -1 (long  vol)  if range_t > b2 * be_t   (tape hotter than implied)
#     s_t =  0 otherwise
# Grid: b1 in {0.8,1.0,1.2} x b2 in {1.3,1.6,2.0} -> 9 combos, picked via
# H.walk_forward only (trailing 1260d train -> next 252d OOS).
#
# CAUSALITY AUDIT (in-code, also verified programmatically below):
#  - range_t = log(High_t/Low_t): known at close t.
#  - VXN_t: known at close t. No shift needed because backtest() shifts s itself.
#  - No full-sample statistic enters the signal (thresholds are fixed multiples
#    of a same-day observable; param selection is walk-forward trailing-train).
#  - No capping/clipping/winsorizing of P&L anywhere; harness P&L is uncapped.

import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
import volarb_harness as H

SQ = np.sqrt(252.0)

df, ret, vxn = H.load()

# observables at close t
rng = np.log(df["High"] / df["Low"]) * 100.0          # daily range, %
be = vxn / SQ                                          # implied daily breakeven, %

# sanity: report the motivating correlation, computed causally (range_t vs |ret|_{t+1})
c = rng.corr(ret.abs().shift(-1))
print("corr(range_t, |ret|_(t+1)) = %.3f" % c)


def make_signal(b1, b2):
    s = pd.Series(0.0, index=ret.index)
    ok = rng.notna() & be.notna()
    s[ok & (rng < b1 * be)] = 1.0    # quiet tape -> sell vol
    s[ok & (rng > b2 * be)] = -1.0   # hot tape -> buy vol
    return s


GRID = [(b1, b2) for b1 in (0.8, 1.0, 1.2) for b2 in (1.3, 1.6, 2.0)]


def build(g):
    b1, b2 = g
    return H.backtest(make_signal(b1, b2), ret, vxn)


# ---- walk-forward ----
oos = H.walk_forward(build, GRID)
picks = oos.attrs["picks"]
n_combos = len(GRID)

m = H.metrics(oos)
print("\n=== OOS (walk-forward) metrics, range_forecast ===")
for k, v in m.items():
    print("  %-12s %s" % (k, v))

# ---- baseline: static always-short s=+1 on the SAME OOS dates ----
s_static = pd.Series(1.0, index=ret.index)
pnl_static = H.backtest(s_static, ret, vxn)
base = pnl_static.reindex(oos.index).dropna()
mb = H.metrics(base)
print("\n=== Baseline static s=+1 on same OOS dates ===")
for k, v in mb.items():
    print("  %-12s %s" % (k, v))

# ---- reconstruct OOS held positions (signal shifted, as the harness trades it) ----
pnl0 = build(GRID[0])
idx = pnl0.index
train, test = 1260, 252
pos_parts = []
start, k = train, 0
while start + test <= len(idx):
    te = idx[start:start + test]
    b1, b2 = picks[k]
    pos = make_signal(b1, b2).shift(1)   # position actually held on day t
    pos_parts.append(pos.reindex(te))
    k += 1
    start += test
oos_pos = pd.concat(pos_parts).reindex(oos.index)
pct_short = float((oos_pos == 1.0).mean() * 100)   # short vol
pct_long = float((oos_pos == -1.0).mean() * 100)   # long vol
pct_flat = float((oos_pos == 0.0).mean() * 100)
print("\nOOS days: short-vol %.1f%% | long-vol %.1f%% | flat %.1f%%"
      % (pct_short, pct_long, pct_flat))

# ---- timing alpha vs the static short-vol stream (OLS on same OOS dates) ----
y = oos.reindex(base.index)
x = base
beta = float(np.cov(y, x)[0, 1] / np.var(x, ddof=1))
alpha = float(y.mean() - beta * x.mean())
resid = y - (alpha + beta * x)
n = len(y)
se_a = float(resid.std(ddof=2) / np.sqrt(n))
t_a = alpha / se_a
print("\nTiming regression  oos = a + b*static:  beta=%.3f  alpha(daily var-units)=%.3e  t(alpha)=%.2f"
      % (beta, alpha, t_a))

# ---- picks summary ----
from collections import Counter
cnt = Counter(picks)
print("\nWalk-forward picks (%d blocks, %d combos in grid):" % (len(picks), n_combos))
for g, c2 in sorted(cnt.items()):
    print("  b1=%.1f b2=%.1f : %d blocks" % (g[0], g[1], c2))

# ---- programmatic look-ahead audit: signal at t unchanged when future data removed ----
b1a, b2a = GRID[4]
s_full = make_signal(b1a, b2a)
cut = len(ret) // 2
df_c, ret_c, vxn_c = df.iloc[:cut], ret.iloc[:cut], vxn.iloc[:cut]
rng_c = np.log(df_c["High"] / df_c["Low"]) * 100.0
be_c = vxn_c / SQ
s_c = pd.Series(0.0, index=ret_c.index)
okc = rng_c.notna() & be_c.notna()
s_c[okc & (rng_c < b1a * be_c)] = 1.0
s_c[okc & (rng_c > b2a * be_c)] = -1.0
same = bool((s_c == s_full.iloc[:cut]).all())
print("\nLook-ahead audit: truncated-sample signal identical on overlap ->", same)
print("OOS span: %s .. %s  (%d days)" % (oos.index[0].date(), oos.index[-1].date(), len(oos)))
