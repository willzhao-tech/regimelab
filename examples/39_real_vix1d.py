"""
Example 39 — THE REAL-DATA TEST: 1-DTE straddle on SPX priced with actual VIX1D.

VIX1D = CBOE's 1-Day Volatility Index (real 0-1DTE SPX implied vol, from listed options).
This removes the k-assumption entirely: premium is priced off the REAL short-dated implied.
Signals: the pre-registered dominant params from the NQ/SPX walk-forwards (A: rich>=2 &
trend<=0 short / rich<=-2 & trend>=0 long; B: range<1.0*be short / range>2.0*be long) —
FIXED, no search (784 days is far too short to re-optimize honestly).

Outputs: (1) the empirical k-path (VIX1D/VIX) incl. its state-dependence — the number the
whole prior analysis hinged on; (2) static vs gated straddle P&L on real pricing.
"""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
from math import erf

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0); DT = 1.0/252.0

spx = pd.read_csv(os.path.join(OUT, "SPX_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
vix = pd.read_csv(os.path.join(OUT, "VIX_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
v1d = pd.read_csv(os.path.join(OUT, "VIX1D_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
v9d = pd.read_csv(os.path.join(OUT, "VIX9D_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
ret = spx["Close"].pct_change()

# ---------- 1) the empirical k-path ----------
idx9 = v9d.index.intersection(vix.index)
k9 = (v9d/vix.reindex(v9d.index)).dropna()
idx1 = v1d.index.intersection(vix.index)
k1 = (v1d/vix.reindex(v1d.index)).dropna()
park10 = (np.sqrt((np.log(spx["High"]/spx["Low"])**2).rolling(10).mean()/(4*np.log(2)))*SQ*100).shift(1)
park42 = (np.sqrt((np.log(spx["High"]/spx["Low"])**2).rolling(42).mean()/(4*np.log(2)))*SQ*100).shift(1)
trend = (park10 - park42).reindex(k1.index)
print("EMPIRICAL k-PATH (real short-dated IV vs 30-day index)")
print(f"  k1 = VIX1D/VIX  ({k1.index.min().date()}..{k1.index.max().date()}, n={len(k1)}):")
print(f"    mean {k1.mean():.3f}  median {k1.median():.3f}  p5 {k1.quantile(.05):.3f}  p95 {k1.quantile(.95):.3f}  max {k1.max():.2f}")
print(f"    state-dependence: mean k1 when vol RISING (trend>0) = {k1[trend>0].mean():.3f}  "
      f"when FALLING = {k1[trend<=0].mean():.3f}")
print(f"  k9 = VIX9D/VIX  (2011+, n={len(k9)}): mean {k9.mean():.3f}  median {k9.median():.3f}  "
      f"p95 {k9.quantile(.95):.3f}  max {k9.max():.2f}")

# ---------- 2) real-priced 1-DTE straddle backtest on SPX (2023+) ----------
idx = ret.index.intersection(v1d.index).intersection(vix.index)
r, v1, vx = ret.reindex(idx), v1d.reindex(idx), vix.reindex(idx)
dfx = spx.reindex(idx)
N = lambda x: 0.5*(1.0 + np.vectorize(erf)(x/np.sqrt(2.0)))
prem = pd.Series(2.0*(2.0*N(0.5*(v1.values/100.0)*np.sqrt(DT)) - 1.0), index=idx)   # REAL pricing

park21s = (np.sqrt((np.log(dfx["High"]/dfx["Low"])**2).rolling(21).mean()/(4*np.log(2)))*SQ*100).shift(1)
park10s = (np.sqrt((np.log(dfx["High"]/dfx["Low"])**2).rolling(10).mean()/(4*np.log(2)))*SQ*100).shift(1)
park42s = (np.sqrt((np.log(dfx["High"]/dfx["Low"])**2).rolling(42).mean()/(4*np.log(2)))*SQ*100).shift(1)
richs = vx - park21s; trends = park10s - park42s
rngs = np.log(dfx["High"]/dfx["Low"])*100.0; bes = vx/SQ

sA = pd.Series(0.0, index=idx)
sA[((richs >= 2) & (trends <= 0)).fillna(False)] = 1.0
sA[((richs <= -2) & (trends >= 0)).fillna(False)] = -1.0
sB = pd.Series(0.0, index=idx)
ok = rngs.notna() & bes.notna()
sB[ok & (rngs < 1.0*bes)] = 1.0
sB[ok & (rngs > 2.0*bes)] = -1.0
blend_sig = 0.5*sA + 0.5*sB

def pnl_of(s, sp):
    pos = s.clip(-1, 1).shift(1).fillna(0.0)
    return (pos*(prem.shift(1) - r.abs()) - sp*prem.shift(1)*pos.abs()).dropna()

def m(p):
    p = p.dropna()
    sh = p.mean()/p.std()*SQ; t = p.mean()/(p.std()/np.sqrt(len(p)))
    return sh, t, p.skew()

print(f"\nREAL-PRICED 1-DTE STRADDLE, SPX, {idx.min().date()}..{idx.max().date()} (n={len(idx)})")
print(f"  E[premium] {prem.mean()*1e4:.0f} bps/day  vs  E|ret| {r.abs().mean()*1e4:.0f} bps/day  "
      f"(realized k-implied carry, no assumption)")
print(f"  {'spread':>8} | {'static short':>20} | {'gated blend (FIXED pre-reg params)':>36}")
for sp in (0.01, 0.025, 0.05):
    st = pnl_of(pd.Series(1.0, index=idx), sp)
    bl = pnl_of(blend_sig, sp)
    shs, ts, sks = m(st); shb, tb, skb = m(bl)
    yx = pd.concat([bl, st], axis=1).dropna().values
    y, x = yx[:, 0], yx[:, 1]; X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X@b
    at = b[0]/np.sqrt((res@res/(len(y)-2))*np.linalg.inv(X.T@X)[0, 0])
    print(f"  {sp*100:>7.1f}% | Sh {shs:>5.2f} (t {ts:>+4.1f}) sk {sks:>4.1f} | "
          f"Sh {shb:>5.2f} (t {tb:>+4.1f}) sk {skb:>4.1f}  alpha-t {at:>+4.1f}  beta {b[1]:.2f}")
pos = blend_sig.shift(1)
print(f"  position mix: short {100*(pos>0).mean():.0f}%  long {100*(pos<0).mean():.0f}%  flat/half {100*(pos==0).mean():.0f}%")
print("\nNOTE: 784 days only; params FIXED from prior walk-forwards (no search -> no deflation needed);")
print("includes Aug-2024 vol spike and 2025-04 tariff crash. This is the genuine out-of-sample,")
print("real-implied-vol validation of the gamma-instrument strategy.")
