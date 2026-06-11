"""Adverse-k stress test for the 1-DTE straddle result: short-dated IV richens vs VXN
exactly when vol is rising (when our LONG legs buy) and cheapens when vol is calm
(when our SHORT legs sell — less premium collected). Both directions ADVERSE:
    k_t = 0.80 when vol-trend > 0 (we'd buy expensive)... but wait, ADVERSE means:
    - when WE BUY (trend rising): k HIGH (pay up)        -> k=1.10
    - when WE SELL (trend falling/calm): k LOW (collect less) -> k=0.80
Also a placebo: signals applied to PURE NOISE k. If alpha-t collapses under adverse k,
the result was a term-structure artifact; if it holds, the timing is genuine."""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
from math import erf

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0); DT = 1.0/252.0; TRAIN, TEST = 1260, 252

df = pd.read_csv(os.path.join(OUT, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
vxn = pd.read_csv(os.path.join(OUT, "VXN_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
ret = df["Close"].pct_change()
idx = ret.index.intersection(vxn.index)
df, ret, vxn = df.loc[idx], ret.loc[idx], vxn.loc[idx]

N = lambda x: 0.5*(1.0 + np.vectorize(erf)(x/np.sqrt(2.0)))
park = lambda w: (np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(w).mean()/(4*np.log(2)))*SQ*100).shift(1)
fc21, fc10, fc42 = park(21), park(10), park(42)
rich = vxn - fc21; trend = fc10 - fc42
rng = np.log(df["High"]/df["Low"])*100.0; be = vxn/SQ

# ADVERSE state-dependent k: expensive when vol rising (our buys), cheap when falling (our sells)
k_t = pd.Series(0.80, index=idx)
k_t[(trend > 0).fillna(False)] = 1.10
sig_1d = k_t * vxn / 100.0
prem = pd.Series(2.0*(2.0*N(0.5*sig_1d.values*np.sqrt(DT)) - 1.0), index=idx)

def sig_A(g):
    r_hi, r_lo, d = g
    s = pd.Series(0.0, index=idx)
    s[((rich >= r_hi) & (trend <= -d)).fillna(False)] = 1.0
    s[((rich <= r_lo) & (trend >= d)).fillna(False)] = -1.0
    return s
def sig_B(g):
    b1, b2 = g
    s = pd.Series(0.0, index=idx); ok = rng.notna() & be.notna()
    s[ok & (rng < b1*be)] = 1.0; s[ok & (rng > b2*be)] = -1.0
    return s
GRID_A = [(a,b,c) for a in (2.,4.,6.) for b in (0.,-2.) for c in (0.,1.,2.)]
GRID_B = [(b1,b2) for b1 in (.8,1.,1.2) for b2 in (1.3,1.6,2.)]

def pnl_of(s, sp=0.025):
    pos = s.clip(-1,1).shift(1).fillna(0.0)
    return (pos*(prem.shift(1) - ret.abs()) - sp*prem.shift(1)*pos.abs()).dropna()

def wf(sig_fn, grid):
    series = [pnl_of(sig_fn(g)) for g in grid]
    bidx = series[0].index; parts=[]; start=TRAIN
    while start+TEST <= len(bidx):
        tr = bidx[start-TRAIN:start]; te = bidx[start:start+TEST]
        def sh(p):
            x=p.reindex(tr).dropna(); return x.mean()/x.std()*SQ if len(x)>60 and x.std()>0 else -9
        bi = max(range(len(grid)), key=lambda i: sh(series[i]))
        parts.append(series[bi].reindex(te)); start += TEST
    return pd.concat(parts).dropna()

stat = pnl_of(pd.Series(1.0, index=idx))
pA, pB = wf(sig_A, GRID_A), wf(sig_B, GRID_B)
common = pA.index.intersection(pB.index)
blend = 0.5*pA.loc[common] + 0.5*pB.loc[common]; statc = stat.reindex(common).dropna()

def m(p):
    p=p.dropna(); return p.mean()/p.std()*SQ, p.skew()
yx = pd.concat([blend, statc], axis=1).dropna().values
y,x = yx[:,0], yx[:,1]; X = np.column_stack([np.ones(len(x)),x])
b,*_ = np.linalg.lstsq(X,y,rcond=None); r = y-X@b
at = b[0]/np.sqrt((r@r/(len(y)-2))*np.linalg.inv(X.T@X)[0,0])
print("ADVERSE-k 1-DTE straddle (buys at k=1.10, sells at k=0.80, spread 2.5%):")
print(f"  static: Sharpe {m(statc)[0]:+.2f} skew {m(statc)[1]:+.1f}")
print(f"  A:      Sharpe {m(pA.loc[common])[0]:+.2f} skew {m(pA.loc[common])[1]:+.1f}")
print(f"  B:      Sharpe {m(pB.loc[common])[0]:+.2f} skew {m(pB.loc[common])[1]:+.1f}")
print(f"  blend:  Sharpe {m(blend)[0]:+.2f} skew {m(blend)[1]:+.1f}   alpha-t vs static {at:+.1f}  beta {b[1]:+.2f}")
