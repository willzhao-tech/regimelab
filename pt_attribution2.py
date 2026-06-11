# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import sleeveA_harness as H
import numpy as np, pandas as pd

SQ = np.sqrt(252.0)
df, fomc = H.load()
ret = df["Close"].pct_change()

rA, posA = H.sleeve_a(df, fomc)  # FULL defaults
common = rA.index
bh = ret.reindex(common).dropna()
idx = rA.index.intersection(bh.index)
y = rA.reindex(idx).values
x = bh.reindex(idx).values

def ols_nw(y, x, L=10):
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X@b
    n,k = X.shape
    XtX_inv = np.linalg.inv(X.T@X)
    # Newey-West HAC
    S = (X*u[:,None]).T @ (X*u[:,None])
    for l in range(1, L+1):
        w = 1 - l/(L+1)
        g = (X[l:]*u[l:,None]).T @ (X[:-l]*u[:-l,None])
        S += w*(g+g.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return b, se, b/se

b, se, t = ols_nw(y, x, L=10)
print("OLS Sleeve A ~ NQ  with Newey-West(L=10) SE:")
print(f"  alpha = {b[0]*1e4:.3f} bp/day   ann = {b[0]*252*100:.2f}%/yr   t_NW = {t[0]:.2f}")
print(f"  beta  = {b[1]:.4f}   t_NW = {t[1]:.1f}")
sig = "SIGNIFICANT" if abs(t[0])>1.96 else "NOT significant"
print(f"  alpha {sig} under HAC")

# --- Where does the alpha live? Split alpha into FOMC-window days vs the rest ---
# Rebuild fomc window mask exactly as harness does
win = pd.Series(False, index=ret.index)
for p in ret.index.get_indexer(fomc, method="bfill"):
    if 1 <= p < len(ret.index):
        win.iloc[p-1] = True; win.iloc[p] = True
win = win.reindex(idx).fillna(False).values

# regress with FOMC dummy interaction to see if alpha is concentrated in FOMC days
yj = y; xj = x
X = np.column_stack([np.ones_like(xj), xj, win.astype(float), win.astype(float)*xj])
bb, *_ = np.linalg.lstsq(X, yj, rcond=None)
print("\nAlpha localization (FOMC dummy):")
print(f"  base alpha (non-FOMC) = {bb[0]*252*100:.2f}%/yr")
print(f"  extra alpha ON FOMC days = {bb[2]*252*100:.2f}%/yr (per FOMC-window day, annualized)")
print(f"  base beta = {bb[1]:.3f},  extra beta on FOMC = {bb[3]:.3f}")
n_fomc = int(win.sum()); print(f"  # FOMC-window days = {n_fomc} of {len(win)} ({100*n_fomc/len(win):.1f}%)")

# --- Sub-period stability of Sleeve A Sharpe and the trend-brake effect ---
print("\nSub-period Sharpe (FULL vs no_trend):")
rNT,_ = H.sleeve_a(df, fomc, use_trend=False)
rNT = rNT.reindex(common)
for lo,hi in [("1999","2009"),("2010","2017"),("2018","2026")]:
    seg = rA.loc[lo:hi]; segNT = rNT.loc[lo:hi]
    print(f"  {lo}-{hi}: FULL={H.metrics(seg)['sharpe']:.3f}  no_trend={H.metrics(segNT)['sharpe']:.3f}  "
          f"(FULL maxDD={H.metrics(seg)['maxdd']*100:.0f}% no_trend maxDD={H.metrics(segNT)['maxdd']*100:.0f}%)")

# --- Does the trend brake at least pay off in the worst crisis windows? ---
print("\nCrisis-window comparison (the trend brake's job is tail protection):")
for lo,hi,label in [("2000-01","2002-12","dotcom"),("2007-10","2009-03","GFC"),
                    ("2020-02","2020-04","covid"),("2022-01","2022-12","2022 bear")]:
    seg=rA.loc[lo:hi]; segNT=rNT.loc[lo:hi]
    eqF=(1+seg).prod()-1; eqN=(1+segNT).prod()-1
    print(f"  {label:<9} {lo}..{hi}: FULL ret={eqF*100:+6.1f}%   no_trend ret={eqN*100:+6.1f}%")
