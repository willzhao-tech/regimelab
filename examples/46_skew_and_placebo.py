# -*- coding: utf-8 -*-
"""Example 46 - (1) four finalists on the SAME axes incl. skew & worst-day; (2) INDEPENDENT
reproduction of the 82nd-pct timing-shuffled placebo on the composed book.
Finalists: baseline equal-risk | riskweight-alone | selection-free floor (invvol x cov) | composed (riskwt x cov).
All under FULL L4 frictions (bookopt_harness). Placebo: circular-rotate each sleeve's composed weight
(preserves on-fraction + autocorrelation, destroys phase-alignment with returns) AND i.i.d. shuffle; N=1000."""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
from math import erf
import bookopt_harness as H

SQ = H.SQ; WIN = 252
Nrm = lambda x: 0.5*(1.0+np.vectorize(erf)(x/np.sqrt(2.0)))

sleeves = {}
for name,_,_,_ in H.PAIRS:
    a, _ = H.market(name)
    if a is not None: sleeves[name] = a

def riskweight(p, win=WIN, eps=0.05):
    sh = (p.rolling(win).mean()/p.rolling(win).std()*SQ).replace([np.inf,-np.inf], np.nan)
    return (sh.clip(lower=0.0)+eps).shift(1)

def invvol(p, win=WIN):                       # inverse trailing vol: RISK only, NO return sign
    return (1.0/p.rolling(win).std()).replace([np.inf,-np.inf], np.nan).shift(1)

def coverage_gate(name, margin_bp=0.0, win=WIN):
    H._load(); df, ret, vi, sp0 = H._DATA[name]; idx = ret.index
    prem = pd.Series(2*(2*Nrm(0.5*(H.K*vi.values/100)*np.sqrt(H.DT))-1), index=idx)
    spread = (pd.Series(sp0,index=idx)*(1+(vi/vi.rolling(63).median().shift(1)-1).clip(lower=0)).fillna(1.)).fillna(sp0)
    if name == "EEM": spread = spread + 0.005
    epsoff = pd.Series(0.00125*np.where(np.arange(len(idx))%2,1,-1), index=idx)
    net = prem.shift(1)-(ret-epsoff).abs()-spread.shift(1)*prem.shift(1)
    edge = net.rolling(win).mean().shift(1)
    cov = (edge >= margin_bp*1e-4).astype(float).where(edge.notna(), 1.0)
    return cov.shift(1)

COV = {n: coverage_gate(n) for n in sleeves}
W_risk  = {n: riskweight(sleeves[n]) for n in sleeves}
W_floor = {n: invvol(sleeves[n])*COV[n].reindex(sleeves[n].index) for n in sleeves}
W_comp  = {n: riskweight(sleeves[n])*COV[n].reindex(sleeves[n].index) for n in sleeves}

FIN = [("baseline equal-risk", None), ("riskweight-alone", W_risk),
       ("selection-free floor (invvol x cov)", W_floor), ("composed (riskwt x cov)", W_comp)]

def stats(r):
    r = r.dropna(); e = (1+r).cumprod(); yrs = (r.index[-1]-r.index[0]).days/365.25
    dd = float((e/e.cummax()-1).min()); cagr = e.iloc[-1]**(1/yrs)-1
    return H.sharpe(r), dd, float(r.skew()), float(r.min()), (cagr/abs(dd) if dd else float("nan")), len(r)

books = {n: H.book_of(sleeves, w) for n, w in FIN}
print("(1) FOUR FINALISTS - native range (matches your table)")
print(f"{'finalist':37}{'Sharpe':>7}{'maxDD':>7}{'skew':>7}{'worstD':>8}{'Calmar':>7}{'n':>6}")
for n, _ in FIN:
    s, dd, sk, wd, cal, k = stats(books[n])
    print(f"{n:37}{s:>7.2f}{dd*100:>6.0f}%{sk:>7.2f}{wd*100:>7.1f}%{cal:>7.2f}{k:>6}")

# common index so the skew/worst-day ranking isn't a different-window artifact
ci = books["composed (riskwt x cov)"].index
for n in books: ci = ci.intersection(books[n].index)
print(f"\n(1b) SAME COMMON INDEX  {ci.min().date()}..{ci.max().date()}  (apples-to-apples skew)")
print(f"{'finalist':37}{'Sharpe':>7}{'maxDD':>7}{'skew':>7}{'worstD':>8}{'Calmar':>7}")
for n, _ in FIN:
    s, dd, sk, wd, cal, _ = stats(books[n].reindex(ci))
    print(f"{n:37}{s:>7.2f}{dd*100:>6.0f}%{sk:>7.2f}{wd*100:>7.1f}%{cal:>7.2f}")

# (2) TIMING-SHUFFLED PLACEBO on the composed book ---------------------------------
real = H.sharpe(books["composed (riskwt x cov)"])
idxs = {n: W_comp[n].index for n in sleeves}
vals = {n: W_comp[n].values.copy() for n in sleeves}
N = 1000
for mode in ("rotate", "shuffle"):
    rng = np.random.default_rng(20260611)
    null = np.empty(N)
    for i in range(N):
        Wsh = {}
        for n in sleeves:
            v = vals[n]
            if mode == "rotate":
                k = int(rng.integers(1, len(v)-1)); vv = np.roll(v, k)        # preserves autocorr + on-fraction
            else:
                vv = rng.permutation(v)                                       # preserves marginal only
            Wsh[n] = pd.Series(vv, index=idxs[n])
        null[i] = H.sharpe(H.book_of(sleeves, Wsh))
    pct = float((null < real).mean()*100); z = float((real-null.mean())/null.std())
    print(f"\n(2) PLACEBO [{mode}]  real composed Sharpe {real:.3f}  vs null mean {null.mean():.3f} "
          f"(sd {null.std():.3f}, 5-95% [{np.percentile(null,5):.2f},{np.percentile(null,95):.2f}])")
    print(f"    -> real at {pct:.0f}th pct,  z = {z:+.2f}  "
          f"({'timing adds real signal' if pct>=95 else 'much of the lift is generic de-leverage, NOT precise timing'})")
