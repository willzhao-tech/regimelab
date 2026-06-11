# -*- coding: utf-8 -*-
"""
APPROACH: vix1d_scaling  --  STATE-DEPENDENT k VIA TERM STRUCTURE (causal).

The harness prices the short-dated straddle premium with a FIXED k=0.82:
    prem = 2*(2*N(0.5*(k*VolIdx/100)*sqrt(dt)) - 1)
But real short-dated IV richens when vol rises. Measured on REAL CBOE data
(ex39): k1 = VIX1D/VIX has mean 0.820, but 0.866 when vol rising / 0.790
falling. This script makes k STATE-DEPENDENT using a CAUSAL proxy = each
market's OWN vol-index level vs its trailing-21d median (shifted), so the
premium we collect/pay reflects the term-structure regime.

CAUSALITY / NON-POST-HOC ARGUMENT
  * The proxy slope(t) = VolIdx(t)/median(VolIdx[t-21..t-1]) - 1 uses ONLY
    data known at decision time (trailing median is .shift(1); the day's
    vol-index close is observable when you'd strike the straddle that day).
  * The k-mapping  k(t) = 0.82 + BETA*clip(slope, -.25, .5)  is calibrated
    ONCE against EXTERNAL ground truth (real SPX VIX1D/VIX, ex39), NOT
    against the book's PnL. Validation (_bo_kcalib.py): the SAME causal
    slope reproduces the measured pattern -- w=21 rising k=0.865 / falling
    0.777 vs measured 0.866 / 0.790; regression k1 = 0.807 + 0.426*slope,
    corr 0.444. So BETA~0.42 is data-derived, not fit to the book.
  * The IDENTICAL mapping is applied to ALL 8 markets -- no per-market
    parameter, no market selection, no weight chosen from full-sample book
    results. Nothing here is decidable only with hindsight.

This is primarily a ROBUSTNESS CHECK: is the 0.84 book Sharpe an artifact of
the convenient fixed-k pricing, or does it survive more honest, regime-aware
pricing (which makes long-vol legs pay MORE in rising-vol regimes -- exactly
the crisis days the long leg is supposed to earn)?
"""
import os, sys
import numpy as np
import pandas as pd
from math import erf
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import bookopt_harness as H

OUT = H.OUT; SQ = H.SQ; DT = H.DT; TRAIN, TEST = H.TRAIN, H.TEST
K0 = H.K                      # 0.82 fixed baseline anchor
Nrm = H.Nrm
GA, GB = H.GA, H.GB

# ---- CAUSAL state-dependent k --------------------------------------------
# slope = VolIdx(t) / trailing-21d-median(VolIdx) - 1   (trailing median shifted -> causal)
# k(t)  = K0 + BETA * clip(slope, -SLO_LO, SLO_HI)
# BETA derived from real VIX1D/VIX regression (~0.426); clip keeps k sane.
def k_series(vi, beta, w=21, lo=0.25, hi=0.50):
    trail = vi.rolling(w).median().shift(1)
    slope = (vi / trail - 1.0)
    k = K0 + beta * slope.clip(-lo, hi)
    return k.fillna(K0)

# ---- local copy of H.market with state-dependent premium ------------------
def market_sk(name, beta, gate=None, mult=1.0, w=21):
    H._load()
    df, ret, vi, sp0 = H._DATA[name]
    idx = ret.index
    if len(idx) < TRAIN + TEST + 50:
        return None, None
    k = k_series(vi, beta, w=w).reindex(idx).fillna(K0)        # <-- state-dependent, causal
    prem = pd.Series(2*(2*Nrm(0.5*(k.values*vi.values/100)*np.sqrt(DT))-1), index=idx)
    pk = lambda ww: (np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(ww).mean()/(4*np.log(2)))*SQ*100).shift(1)
    rich = vi - pk(21); trend = pk(10) - pk(42)
    rng = np.log(df["High"]/df["Low"])*100; be = vi/SQ
    spread = (pd.Series(sp0,index=idx)*(1+(vi/vi.rolling(63).median().shift(1)-1).clip(lower=0)).fillna(1.)).fillna(sp0)
    if name == "EEM": spread = spread + 0.005
    spread = spread*mult
    eps = pd.Series(0.00125*np.where(np.arange(len(idx))%2,1,-1), index=idx)
    ctx = dict(prem=prem, spread=spread, rich=rich, trend=trend, rng=rng, be=be, vi=vi, ret=ret)
    g = gate(ctx).reindex(idx).fillna(0.0) if gate is not None else pd.Series(1.0, index=idx)
    def sA(p):
        s = pd.Series(0., index=idx)
        s[((rich>=p[0])&(trend<=-p[2])).fillna(False)] = 1.
        s[((rich<=p[1])&(trend>=p[2])).fillna(False)] = -1.
        return s*g
    def sB(p):
        s = pd.Series(0., index=idx); ok = rng.notna()&be.notna()
        s[ok&(rng<p[0]*be)] = 1.; s[ok&(rng>p[1]*be)] = -1.
        return s*g
    def pnl(s):
        pos = s.clip(-1,1).shift(1).fillna(0.)
        payoff = prem.shift(1) - (ret-eps).abs()
        c = spread.shift(1)*prem.shift(1)*pos.abs()
        c = c + spread.shift(1)*prem.shift(1)*pos.abs().where(pos<0, 0.)     # long legs 2x
        c = c.where(~(pos.abs()>0), np.maximum(c, 0.00015))                  # 1.5bp floor
        return (pos*payoff - c).dropna()
    def wf(fn, grid):
        ser = [pnl(fn(p)) for p in grid]
        bi = ser[0].index; parts = []; st = TRAIN
        while st+TEST <= len(bi):
            tr = bi[st-TRAIN:st]; te = bi[st:st+TEST]
            def shp(p):
                x = p.reindex(tr).dropna()
                return x.mean()/x.std()*SQ if len(x)>60 and x.std()>0 else -9.
            kk = max(range(len(grid)), key=lambda i: shp(ser[i]))
            parts.append(ser[kk].reindex(te)); st += TEST
        return pd.concat(parts).dropna() if parts else None
    pA, pB = wf(sA,GA), wf(sB,GB)
    if pA is None or pB is None: return None, None
    cm = pA.index.intersection(pB.index)
    return 0.5*pA.loc[cm]+0.5*pB.loc[cm], pnl(pd.Series(1.,index=idx)).reindex(cm).dropna()

def build_book(beta, w=21):
    sl = {}
    for name,_,_,_ in H.PAIRS:
        g,_ = market_sk(name, beta, w=w)
        if g is not None: sl[name]=g
    return H.book_of(sl), sl

def report(tag, bk):
    h1,h2 = H.split_halves(bk)
    print("%-34s Sharpe %+.4f  maxDD %+.4f  halves %.3f / %.3f" % (
        tag, H.sharpe(bk), H.maxdd(bk), h1, h2))
    return H.sharpe(bk), H.maxdd(bk), h1, h2

if __name__ == "__main__":
    # 0) baseline reproduction (beta=0 -> identical fixed-k pricing)
    bk0,_ = build_book(0.0)
    s0,dd0,a0,b0 = report("BASELINE (beta=0, fixed k=0.82)", bk0)

    # 1) state-dependent k, BETA calibrated from real VIX1D regression (~0.426)
    print("\n-- state-dependent k via causal VIX-slope proxy (beta from real VIX1D fit) --")
    res = {}
    for beta in [0.20, 0.426, 0.60]:
        bk,_ = build_book(beta)
        res[beta] = report("beta=%.3f" % beta, bk)

    # 2) robustness to the trailing window of the proxy (no tuning -> show a few)
    print("\n-- robustness to proxy window (beta=0.426 fixed) --")
    for w in [10, 21, 63]:
        bk,_ = build_book(0.426, w=w)
        report("beta=0.426 w=%d" % w, bk)

    # 3) headline at the data-derived beta
    print("\n==== HEADLINE ====")
    s,dd,a,b = res[0.426]
    print("baseline           : Sharpe %.4f  maxDD %.4f  halves %.3f/%.3f" % (s0,dd0,a0,b0))
    print("state-dep k (0.426): Sharpe %.4f  maxDD %.4f  halves %.3f/%.3f" % (s,dd,a,b))
    print("delta Sharpe       : %+.4f" % (s-s0))
