# -*- coding: utf-8 -*-
"""Shared builder for THE BOOK = the selection-free floor (invvol x coverage, NO return info),
under FULL L4 frictions. Single source of truth reused by ex 47-50 so every analysis is consistent.
Respects H.TRAIN/H.TEST/H.K at call time (monkeypatch those for window / k robustness)."""
import numpy as np, pandas as pd
from math import erf
import bookopt_harness as H

SQ = H.SQ; WIN = 252
Nrm = lambda x: 0.5*(1.0+np.vectorize(erf)(x/np.sqrt(2.0)))

def invvol(p, win=WIN):                       # RISK only, NO return sign -> selection-free
    return (1.0/p.rolling(win).std()).replace([np.inf,-np.inf], np.nan).shift(1)

def coverage_gate(name, mult=1.0, margin_bp=0.0, win=WIN):
    H._load(); df, ret, vi, sp0 = H._DATA[name]; idx = ret.index
    prem = pd.Series(2*(2*Nrm(0.5*(H.K*vi.values/100)*np.sqrt(H.DT))-1), index=idx)
    spread = (pd.Series(sp0,index=idx)*(1+(vi/vi.rolling(63).median().shift(1)-1).clip(lower=0)).fillna(1.)).fillna(sp0)
    if name == "EEM": spread = spread + 0.005
    spread = spread*mult
    epsoff = pd.Series(0.00125*np.where(np.arange(len(idx))%2,1,-1), index=idx)
    net = prem.shift(1)-(ret-epsoff).abs()-spread.shift(1)*prem.shift(1)
    edge = net.rolling(win).mean().shift(1)
    cov = (edge >= margin_bp*1e-4).astype(float).where(edge.notna(), 1.0)
    return cov.shift(1)

def build(fund_rf=0.0, mult=1.0, margin_frac=0.15):
    """Return (book_series, sleeves_dict, weights_dict) for the floor book under current H.TRAIN/TEST/K."""
    sleeves = {}
    for name,_,_,_ in H.PAIRS:
        a, _ = H.market(name, mult=mult, fund_rf=fund_rf, margin_frac=margin_frac)
        if a is not None: sleeves[name] = a
    W = {n: invvol(sleeves[n])*coverage_gate(n, mult).reindex(sleeves[n].index) for n in sleeves}
    return H.book_of(sleeves, W), sleeves, W

def stat_line(r):
    r = r.dropna(); e = (1+r).cumprod(); yrs = (r.index[-1]-r.index[0]).days/365.25
    dd = float((e/e.cummax()-1).min())
    return dict(sharpe=H.sharpe(r), maxdd=dd, skew=float(r.skew()), worst=float(r.min()),
                calmar=(e.iloc[-1]**(1/yrs)-1)/abs(dd) if dd else float("nan"),
                cagr=e.iloc[-1]**(1/yrs)-1, n=len(r))
