# -*- coding: utf-8 -*-
"""Shared harness for optimizing the multi-market vol book under FULL (L4) frictions.
Extracted from examples/44_book_stress.py. All optimization hooks are CAUSAL:
 - gate(ctx) -> 0/1 Series multiplying the signal (ctx: prem, spread, rich, trend, rng, be, vi, ret)
 - include(sleeve_pnl) handled by caller via causal trailing stats
Baseline (no hooks) must reproduce book Sharpe ~0.84 (ex 44 L4)."""
import os
import numpy as np
import pandas as pd
from math import erf

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0); DT = 1.0/252.0; TRAIN, TEST = 1260, 252; K = 0.82
PAIRS = [("SPX","SPX","VIX",.025),("NQ","NQ_F","VXN",.025),("EEM","EEM","VXEEM",.025),
         ("DAX","DAX","VDAX",.04),("SX5E","SX5E","VSTOXX",.04),("N225","N225","JNIV",.04),
         ("HSI","HSI","VHSI",.04),("NIFTY","NSEI","INDIAVIX",.04)]
GA = [(a,b,c) for a in (2.,4.,6.) for b in (0.,-2.) for c in (0.,1.,2.)]
GB = [(b1,b2) for b1 in (.8,1.,1.2) for b2 in (1.3,1.6,2.)]
Nrm = lambda x: 0.5*(1.0+np.vectorize(erf)(x/np.sqrt(2.0)))

_DATA = {}
def _load():
    if _DATA: return
    for name, uf, vf, sp in PAIRS:
        df = pd.read_csv(os.path.join(OUT,uf+"_all_history.csv"),parse_dates=["Date"]).set_index("Date").sort_index()
        vi = pd.read_csv(os.path.join(OUT,vf+"_all_history.csv"),parse_dates=["Date"]).set_index("Date")["Close"].dropna()
        ret = df["Close"].pct_change(fill_method=None)
        idx = ret.index.intersection(vi.index)
        _DATA[name] = (df.loc[idx], ret.loc[idx], vi.loc[idx], sp)

def market(name, gate=None, mult=1.0, fund_rf=0.0, margin_frac=0.15, return_pos=False,
           emit_partial=False):
    """Gated blend sleeve + static, FULL L4 frictions. gate(ctx)->0/1 Series (causal!) ANDed onto signals.
    fund_rf: annual financing rate on posted margin (margin_frac of notional) charged per in-market day.
    Default fund_rf=0 -> no change to pre-existing results; set >0 for the P0.2 margin-funding sensitivity."""
    _load()
    df, ret, vi, sp0 = _DATA[name]
    idx = ret.index
    if len(idx) < TRAIN+TEST+50: return None, None
    prem = pd.Series(2*(2*Nrm(0.5*(K*vi.values/100)*np.sqrt(DT))-1), index=idx)
    pk = lambda w: (np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(w).mean()/(4*np.log(2)))*SQ*100).shift(1)
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
        fund = (fund_rf/252.0)*margin_frac*(pos.abs() > 0).astype(float)     # margin-financing drag, in-market days
        return (pos*payoff - c - fund).dropna()
    posof = lambda s: s.clip(-1,1).shift(1).fillna(0.)
    def wf(fn, grid):
        sigs = [fn(p) for p in grid]; ser = [pnl(s) for s in sigs]; poss = [posof(s) for s in sigs]
        bi = ser[0].index; pp = []; qq = []; st = TRAIN
        while st+TEST <= len(bi):
            tr = bi[st-TRAIN:st]; te = bi[st:st+TEST]
            def shp(p):
                x = p.reindex(tr).dropna()
                return x.mean()/x.std()*SQ if len(x)>60 and x.std()>0 else -9.
            kk = max(range(len(grid)), key=lambda i: shp(ser[i]))
            pp.append(ser[kk].reindex(te)); qq.append(poss[kk].reindex(te)); st += TEST
        if emit_partial and pp and st < len(bi):
            # trailing PARTIAL test window for live tracking — params from the last
            # COMPLETED train window, so it is exactly as causal as the full blocks.
            # Backtests keep the default (complete windows only) so published stats
            # are unchanged; the paper tracker needs this to track the present.
            tr = bi[st-TRAIN:st]; te = bi[st:]
            def shp2(p):
                x = p.reindex(tr).dropna()
                return x.mean()/x.std()*SQ if len(x)>60 and x.std()>0 else -9.
            kk = max(range(len(grid)), key=lambda i: shp2(ser[i]))
            pp.append(ser[kk].reindex(te)); qq.append(poss[kk].reindex(te))
        if not pp: return None, None
        return pd.concat(pp).dropna(), pd.concat(qq)
    pA, qA = wf(sA,GA); pB, qB = wf(sB,GB)
    if pA is None or pB is None:
        return (None, None, None) if return_pos else (None, None)
    cm = pA.index.intersection(pB.index)
    blend = 0.5*pA.loc[cm]+0.5*pB.loc[cm]
    static = pnl(pd.Series(1.,index=idx)).reindex(cm).dropna()
    if return_pos:
        return blend, static, dict(pA=pA.loc[cm], pB=pB.loc[cm], posA=qA.reindex(cm), posB=qB.reindex(cm))
    return blend, static

def book_of(d, weights=None):
    """Equal-risk book; optional CAUSAL weights dict of Series (caller must shift!)."""
    sc = {k: (p/(p.rolling(63).std().shift(1))) for k, p in d.items()}
    P = pd.DataFrame(sc)
    if weights is not None:
        W = pd.DataFrame(weights).reindex(P.index)
        P = P*W
        bk = (P.sum(axis=1, skipna=True)/W.where(P.notna()).sum(axis=1)).dropna()
    else:
        bk = P.mean(axis=1, skipna=True).dropna()
    lev = (0.10/(bk.rolling(63).std().shift(1)*SQ)).clip(upper=4.0)
    return (lev*bk).dropna()

def sharpe(r):
    r = pd.Series(r).dropna()
    return float(r.mean()/r.std()*SQ) if len(r) > 100 and r.std() > 0 else float("nan")

def maxdd(r):
    e = (1+pd.Series(r).dropna()).cumprod()
    return float((e/e.cummax()-1).min())

def split_halves(r):
    r = pd.Series(r).dropna(); h = r.index[len(r)//2]
    return sharpe(r[:h]), sharpe(r[h:])
