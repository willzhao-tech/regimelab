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

import volbook_config as CFG

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0); DT = CFG.DT; TRAIN, TEST = CFG.TRAIN, CFG.TEST; K = CFG.K
PAIRS = CFG.PAIRS
GA = CFG.GRID_A
GB = CFG.GRID_B
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

def market(name, gate=None, mult=1.0, fund_rf=0.0, margin_frac=CFG.MARGIN_FRAC, return_pos=False,
           emit_partial=False, families=None):
    """Gated blend sleeve + static, FULL L4 frictions. gate(ctx)->0/1 Series (causal!) ANDed onto signals.
    fund_rf: annual financing rate on posted margin (margin_frac of notional) charged per in-market day.
    Default fund_rf=0 -> no change to pre-existing results; set >0 for the P0.2 margin-funding sensitivity.
    families: NEW-STRATEGY hook (strategy_lab). List of (fn, grid) where fn(ctx, params) -> signal
    Series in [-1,+1] (+1 = short vol). ctx keys: prem, spread, rich, trend, rng, be, vi, ret, df.
    Signals computed on day t are applied on t+1 (the harness shifts); the established families
    use same-day vi/rng, so doing likewise is causal. None -> the default A/B blend (unchanged)."""
    _load()
    df, ret, vi, sp0 = _DATA[name]
    idx = ret.index
    if len(idx) < TRAIN+TEST+50: return None, None
    prem = pd.Series(2*(2*Nrm(0.5*(K*vi.values/100)*np.sqrt(DT))-1), index=idx)
    pk = lambda w: (np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(w).mean()/(4*np.log(2)))*SQ*100).shift(1)
    rich = vi - pk(21); trend = pk(10) - pk(42)
    rng = np.log(df["High"]/df["Low"])*100; be = vi/SQ
    spread = (pd.Series(sp0,index=idx)*(1+(vi/vi.rolling(63).median().shift(1)-1).clip(lower=0)).fillna(1.)).fillna(sp0)
    if name == "EEM": spread = spread + CFG.EEM_ASSIGN_SPREAD
    spread = spread*mult
    eps = pd.Series(CFG.STRIKE_OFFSET*np.where(np.arange(len(idx))%2,1,-1), index=idx)
    ctx = dict(prem=prem, spread=spread, rich=rich, trend=trend, rng=rng, be=be, vi=vi, ret=ret, df=df)
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
        c = c.where(~(pos.abs()>0), np.maximum(c, CFG.COST_FLOOR))           # 1.5bp floor
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
    if families is None:
        fam_list = [(sA, GA), (sB, GB)]
    else:
        fam_list = [(lambda p, f=f: f(ctx, p).reindex(idx).fillna(0.)*g, grid)
                    for f, grid in families]
    parts = [wf(fn, grid) for fn, grid in fam_list]
    if any(p[0] is None for p in parts):
        return (None, None, None) if return_pos else (None, None)
    cm = parts[0][0].index
    for p, _ in parts[1:]:
        cm = cm.intersection(p.index)
    blend = sum(p.loc[cm] for p, _ in parts) / len(parts)
    static = pnl(pd.Series(1.,index=idx)).reindex(cm).dropna()
    if return_pos:
        info = dict(parts=[p.loc[cm] for p, _ in parts], poss=[q.reindex(cm) for _, q in parts])
        if len(parts) == 2:                      # legacy keys (paper_trade, ex 48)
            info.update(pA=parts[0][0].loc[cm], pB=parts[1][0].loc[cm],
                        posA=parts[0][1].reindex(cm), posB=parts[1][1].reindex(cm))
        return blend, static, info
    return blend, static

def book_of(d, weights=None, prescale=True):
    """Equal-risk book; optional CAUSAL weights dict of Series (caller must shift!).
    Zero-variance trailing windows (possible for SPARSE custom signals; never observed for the
    production sleeves) scale to NaN, not inf, so a quiet sleeve drops out instead of poisoning.
    prescale=False skips the inner 63d unit-risk scaling (the caller's weights then carry ALL
    risk normalization — used by strategy_lab for sparse signals where short-window scaling
    explodes on re-entry after flat stretches). Default True = production behavior, unchanged."""
    def _scale(p):
        sd = p.rolling(63).std().shift(1)
        return p / sd.where(sd > 0)
    sc = {k: (_scale(p) if prescale else p) for k, p in d.items()}
    P = pd.DataFrame(sc)
    if weights is not None:
        W = pd.DataFrame(weights).reindex(P.index)
        P = P*W
        bk = (P.sum(axis=1, skipna=True)/W.where(P.notna()).sum(axis=1)).dropna()
    else:
        bk = P.mean(axis=1, skipna=True).dropna()
    lev = (CFG.TARGET_VOL/(bk.rolling(63).std().shift(1)*SQ)).clip(upper=CFG.LEV_CAP)
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
