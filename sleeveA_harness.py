
import numpy as np, pandas as pd, os
DATA = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

def load():
    df = pd.read_csv(os.path.join(DATA,"NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
    fomc = pd.read_csv(os.path.join(DATA,"FOMC_dates.csv"), parse_dates=["fomc_date"])["fomc_date"].sort_values()
    kept,last=[],None
    for d in fomc:
        if last is None or (d-last).days>=20: kept.append(d); last=d
    return df, pd.DatetimeIndex(kept)

def _vol(df, kind, win):
    if kind=="parkinson":
        return np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(win).mean()/(4*np.log(2)))*SQ
    return df["Close"].pct_change().rolling(win).std()*SQ

def _monthly(desired):
    out,cur,pm=[],0.0,None
    for d,v in desired.items():
        if pm is None or d.month!=pm.month: cur=float(v)
        out.append(cur); pm=d
    return pd.Series(out, index=desired.index)

def sleeve_a(df, fomc, vol_kind="parkinson", vol_win=21, trend_win=200, target=0.15,
             fomc_boost=0.5, cost=0.0005, max_lev=3.0, use_trend=True, use_voltarget=True, use_fomc=True,
             rebal="M", exec_lag=1):
    px=df["Close"]; ret=px.pct_change()
    vol=_vol(df,vol_kind,vol_win)
    sig=(px>px.rolling(trend_win).mean()).astype(float) if use_trend else pd.Series(1.0,index=px.index)
    lev=(target/vol).clip(upper=max_lev) if use_voltarget else pd.Series(1.0,index=px.index)
    desired=(lev*sig).shift(exec_lag).fillna(0.0)
    if rebal=="M": base=_monthly(desired)
    elif rebal=="D": base=desired
    elif rebal=="W":
        base=desired.copy(); cur=0.0; pw=None
        vals=[]
        for d,v in desired.items():
            wk=d.isocalendar()[1]
            if pw is None or wk!=pw: cur=float(v)
            vals.append(cur); pw=wk
        base=pd.Series(vals,index=desired.index)
    else: base=desired
    if use_fomc:
        win=pd.Series(False,index=ret.index)
        for p in ret.index.get_indexer(fomc, method="bfill"):
            if 1<=p<len(ret.index): win.iloc[p-1]=True; win.iloc[p]=True
        pos=base.where(~win, np.minimum(base+fomc_boost,4.0))
    else:
        pos=base
    r=(pos*ret - cost*pos.diff().abs().fillna(0.0)).dropna()
    return r, pos

def voltarget_nq(df, target=0.15, vol_kind="parkinson", vol_win=21, max_lev=3.0, cost=0.0005):
    # vol-controlled long NQ, NO trend, NO fomc -> the 'is it just beta?' benchmark
    return sleeve_a(df, pd.DatetimeIndex([]), vol_kind=vol_kind, vol_win=vol_win, target=target,
                    cost=cost, max_lev=max_lev, use_trend=False, use_voltarget=True, use_fomc=False)[0]

def metrics(r):
    r=pd.Series(r).dropna()
    if len(r)<30 or r.std()==0: return dict(sharpe=float('nan'),cagr=float('nan'),maxdd=float('nan'),calmar=float('nan'),worst1y=float('nan'),vol=float('nan'),n=int(len(r)))
    eq=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.25
    cagr=float(eq.iloc[-1]**(1/yrs)-1); mdd=float((eq/eq.cummax()-1).min())
    return dict(sharpe=float(r.mean()/r.std()*SQ), cagr=cagr, maxdd=mdd,
                calmar=float(cagr/abs(mdd)) if mdd<0 else float('nan'),
                worst1y=float(eq.pct_change(252).min()), vol=float(r.std()*SQ), n=int(len(r)))
