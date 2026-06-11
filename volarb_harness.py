
import numpy as np, pandas as pd, os
DATA=r"C:\Users\ASUS\Desktop\claude doc\1"
SQ=np.sqrt(252.0)
def load():
    df=pd.read_csv(os.path.join(DATA,"NQ_F_all_history.csv"),parse_dates=["Date"]).set_index("Date").sort_index()
    vxn=pd.read_csv(os.path.join(DATA,"VXN_all_history.csv"),parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    ret=df["Close"].pct_change()
    idx=ret.index.intersection(vxn.index)
    return df.loc[idx], ret.loc[idx], vxn.loc[idx]
def iv_rvar(ret,vxn):
    iv=(vxn.shift(1)/100.0)**2/252.0    # strike set at prior close, accrues day t
    rvar=ret**2
    return iv,rvar
def fcast_vol(df,ret,kind="park21"):
    """Trailing annualized vol forecast in %, CAUSAL (shifted: value at t uses data through t-1)."""
    if kind.startswith("park"):
        w=int(kind[4:]); v=np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(w).mean()/(4*np.log(2)))*SQ*100
    elif kind.startswith("ewma"):
        lam=float(kind[4:])/100.0; v=np.sqrt(ret.pow(2).ewm(alpha=1-lam).mean()*252)*100
    else:
        w=int(kind[2:]); v=ret.rolling(w).std()*SQ*100
    return v.shift(1)
def backtest(s, ret, vxn, cost_volpt=0.5):
    """s in [-1,1] computed with data through close t; harness shifts it. UNCAPPED variance P&L."""
    iv,rvar=iv_rvar(ret,vxn)
    pos=s.reindex(ret.index).clip(-1,1).shift(1).fillna(0.0)
    cost=(2*vxn.shift(1)*cost_volpt/1e4/252).fillna(0.0)*pos.diff().abs().fillna(0.0)
    return (pos*(iv-rvar)-cost).dropna()
def metrics(pnl):
    p=pd.Series(pnl).dropna()
    if len(p)<60 or p.std()==0: return dict(sharpe=float('nan'),tstat=float('nan'),skew=float('nan'),worst_ratio=float('nan'),maxdd_units=float('nan'),n=int(len(p)))
    sh=float(p.mean()/p.std()*SQ); t=float(p.mean()/(p.std()/np.sqrt(len(p))))
    cum=p.cumsum(); mdd=float((cum-cum.cummax()).min())
    return dict(sharpe=sh,tstat=t,skew=float(p.skew()),
                worst_ratio=float(abs(p.min())/abs(p.mean())) if p.mean()!=0 else float('inf'),
                maxdd_units=float(mdd/abs(p.mean()*252)) if p.mean()!=0 else float('nan'), n=int(len(p)))
def walk_forward(build_fn, grid, train=1260, test=252):
    series=[build_fn(g) for g in grid]
    idx=series[0].index; parts=[];picks=[];start=train
    while start+test<=len(idx):
        tr=idx[start-train:start]; te=idx[start:start+test]
        def sh(s):
            x=s.reindex(tr).dropna(); return (x.mean()/x.std()*SQ) if len(x)>60 and x.std()>0 else -9.0
        bi=max(range(len(grid)),key=lambda i: sh(series[i]))
        picks.append(grid[bi]); parts.append(series[bi].reindex(te))
        start+=test
    out=pd.concat(parts).dropna() if parts else pd.Series(dtype=float)
    out.attrs["picks"]=picks
    return out
