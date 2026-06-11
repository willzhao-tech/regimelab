
import numpy as np, pandas as pd, os
DATA = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

def load():
    df = pd.read_csv(os.path.join(DATA,"NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
    vxn = pd.read_csv(os.path.join(DATA,"VXN_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    ret = df["Close"].pct_change()
    idx = ret.index.intersection(vxn.index)
    return df.loc[idx].copy(), ret.loc[idx], vxn.loc[idx]

def hedged_pnl(ret, vxn, cap=0.05, smile=1.5, cost_win=252, exposure=None):
    """Causal hedged short-variance daily P&L per unit notional. cap=daily-move wing strike;
    smile=markup on trailing fair wing cost (>=1, realistic>=1.5). exposure: optional CAUSAL
    multiplier series (you must .shift(1) it yourself before passing)."""
    iv = (vxn.shift(1)/100.0)**2/252.0
    rvar = ret**2
    capv = cap**2
    tail = np.maximum(rvar - capv, 0.0)
    wing = smile * tail.rolling(cost_win, min_periods=60).mean().shift(1)   # trailing, causal
    pnl = iv - np.minimum(rvar, capv) - wing
    if exposure is not None:
        pnl = exposure.reindex(pnl.index).fillna(0.0) * pnl
    return pnl.dropna()

def causal_scale(pnl, target=0.10, win=63, cap_lev=8.0):
    s = pnl.rolling(win).std().shift(1)*SQ
    lev = (target/s).replace([np.inf,-np.inf], np.nan).clip(upper=cap_lev).fillna(0.0)
    return (lev*pnl).dropna()

def metrics(r):
    r = pd.Series(r).dropna()
    if len(r)<30 or r.std()==0: return dict(sharpe=float('nan'),cagr=float('nan'),maxdd=float('nan'),skew=float('nan'),worst=float('nan'),n=int(len(r)))
    eq=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.25
    return dict(sharpe=float(r.mean()/r.std()*SQ), cagr=float(eq.iloc[-1]**(1/yrs)-1),
                maxdd=float((eq/eq.cummax()-1).min()), skew=float(r.skew()), worst=float(r.min()), n=int(len(r)))

def walk_forward(build_fn, grid, train=1260, test=252):
    """build_fn(params)->causal daily pnl Series over full history. grid=list of param dicts.
    Per test block pick the grid params with best trailing-train Sharpe; apply to next test block; roll.
    Returns the concatenated OUT-OF-SAMPLE pnl series (and the picks list via .attrs)."""
    series = [build_fn(p) for p in grid]
    idx = series[0].index
    parts=[]; picks=[]; start=train
    while start+test<=len(idx):
        tr=idx[start-train:start]; te=idx[start:start+test]
        def sh(s):
            x=s.reindex(tr); return (x.mean()/x.std()*SQ) if x.std()>0 else -9.0
        bi=max(range(len(grid)), key=lambda i: sh(series[i]))
        picks.append(grid[bi]); parts.append(series[bi].reindex(te))
        start+=test
    out = pd.concat(parts).dropna() if parts else pd.Series(dtype=float)
    out.attrs["picks"]=picks
    return out
