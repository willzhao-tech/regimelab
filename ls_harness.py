
import numpy as np, pandas as pd, os
DATA=r"C:\Users\ASUS\Desktop\claude doc\1"
def load_1h():
    df=pd.read_csv(os.path.join(DATA,"NQ_1h_6m.csv"),parse_dates=["Datetime"]).set_index("Datetime").sort_index()
    return df
def load_daily6m():
    df=pd.read_csv(os.path.join(DATA,"NQ_F_all_history.csv"),parse_dates=["Date"]).set_index("Date").sort_index()
    return df.loc[df.index>=df.index.max()-pd.Timedelta(days=190)]
def split(s, frac=0.5):
    n=int(len(s)*frac); return s.iloc[:n], s.iloc[n:]
def backtest(signal, ret, leverage=1.0, cost_bps=1.0, ppy=252):
    signal=signal.reindex(ret.index).clip(-1,1)
    pos=signal.shift(1).fillna(0.0)*leverage
    turn=pos.diff().abs().fillna(pos.abs())
    lr=pos*ret - (cost_bps/1e4)*turn
    e=1.0; eq=[]; ruined=False; rd=None
    for dt,x in lr.items():
        e=e*(1+x)
        if e<=0: e=0.0; ruined=True; rd=dt; eq.append(e); break
        eq.append(e)
    eq=pd.Series(eq,index=lr.index[:len(eq)]); r=lr.loc[eq.index]
    return dict(ret=float(eq.iloc[-1]-1), sharpe=float(r.mean()/r.std()*np.sqrt(ppy)) if r.std()>0 else float('nan'),
                maxdd=float((eq/eq.cummax()-1).min()), ruined=bool(ruined), ruin_dt=str(rd), n=int(len(r)))
def lev_table(signal, ret, cost_bps=1.0, ppy=252):
    return {L: backtest(signal,ret,L,cost_bps,ppy) for L in (5,10,15,20)}
