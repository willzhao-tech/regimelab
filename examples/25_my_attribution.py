"""Independent attribution of Sleeve A's Sharpe — my own check vs the workflow.
Question: is the 0.70 the trend edge, or just long-NQ beta wearing a vol-target?
"""
import os, numpy as np, pandas as pd
DATA = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

df = pd.read_csv(os.path.join(DATA,"NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
fomc = pd.read_csv(os.path.join(DATA,"FOMC_dates.csv"), parse_dates=["fomc_date"])["fomc_date"].sort_values()
kept,last=[],None
for d in fomc:
    if last is None or (d-last).days>=20: kept.append(d); last=d
fomc=pd.DatetimeIndex(kept)

px=df["Close"]; ret=px.pct_change()
park=np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(21).mean()/(4*np.log(2)))*SQ
trend=(px>px.rolling(200).mean()).astype(float)
lev=(0.15/park).clip(upper=3.0)

def monthly(desired):
    out,cur,pm=[],0.0,None
    for d,v in desired.items():
        if pm is None or d.month!=pm.month: cur=float(v)
        out.append(cur); pm=d
    return pd.Series(out,index=desired.index)

def fomc_pos(base):
    win=pd.Series(False,index=ret.index)
    for p in ret.index.get_indexer(fomc, method="bfill"):
        if 1<=p<len(ret.index): win.iloc[p-1]=True; win.iloc[p]=True
    return base.where(~win, np.minimum(base+0.5,4.0))

def run(pos): return (pos*ret - 0.0005*pos.diff().abs().fillna(0.0)).dropna()
def sh(r): r=r.dropna(); return r.mean()/r.std()*SQ
def mdd(r): e=(1+r.dropna()).cumprod(); return float((e/e.cummax()-1).min())

sleeveA = run(fomc_pos(monthly((lev*trend).shift(1).fillna(0))))
vt_only = run(monthly((lev*1.0).shift(1).fillna(0)))          # vol-target, NO trend
trend1x = run(monthly((1.0*trend).shift(1).fillna(0)))        # trend long/cash at 1x, NO vol-target
bh = ret.dropna()

print("component Sharpe / maxDD:")
for n,r in [("buy&hold NQ",bh),("trend-only 1x (long/cash)",trend1x),
            ("vol-target NQ (no trend)",vt_only),("Sleeve A (full)",sleeveA)]:
    print(f"  {n:<28} Sharpe {sh(r):+.2f}   maxDD {mdd(r)*100:>4.0f}%   vol {r.std()*SQ*100:>4.0f}%")

# regress Sleeve A on vol-target-NQ (the 'is it just the vol-controlled beta?' test)
common = sleeveA.index.intersection(vt_only.index)
y=sleeveA.loc[common].values; x=vt_only.loc[common].values
beta=np.cov(x,y)[0,1]/np.var(x); alpha=(y.mean()-beta*x.mean())
resid=y-(alpha+beta*x); r2=1-resid.var()/y.var()
talpha=alpha/(resid.std()/np.sqrt(len(y)))
print(f"\nOLS  Sleeve A ~ alpha + beta*(vol-target NQ):")
print(f"  beta {beta:.2f}   alpha {alpha*252*100:+.1f}%/yr (t {talpha:+.1f})   R^2 {r2:.2f}")

# also vs raw NQ
x2=bh.reindex(common).values
b2=np.cov(x2,y)[0,1]/np.var(x2); a2=y.mean()-b2*x2.mean(); res2=y-(a2+b2*x2)
ta2=a2/(res2.std()/np.sqrt(len(y)))
print(f"  vs raw NQ: beta {b2:.2f}  alpha {a2*252*100:+.1f}%/yr (t {ta2:+.1f})  R^2 {1-res2.var()/y.var():.2f}")
print("\nRead: if alpha vs vol-target-NQ is ~0 / insignificant, the trend+FOMC add NO Sharpe over")
print("just vol-controlling NQ — Sleeve A's number is the vol-controlled long-NQ beta, plus (maybe)")
print("a drawdown improvement that must be judged separately.")
