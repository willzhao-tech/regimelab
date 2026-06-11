"""Sleeve A falsification battery (run by me, not trusting the failed workflow).
Attacks: (1) param robustness, (2) sub-period + few-crisis, (3) cost sensitivity,
(4) 2020 fast-crash whipsaw, (5) rebal-timing luck."""
import os, numpy as np, pandas as pd
DATA = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

df = pd.read_csv(os.path.join(DATA,"NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
fomc = pd.read_csv(os.path.join(DATA,"FOMC_dates.csv"), parse_dates=["fomc_date"])["fomc_date"].sort_values()
kept,last=[],None
for d in fomc:
    if last is None or (d-last).days>=20: kept.append(d); last=d
FOMC=pd.DatetimeIndex(kept)
px=df["Close"]; ret=px.pct_change()

def vol_est(kind,win):
    if kind=="parkinson": return np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(win).mean()/(4*np.log(2)))*SQ
    return ret.rolling(win).std()*SQ

def monthly(desired, offset=None):
    idx=desired.index
    if offset is None:
        out,cur,pm=[],0.0,None
        for d,v in desired.items():
            if pm is None or d.month!=pm.month: cur=float(v)
            out.append(cur); pm=d
        return pd.Series(out,index=idx)
    dom=pd.Series(idx,index=idx).groupby([idx.year,idx.month]).cumcount()
    out,cur=[],0.0
    for d in idx:
        if dom[d]==offset: cur=float(desired[d])
        out.append(cur)
    return pd.Series(out,index=idx)

def build(vol_kind="parkinson",vol_win=21,trend_win=200,target=0.15,fomc_boost=0.5,cost=0.0005,
          use_trend=True,use_fomc=True,offset=None):
    vol=vol_est(vol_kind,vol_win)
    sig=(px>px.rolling(trend_win).mean()).astype(float) if use_trend else pd.Series(1.0,index=px.index)
    lev=(target/vol).clip(upper=3.0)
    base=monthly((lev*sig).shift(1).fillna(0.0), offset)
    if use_fomc:
        win=pd.Series(False,index=ret.index)
        for p in ret.index.get_indexer(FOMC,method="bfill"):
            if 1<=p<len(ret.index): win.iloc[p-1]=True; win.iloc[p]=True
        pos=base.where(~win, np.minimum(base+fomc_boost,4.0))
    else: pos=base
    return (pos*ret - cost*pos.diff().abs().fillna(0.0)).dropna(), pos

def sh(r): r=r.dropna(); return r.mean()/r.std()*SQ if r.std()>0 else np.nan
def mdd(r): e=(1+r.dropna()).cumprod(); return float((e/e.cummax()-1).min())
def cagr(r): r=r.dropna(); e=(1+r).cumprod(); return e.iloc[-1]**(252/len(r))-1

A,posA=build()
VT,_=build(use_trend=False,use_fomc=False)
print(f"baseline: Sleeve A Sharpe {sh(A):.2f} maxDD {mdd(A)*100:.0f}% | vol-target-NQ {sh(VT):.2f} {mdd(VT)*100:.0f}% | NQ {sh(ret):.2f} {mdd(ret)*100:.0f}%")

print("\n(1) PARAM ROBUSTNESS — Sleeve A Sharpe across 120 configs")
shs=[]
for vk in ("parkinson","close"):
    for vw in (10,21,42,63):
        for tw in (100,150,200,250,300):
            for tg in (0.10,0.15,0.20):
                r,_=build(vk,vw,tw,tg); shs.append(sh(r))
shs=np.array(shs); chosen=sh(A)
print(f"  n={len(shs)}  mean {shs.mean():.2f}  std {shs.std():.2f}  min {shs.min():.2f}  max {shs.max():.2f}")
print(f"  chosen config {chosen:.2f} sits at {(shs<chosen).mean()*100:.0f}th pct  |  % configs Sharpe>0.6: {(shs>0.6).mean()*100:.0f}%  >VT(0.72): {(shs>sh(VT)).mean()*100:.0f}%")

print("\n(2) SUB-PERIODS & FEW-CRISIS")
for lo,hi in [("2001","2008"),("2009","2016"),("2017","2026")]:
    a=A[lo:hi]; v=VT.reindex(A.index)[lo:hi]; n=ret.reindex(A.index)[lo:hi]
    print(f"  {lo}-{hi}: SleeveA Sh {sh(a):+.2f} DD {mdd(a)*100:>4.0f}% | volNQ {sh(v):+.2f} {mdd(v)*100:>4.0f}% | NQ {sh(n):+.2f} {mdd(n)*100:>4.0f}%")
crashes=[("2001-01","2002-12"),("2008-08","2009-03"),("2020-02","2020-04"),("2022-01","2022-12")]
mask=pd.Series(True,index=A.index)
for lo,hi in crashes: mask[lo:hi]=False
print(f"  EXCLUDING crashes: SleeveA Sh {sh(A[mask]):+.2f} DD {mdd(A[mask])*100:.0f}% | vol-target-NQ {sh(VT.reindex(A.index)[mask]):+.2f} {mdd(VT.reindex(A.index)[mask])*100:.0f}%")
print(f"  -> outside crises the trend brake's drawdown edge vanishes; does it still beat volNQ on Sharpe? {'yes' if sh(A[mask])>sh(VT.reindex(A.index)[mask]) else 'NO'}")

print("\n(3) COST SENSITIVITY (turnover & where edge dies)")
to=posA.diff().abs().sum()/((A.index[-1]-A.index[0]).days/365.25)
print(f"  Sleeve A turnover {to:.1f}x/yr (vs NQ buy&hold 0x)")
for c in (0.0002,0.0005,0.0010,0.0020,0.0030):
    r,_=build(cost=c); print(f"  cost {c*1e4:>2.0f}bps: Sleeve A Sharpe {sh(r):.2f}  CAGR {cagr(r)*100:.1f}%")

print("\n(4) 2020 FAST-CRASH WHIPSAW")
for lo,hi,lbl in [("2020-01","2020-12","full 2020"),("2020-03","2020-06","crash+rebound")]:
    print(f"  {lbl}: SleeveA {(((1+A[lo:hi]).prod())-1)*100:+.1f}% | volNQ {(((1+VT.reindex(A.index)[lo:hi]).prod())-1)*100:+.1f}% | NQ {(((1+ret.reindex(A.index)[lo:hi]).prod())-1)*100:+.1f}%")
trend_on=(px>px.rolling(200).mean())
print(f"  trend exited (px<200MA) {trend_on['2020-03':'2020-06'].eq(False).sum()} of {len(trend_on['2020-03':'2020-06'])} days in Mar-Jun 2020 (sold the dip, re-entered late)")

print("\n(5) REBAL-TIMING LUCK — Sharpe by which day-of-month you rebalance")
offs=[sh(build(offset=o)[0]) for o in range(0,19,3)]
print(f"  offsets 0,3,6,9,12,15,18: {[round(x,2) for x in offs]}")
print(f"  spread max-min = {max(offs)-min(offs):.2f}  ({'TIMING-LUCK SENSITIVE' if max(offs)-min(offs)>0.15 else 'stable'})")
