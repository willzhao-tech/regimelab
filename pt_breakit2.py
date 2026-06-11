# -*- coding: utf-8 -*-
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import sleeveA_harness as H
df, fomc = H.load(); df=df[~df.index.duplicated(keep="last")].sort_index()

# Full-sample: does the trend brake add or subtract net? (no fomc to isolate trend)
r_trend,_ = H.sleeve_a(df, fomc, use_trend=True,  use_fomc=False)
r_notr ,_ = H.sleeve_a(df, fomc, use_trend=False, use_fomc=False)
mt=H.metrics(r_trend); mn=H.metrics(r_notr)
print("FULL SAMPLE trend-brake effect (no fomc):")
print(f"  with trend : sharpe={mt['sharpe']:.3f} cagr={mt['cagr']*100:.2f}% mdd={mt['maxdd']*100:.2f}%")
print(f"  no   trend : sharpe={mn['sharpe']:.3f} cagr={mn['cagr']*100:.2f}% mdd={mn['maxdd']*100:.2f}%")
print(f"  => trend cuts MDD by {(mn['maxdd']-mt['maxdd'])*100:+.1f}pp but cuts Sharpe by {(mn['sharpe']-mt['sharpe']):.3f}")

# Year-by-year: trend-brake P&L edge each calendar year (when does it help vs hurt?)
print("\nYear-by-year trend-brake edge (with_trend - no_trend), total-return pp:")
def cum(s): return (1+s).prod()-1
yrs=range(2000,2026)
hurts=0; helps=0; tot=0.0
for y in yrs:
    sl=slice(f"{y}-01-01", f"{y}-12-31")
    a=cum(r_trend.loc[sl]); b=cum(r_notr.loc[sl]); e=(a-b)*100; tot+=e
    tag = "HELP" if e>0.5 else ("HURT" if e<-0.5 else "  ~ ")
    if e>0.5: helps+=1
    if e<-0.5: hurts+=1
    print(f"   {y}: with={a*100:7.2f}%  no={b*100:7.2f}%  edge={e:+7.2f}pp  {tag}")
print(f"  SUM of yearly edges = {tot:+.1f}pp over 26y; helped {helps}y, hurt {hurts}y")

# How many DISTINCT crash events did the trend brake actually protect? define crash = vol-only year < -15%
print("\nDid trend protect in the genuinely bad years (no-trend year < -10%)?")
for y in yrs:
    sl=slice(f"{y}-01-01", f"{y}-12-31"); b=cum(r_notr.loc[sl])
    if b<-0.10:
        a=cum(r_trend.loc[sl])
        print(f"   {y}: no-trend={b*100:.1f}%  with-trend={a*100:.1f}%  protection={(a-b)*100:+.1f}pp")

# Whipsaw count: how often does the 200d signal flip within a 21-day window (round-trips)?
px=df["Close"]; sig=(px>px.rolling(200).mean()).astype(float)
flips=sig.diff().abs().fillna(0)
print(f"\nTotal 200d trend flips 1999-2026: {int(flips.sum())}")
# round-trips: a flip followed by reverse within 21 trading days
ft=flips[flips>0].index
rt=0
for i in range(len(ft)-1):
    if (ft[i+1]-ft[i]).days<=31: rt+=1
print(f"  flips reversed within ~1 month (whipsaws): {rt}")
