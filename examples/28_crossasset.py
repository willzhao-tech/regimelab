"""Run the popular strategies (ex 27) across 7 assets — the cross-asset OOS test.
NQ = in-sample reference; NVDA/A50/US10Y/EUR/GOLD/OIL = out-of-sample. Same strict-causal
harness, net of 2bps. Question: does ANY popular strategy beat buy&hold across MOST assets?"""
import os, numpy as np, pandas as pd
DATA = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0); COST = 0.0002

ASSETS = {"NQ":"NQ_F_all_history.csv","NVDA":"NVDA_all_history.csv","A50":"A50_all_history.csv",
          "US10Y":"US10Y_all_history.csv","EUR":"EURUSD_all_history.csv","GOLD":"XAU_all_history.csv",
          "OIL":"WTI_all_history.csv"}

def rsi(s,n):
    d=s.diff(); up=d.clip(lower=0); dn=(-d).clip(lower=0)
    return 100-100/(1+up.ewm(alpha=1/n,adjust=False).mean()/dn.ewm(alpha=1/n,adjust=False).mean())

def state(entry,exit_):
    pos=np.zeros(len(entry)); s=0; e1=entry.values; e2=exit_.values
    for i in range(len(e1)):
        if s==0 and e1[i]: s=1
        elif s==1 and e2[i]: s=0
        pos[i]=s
    return pd.Series(pos,index=entry.index)

def strategies(df):
    o,h,l,c=df["Open"],df["High"],df["Low"],df["Close"]; ret=c.pct_change()
    sma=lambda n:c.rolling(n).mean(); r2=rsi(c,2)
    mid=sma(20); sd=c.rolling(20).std()
    ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean()
    macd=ema12-ema26; msig=macd.ewm(span=9,adjust=False).mean()
    idx=c.index; dom=pd.Series(idx,index=idx).groupby([idx.year,idx.month]).cumcount()
    domr=pd.Series(idx,index=idx).groupby([idx.year,idx.month]).cumcount(ascending=False)
    park=np.sqrt((np.log(h/l)**2).rolling(21).mean()/(4*np.log(2)))*SQ
    d3=(ret<0)&(ret.shift(1)<0)&(ret.shift(2)<0)
    return {
      "buy_hold":pd.Series(1.0,index=c.index),
      "sma200":(c>sma(200)).astype(float),
      "golden_50_200":(sma(50)>sma(200)).astype(float),
      "ma_20_50":(sma(20)>sma(50)).astype(float),
      "tsmom_12m":(c>c.shift(252)).astype(float),
      "macd":(macd>msig).astype(float),
      "rsi2<10":state((r2<10)&(c>sma(200)),c>sma(5)),
      "bollinger_mr":state(c<(mid-2*sd),c>mid),
      "donchian":state(c>=h.rolling(20).max().shift(1),c<=l.rolling(10).min().shift(1)),
      "buy_dip_3d":state(d3,ret>0),
      "turn_of_month":((dom<3)|(domr==0)).astype(float),
      "sell_in_may":pd.Series(idx.month.isin([11,12,1,2,3,4]).astype(float),index=idx),
      "vol_target":(0.15/park).clip(upper=3.0),
    }, ret

def ev(sig,ret):
    pos=sig.shift(1).fillna(0.0); r=(pos*ret-COST*pos.diff().abs().fillna(0.0)).dropna()
    if r.std()==0 or len(r)<60: return np.nan
    return float(r.mean()/r.std()*SQ)

sharpe={}; spans={}; worst={}
for a,f in ASSETS.items():
    df=pd.read_csv(os.path.join(DATA,f),parse_dates=["Date"]).set_index("Date").sort_index()
    S,ret=strategies(df)
    sharpe[a]={k:ev(v,ret) for k,v in S.items()}
    spans[a]=f"{df.index.min().date()}..{df.index.max().date()}"
    worst[a]=ret.min()*100

mat=pd.DataFrame(sharpe)
strat_order=["buy_hold","vol_target","golden_50_200","sma200","ma_20_50","tsmom_12m",
             "rsi2<10","buy_dip_3d","bollinger_mr","donchian","macd","turn_of_month","sell_in_may"]
mat=mat.loc[strat_order]
print("CROSS-ASSET SHARPE (net 2bps, long/flat)  — NQ=in-sample, rest=OOS\n")
print("data spans:", {a:spans[a] for a in ASSETS})
print("min daily ret % (split-artifact check):", {a:round(worst[a],0) for a in ASSETS}, "\n")
print(f"{'strategy':<15}" + "".join(f"{a:>8}" for a in ASSETS) + f"{'avg':>7}{'>BH':>5}")
bh=mat.loc["buy_hold"]
for s in strat_order:
    row=mat.loc[s]
    nbeat=sum(1 for a in ASSETS if pd.notna(row[a]) and pd.notna(bh[a]) and row[a]>bh[a]) if s!="buy_hold" else 0
    cells="".join(f"{row[a]:>8.2f}" if pd.notna(row[a]) else f"{'-':>8}" for a in ASSETS)
    print(f"{s:<15}{cells}{row.mean():>7.2f}{(str(nbeat)+'/6') if s!='buy_hold' else '':>5}")
print("\n'>BH' = # of the 6 OOS assets where the strategy beats THAT asset's buy&hold Sharpe.")
print("A strategy that 'works' should beat buy&hold across MOST assets, not just NQ.")
