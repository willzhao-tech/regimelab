"""Run the popular NQ strategies (from web research) over all history, honestly.
Strict causal convention: every signal uses data through close_t; position is signal.shift(1)
applied to ret_t (decide at close t, hold day t+1). Net of cost. Compare all vs buy & hold.
"""
import os, numpy as np, pandas as pd
DATA = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)
COST = 0.0002  # 2 bps per unit turnover (liquid NQ futures)

df = pd.read_csv(os.path.join(DATA,"NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
o,h,l,c = df["Open"],df["High"],df["Low"],df["Close"]
ret = c.pct_change()
yrs = (c.index[-1]-c.index[0]).days/365.25

def rsi(s,n):
    d=s.diff(); up=d.clip(lower=0); dn=(-d).clip(lower=0)
    return 100-100/(1+up.ewm(alpha=1/n,adjust=False).mean()/dn.ewm(alpha=1/n,adjust=False).mean())
sma=lambda n: c.rolling(n).mean()

def state(entry, exit_):
    pos=np.zeros(len(c)); s=0
    e1=entry.values; e2=exit_.values
    for i in range(len(c)):
        if s==0 and e1[i]: s=1
        elif s==1 and e2[i]: s=0
        pos[i]=s
    return pd.Series(pos,index=c.index)

r2=rsi(c,2)
mid=sma(20); sd=c.rolling(20).std()
ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean()
macd=ema12-ema26; macsig=macd.ewm(span=9,adjust=False).mean()
idx=c.index; dom=pd.Series(idx,index=idx).groupby([idx.year,idx.month]).cumcount()
dom_rev=pd.Series(idx,index=idx).groupby([idx.year,idx.month]).cumcount(ascending=False)
park=np.sqrt((np.log(h/l)**2).rolling(21).mean()/(4*np.log(2)))*SQ
down3=(ret<0)&(ret.shift(1)<0)&(ret.shift(2)<0)

S={}
S["buy_hold"]=pd.Series(1.0,index=c.index)
S["sma200_filter"]=(c>sma(200)).astype(float)
S["golden_cross_50_200"]=(sma(50)>sma(200)).astype(float)
S["ma_cross_20_50"]=(sma(20)>sma(50)).astype(float)
S["tsmom_12m"]=(c>c.shift(252)).astype(float)
S["macd"]=(macd>macsig).astype(float)
S["rsi2_connors<10"]=state((r2<10)&(c>sma(200)), c>sma(5))
S["rsi2_extreme<5"]=state((r2<5)&(c>sma(200)), c>sma(5))
S["bollinger_mr"]=state(c<(mid-2*sd), c>mid)
S["donchian_20_10"]=state(c>=h.rolling(20).max().shift(1), c<=l.rolling(10).min().shift(1))
S["buy_the_dip_3d"]=state(down3, ret>0)
S["turn_of_month"]=((dom<3)|(dom_rev==0)).astype(float)
S["sell_in_may"]=pd.Series(c.index.month.isin([11,12,1,2,3,4]).astype(float),index=c.index)
S["vol_target_15"]=(0.15/park).clip(upper=3.0)   # the session's survivor (leverage)

def evalr(sig):
    pos=sig.shift(1).fillna(0.0)
    r=(pos*ret - COST*pos.diff().abs().fillna(0.0)).dropna()
    e=(1+r).cumprod()
    cg=e.iloc[-1]**(252/len(r))-1; dd=float((e/e.cummax()-1).min())
    return dict(sharpe=r.mean()/r.std()*SQ if r.std()>0 else np.nan, cagr=cg, maxdd=dd,
                calmar=cg/abs(dd) if dd<0 else np.nan, vol=r.std()*SQ,
                inmkt=pos.mean()*100, trades=pos.diff().abs().sum()/2/yrs, term=float(e.iloc[-1]))

rows={k:evalr(v) for k,v in S.items()}
# overnight vs intraday (where return accrues) - separate streams
on=(o/c.shift(1)-1).dropna(); intra=(c/o-1).dropna()
rows["overnight_only"]=dict(sharpe=on.mean()/on.std()*SQ,cagr=(1+on).prod()**(252/len(on))-1,
    maxdd=float(((1+on).cumprod()/(1+on).cumprod().cummax()-1).min()),calmar=np.nan,vol=on.std()*SQ,inmkt=100,trades=0,term=float((1+on).prod()))
rows["intraday_only"]=dict(sharpe=intra.mean()/intra.std()*SQ,cagr=(1+intra).prod()**(252/len(intra))-1,
    maxdd=float(((1+intra).cumprod()/(1+intra).cumprod().cummax()-1).min()),calmar=np.nan,vol=intra.std()*SQ,inmkt=100,trades=0,term=float((1+intra).prod()))

tab=pd.DataFrame(rows).T.sort_values("sharpe",ascending=False)
print(f"NQ {c.index.min().date()}..{c.index.max().date()} ({yrs:.0f}y), net of {COST*1e4:.0f}bps/turnover\n")
print(f"{'strategy':<22}{'Sharpe':>7}{'CAGR':>7}{'maxDD':>7}{'Calmar':>7}{'vol':>6}{'%inMkt':>7}{'trd/yr':>7}{'$1->':>8}")
for k,r in tab.iterrows():
    print(f"{k:<22}{r['sharpe']:>7.2f}{r['cagr']*100:>6.1f}%{r['maxdd']*100:>6.0f}%"
          f"{r['calmar']:>7.2f}{r['vol']*100:>5.0f}%{r['inmkt']:>6.0f}%{r['trades']:>7.1f}{r['term']:>7.1f}x")
bh=rows["buy_hold"]
print(f"\nvs buy&hold (Sharpe {bh['sharpe']:.2f}, CAGR {bh['cagr']*100:.1f}%, maxDD {bh['maxdd']*100:.0f}%, {bh['term']:.0f}x):")
beat_sh=[k for k,r in rows.items() if r['sharpe']>bh['sharpe'] and k!='buy_hold']
beat_cagr=[k for k,r in rows.items() if r['cagr']>bh['cagr'] and k!='buy_hold']
print(f"  beat B&H on Sharpe: {beat_sh}")
print(f"  beat B&H on CAGR:   {beat_cagr}")
