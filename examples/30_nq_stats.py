"""NQ all-history statistical features — returns + range vol, one page."""
import os, numpy as np, pandas as pd
D = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

df = pd.read_csv(os.path.join(D, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
ret = c.pct_change().dropna()
rv = ((h - l) / o).dropna()
yrs = (c.index[-1] - c.index[0]).days / 365.25
eq = (1 + ret).cumprod()
dd = eq / eq.cummax() - 1

print(f"NQ (Nasdaq-100 future) {c.index.min().date()} .. {c.index.max().date()}  "
      f"({len(ret)} days, {yrs:.1f} years)")

print("\n" + "=" * 76)
print("DAILY RETURNS")
print("=" * 76)
q = ret.quantile([.01, .05, .25, .50, .75, .95, .99])
print(f"  mean {ret.mean()*1e4:+.1f} bps/day  ann return {eq.iloc[-1]**(1/yrs)-1:+.1%}  "
      f"ann vol {ret.std()*SQ:.1%}  Sharpe {ret.mean()/ret.std()*SQ:.2f}")
print(f"  skew {ret.skew():+.2f}   kurtosis {ret.kurt():.1f} (normal=0)   "
      f"min {ret.min():+.1%} ({ret.idxmin().date()})   max {ret.max():+.1%} ({ret.idxmax().date()})")
print("  percentiles: " + "  ".join(f"p{int(p*100)}={v*100:+.2f}%" for p, v in q.items()))
print(f"  %days up {100*(ret>0).mean():.1f}%   autocorr lag1 {ret.autocorr(1):+.2f} (returns ~unpredictable)")
print(f"  |moves|>3%: {(ret.abs()>0.03).sum()} days ({(ret.abs()>0.03).mean()*100:.1f}%)   "
      f">5%: {(ret.abs()>0.05).sum()}   normal would predict ~{len(ret)*2*0.00135:.0f} >3sigma days, actual {(ret.abs()>3*ret.std()).sum()}")

print("\n" + "=" * 76)
print("DRAWDOWN / PATH")
print("=" * 76)
print(f"  terminal $1 -> {eq.iloc[-1]:.1f}x   maxDD {dd.min():.0%} (trough {dd.idxmin().date()})   "
      f"days in drawdown {100*(dd<0).mean():.0f}%")
worst_yr = (1 + ret).groupby(ret.index.year).prod().sub(1)
print(f"  best year {worst_yr.max():+.0%} ({worst_yr.idxmax()})   worst year {worst_yr.min():+.0%} ({worst_yr.idxmin()})")

print("\n" + "=" * 76)
print("RANGE VOLATILITY  (High-Low)/Open")
print("=" * 76)
qr = rv.quantile([.05, .25, .50, .75, .90, .95, .99])
print(f"  mean {rv.mean():.2%}  median {rv.median():.2%}  std {rv.std():.2%}  max {rv.max():.1%} ({rv.idxmax().date()})")
print(f"  skew {rv.skew():+.2f}  kurtosis {rv.kurt():.1f}  (right-skewed, fat-tailed)")
print("  percentiles: " + "  ".join(f"p{int(p*100)}={v*100:.2f}%" for p, v in qr.items()))
park = np.sqrt((np.log(h/l)**2).mean()/(4*np.log(2)))*SQ
print(f"  Parkinson ann vol {park:.1%} vs close-close {ret.std()*SQ:.1%}  (range estimator, ~5x more efficient)")
print(f"  persistence (vol clustering): autocorr lag1 {rv.autocorr(1):.2f}  lag5 {rv.autocorr(5):.2f}  lag20 {rv.autocorr(20):.2f}")
print(f"  corr(range, same-day return) {rv.corr(ret.reindex(rv.index)):+.2f}   "
      f"corr(range_t, |ret|_t+1) {rv.corr(ret.abs().shift(-1).reindex(rv.index)):+.2f}  <- vol predicts vol, not direction")
