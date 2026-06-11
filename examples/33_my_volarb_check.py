"""Independent ground-truth check for the vol-arb workflow:
(1) does richness-timing add alpha vs static short-vol?  (2) do long-vol legs pay?"""
import os, numpy as np, pandas as pd
D = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

df = pd.read_csv(os.path.join(D, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
vxn = pd.read_csv(os.path.join(D, "VXN_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
ret = df["Close"].pct_change()
idx = ret.index.intersection(vxn.index)
df, ret, vxn = df.loc[idx], ret.loc[idx], vxn.loc[idx]

iv = (vxn.shift(1) / 100.0) ** 2 / 252.0
rvar = ret ** 2
static = (iv - rvar).dropna()                       # always-short variance

park21 = (np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(21).mean()/(4*np.log(2)))*SQ*100).shift(1)
rich = (vxn - park21)                                # vol pts of richness, causal

def m(p):
    p = p.dropna()
    return (p.mean()/p.std()*SQ, p.mean()/(p.std()/np.sqrt(len(p))), p.skew(), len(p))

print(f"period {idx.min().date()}..{idx.max().date()}")
sh, t, sk, n = m(static)
print(f"static short-vol: Sharpe {sh:+.2f} (t {t:+.1f}) skew {sk:+.1f}  [the bar]")

print("\n(1) richness deciles -> NEXT-day short-vol pnl (is the premium conditional on richness?)")
dec = pd.qcut(rich.dropna(), 5, labels=False)
tbl = pd.DataFrame({"d": dec, "pnl": static.reindex(dec.index)}).dropna()
for d, g in tbl.groupby("d"):
    sh_d = g["pnl"].mean()/g["pnl"].std()*SQ
    print(f"   richness quintile {int(d)+1} ({'cheapest' if d==0 else 'richest' if d==4 else ''}): "
          f"mean {g['pnl'].mean()*1e6:+.2f}e-6  Sharpe {sh_d:+.2f}  skew {g['pnl'].skew():+.1f}  n={len(g)}")

print("\n(2) simple timed long-short (no optimization, single a-priori rule: k=2 vol pts):")
s = pd.Series(0.0, index=idx); s[rich > 2] = 1.0; s[rich < -2] = -1.0
pos = s.shift(1).fillna(0.0)
cost = (2*vxn.shift(1)*0.5/1e4/252).fillna(0)*pos.diff().abs().fillna(0)
ls = (pos*(iv - rvar) - cost).dropna()
sh2, t2, sk2, n2 = m(ls)
print(f"   long-short k=2: Sharpe {sh2:+.2f} (t {t2:+.1f}) skew {sk2:+.1f}  "
      f"%short {100*(pos>0).mean():.0f}%  %long {100*(pos<0).mean():.0f}%")
# alpha vs static
common = ls.index.intersection(static.index)
y = ls.loc[common].values; x = static.loc[common].values
b = np.cov(x, y)[0,1]/np.var(x); a = y.mean() - b*x.mean()
resid = y - (a + b*x); ta = a/(resid.std()/np.sqrt(len(y)))
print(f"   vs static: beta {b:.2f}  alpha t-stat {ta:+.1f}  ({'ALPHA' if abs(ta)>=2 else 'just timed beta'})")

print("\n(3) do LONG-vol legs pay? pnl of the long legs only (rich < -2):")
lv = (-(iv - rvar)).reindex(idx)[rich.shift(0) < -2].dropna()   # long-vol pnl on cheap days
if len(lv) > 60:
    shl, tl, skl, nl = m(lv)
    print(f"   long-vol on 'cheap' days: Sharpe {shl:+.2f} (t {tl:+.1f}) skew {skl:+.1f} n={nl}")
else:
    print(f"   only {len(lv)} long-vol days — too few to judge")
