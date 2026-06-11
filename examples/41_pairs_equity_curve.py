"""Example 41 - equity curves + benchmark for the vol-arb blend across ALL 8 pairs.
Curves constant-scaled to 10% ann vol (display only), compounded. Proxy instrument."""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0); TRAIN, TEST = 1260, 252
PAIRS = [("SPX/VIX","SPX","VIX",1),("NQ/VXN","NQ_F","VXN",1),("DJI/VXD","DJI","VXD",1),
         ("SX5E/VSTOXX","SX5E","VSTOXX",1),("NSEI/INDIAVIX","NSEI","INDIAVIX",1),
         ("WTI/OVX","WTI","OVX",0),("XAU/GVZ","XAU","GVZ",0),("EUR/EVZ","EURUSD","EVZ",0)]
GA = [(a,b,c) for a in (2.,4.,6.) for b in (0.,-2.) for c in (0.,1.,2.)]
GB = [(b1,b2) for b1 in (.8,1.,1.2) for b2 in (1.3,1.6,2.)]

def run(uf, vf):
    df = pd.read_csv(os.path.join(OUT,uf+"_all_history.csv"),parse_dates=["Date"]).set_index("Date").sort_index()
    vi = pd.read_csv(os.path.join(OUT,vf+"_all_history.csv"),parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    ret = df["Close"].pct_change(fill_method=None)
    idx = ret.index.intersection(vi.index)
    df, ret, vi = df.loc[idx], ret.loc[idx], vi.loc[idx]
    if len(idx) < TRAIN+TEST+50: return None, None
    pk = lambda w: (np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(w).mean()/(4*np.log(2)))*SQ*100).shift(1)
    rich = vi - pk(21); trend = pk(10) - pk(42)
    rng = np.log(df["High"]/df["Low"])*100; be = vi/SQ
    iv = (vi.shift(1)/100)**2/252; rvar = ret**2
    def sA(g):
        s = pd.Series(0., index=idx)
        s[((rich>=g[0])&(trend<=-g[2])).fillna(False)] = 1.
        s[((rich<=g[1])&(trend>=g[2])).fillna(False)] = -1.
        return s
    def sB(g):
        s = pd.Series(0., index=idx); ok = rng.notna()&be.notna()
        s[ok&(rng<g[0]*be)] = 1.; s[ok&(rng>g[1]*be)] = -1.
        return s
    def pnl(s):
        pos = s.clip(-1,1).shift(1).fillna(0.)
        c = (2*vi.shift(1)*0.5/1e4/252).fillna(0.)*pos.diff().abs().fillna(0.)
        return (pos*(iv-rvar)-c).dropna()
    def wf(fn, grid):
        ser = [pnl(fn(g)) for g in grid]
        bi = ser[0].index; parts = []; st = TRAIN
        while st+TEST <= len(bi):
            tr = bi[st-TRAIN:st]; te = bi[st:st+TEST]
            def sh(p):
                x = p.reindex(tr).dropna()
                return x.mean()/x.std()*SQ if len(x)>60 and x.std()>0 else -9.
            k = max(range(len(grid)), key=lambda i: sh(ser[i]))
            parts.append(ser[k].reindex(te)); st += TEST
        return pd.concat(parts).dropna() if parts else None
    pA, pB = wf(sA,GA), wf(sB,GB)
    if pA is None or pB is None: return None, None
    cm = pA.index.intersection(pB.index)
    return 0.5*pA.loc[cm]+0.5*pB.loc[cm], pnl(pd.Series(1.,index=idx)).reindex(cm).dropna()

rows = []; curves = {}
for name, uf, vf, iseq in PAIRS:
    bl, st = run(uf, vf)
    if bl is None: continue
    m = lambda p: (p.mean()/p.std()*SQ, p.mean()/(p.std()/np.sqrt(len(p))), p.skew())
    shB,tB,skB = m(bl); shS,tS,skS = m(st)
    yx = pd.concat([bl,st],axis=1).dropna().values; y,x = yx[:,0],yx[:,1]
    X = np.column_stack([np.ones(len(x)),x]); b,*_ = np.linalg.lstsq(X,y,rcond=None)
    r = y-X@b; at = b[0]/np.sqrt((r@r/(len(y)-2))*np.linalg.inv(X.T@X)[0,0])
    eq = (1 + bl*(0.10/(bl.std()*SQ))).cumprod()
    yrs = (bl.index[-1]-bl.index[0]).days/365.25
    rows.append(dict(pair=name, equity=("Y" if iseq else "N"), n=len(bl),
                     start=str(bl.index.min().date()), blend_sh=shB, blend_skew=skB,
                     static_sh=shS, static_skew=skS, alpha_t=at,
                     cagr10=eq.iloc[-1]**(1/yrs)-1, maxdd10=float((eq/eq.cummax()-1).min()),
                     term10=float(eq.iloc[-1])))
    curves[name] = (eq, iseq)

t = pd.DataFrame(rows).set_index("pair")
print("VOL-ARB BLEND ACROSS ALL PAIRS  (@10% vol footing; Sharpe/skew native; proxy instrument)")
print(f"{'pair':<15}{'eq?':>4}{'OOS start':>11}{'blend Sh':>9}{'skew':>6}{'static':>8}{'alpha-t':>8}{'CAGR@10%':>9}{'maxDD':>7}{'$1->':>7}")
for n_, r in t.sort_values('blend_sh', ascending=False).iterrows():
    print(f"{n_:<15}{r['equity']:>4}{r['start']:>11}{r['blend_sh']:>9.2f}{r['blend_skew']:>6.1f}"
          f"{r['static_sh']:>8.2f}{r['alpha_t']:>8.1f}{r['cagr10']*100:>8.1f}%{r['maxdd10']*100:>6.0f}%{r['term10']:>6.1f}x")
t.to_csv(os.path.join(OUT,"pairs_benchmark.csv"))

fig, ax = plt.subplots(figsize=(13,7.5))
for name,(eq,iseq) in curves.items():
    ax.plot(eq.index, eq.values, lw=1.8 if iseq else 1.2, ls="-" if iseq else "--",
            label=f"{name}  (Sh {t.loc[name,'blend_sh']:+.2f}, a-t {t.loc[name,'alpha_t']:+.1f})")
ax.set_yscale("log"); ax.axhline(1, color="grey", lw=.5)
ax.set_ylabel("growth of $1 (log) - each pair's blend at identical 10% vol")
ax.set_title("Regime-gated long-short vol-arb across 8 markets - walk-forward OOS, proxy instrument\n"
             "solid = equity-index pairs (the strategy's domain), dashed = non-equity (fails)")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"pairs_equity_curve.png"), dpi=110); plt.close(fig)
print(f"\nchart -> {os.path.join(OUT,'pairs_equity_curve.png')}")
