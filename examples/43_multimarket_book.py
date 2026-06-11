"""Example 43 - REAL-WORLD TRADEABLE MULTI-MARKET EQUITY-VOL BOOK.
Instrument: short-dated delta-hedged ATM straddle (gamma expression), premium k*VolIdx,
k=0.82 (EMPIRICALLY MEASURED from VIX1D/VIX, ex 39). short pnl = prem - |ret|.
Costs: 2.5% of premium per in-market day (US/ETF), 4% (non-US weekly markets).
Signals: established A/B gates, per-market walk-forward (1260/252) - no new search.
Book rules A-PRIORI: equal-risk sleeves (causal 63d vol, shifted), dynamic availability,
book-level causal vol-target 10% (cap 4, pinning disclosed).
Benchmarks: static short-vol book (same construction), SPX buy&hold, underlying basket."""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
from math import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0); DT = 1/252.0; TRAIN, TEST = 1260, 252; K = 0.82
PAIRS = [("SPX","SPX","VIX",.025),("NQ","NQ_F","VXN",.025),("EEM","EEM","VXEEM",.025),
         ("DAX","DAX","VDAX",.04),("SX5E","SX5E","VSTOXX",.04),("N225","N225","JNIV",.04),
         ("HSI","HSI","VHSI",.04),("NIFTY","NSEI","INDIAVIX",.04)]
GA = [(a,b,c) for a in (2.,4.,6.) for b in (0.,-2.) for c in (0.,1.,2.)]
GB = [(b1,b2) for b1 in (.8,1.,1.2) for b2 in (1.3,1.6,2.)]
Nrm = lambda x: 0.5*(1.0+np.vectorize(erf)(x/np.sqrt(2.0)))

def market(uf, vf, spread):
    df = pd.read_csv(os.path.join(OUT,uf+"_all_history.csv"),parse_dates=["Date"]).set_index("Date").sort_index()
    vi = pd.read_csv(os.path.join(OUT,vf+"_all_history.csv"),parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    ret = df["Close"].pct_change(fill_method=None)
    idx = ret.index.intersection(vi.index)
    df, ret, vi = df.loc[idx], ret.loc[idx], vi.loc[idx]
    if len(idx) < TRAIN+TEST+50: return None
    prem = pd.Series(2*(2*Nrm(0.5*(K*vi.values/100)*np.sqrt(DT))-1), index=idx)
    pk = lambda w: (np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(w).mean()/(4*np.log(2)))*SQ*100).shift(1)
    rich = vi - pk(21); trend = pk(10) - pk(42)
    rng = np.log(df["High"]/df["Low"])*100; be = vi/SQ
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
        return (pos*(prem.shift(1)-ret.abs()) - spread*prem.shift(1)*pos.abs()).dropna()
    def wf(fn, grid):
        ser = [pnl(fn(g)) for g in grid]
        bi = ser[0].index; parts = []; st = TRAIN
        while st+TEST <= len(bi):
            tr = bi[st-TRAIN:st]; te = bi[st:st+TEST]
            def sh(p):
                x = p.reindex(tr).dropna()
                return x.mean()/x.std()*SQ if len(x)>60 and x.std()>0 else -9.
            kk = max(range(len(grid)), key=lambda i: sh(ser[i]))
            parts.append(ser[kk].reindex(te)); st += TEST
        return pd.concat(parts).dropna() if parts else None
    pA, pB = wf(sA,GA), wf(sB,GB)
    if pA is None or pB is None: return None
    cm = pA.index.intersection(pB.index)
    return (0.5*pA.loc[cm]+0.5*pB.loc[cm], pnl(pd.Series(1.,index=idx)).reindex(cm).dropna(), ret)

sleeves = {}; statics = {}; unds = {}
for name, uf, vf, sp in PAIRS:
    r = market(uf, vf, sp)
    if r: sleeves[name], statics[name], unds[name] = r

def book_of(d):
    # causal equal-risk: scale each sleeve on its OWN calendar (else cross-market holiday
    # NaNs starve the rolling windows), then union-align and average what's available
    sc = {k: (p/(p.rolling(63).std().shift(1))) for k, p in d.items()}
    bk = pd.DataFrame(sc).mean(axis=1, skipna=True).dropna()
    lev = (0.10/(bk.rolling(63).std().shift(1)*SQ)).clip(upper=4.0)
    return (lev*bk).dropna(), float((lev >= 3.99).mean())

book, pin_b = book_of(sleeves)
static_bk, pin_s = book_of(statics)
spx = unds["SPX"].reindex(book.index).fillna(0.)
basket = pd.DataFrame(unds).reindex(book.index).mean(axis=1).fillna(0.)

def stats(r):
    r = r.dropna(); e = (1+r).cumprod(); yrs = (r.index[-1]-r.index[0]).days/365.25
    return dict(sh=r.mean()/r.std()*SQ, t=r.mean()/(r.std()/np.sqrt(len(r))), sk=r.skew(),
                cagr=e.iloc[-1]**(1/yrs)-1, dd=float((e/e.cummax()-1).min()),
                w1y=float(e.pct_change(252).min()), term=float(e.iloc[-1]))

cm = book.index
rows = [("VOL BOOK (gated)", book), ("static short-vol book", static_bk.reindex(cm).dropna()),
        ("SPX buy&hold", spx), ("equal-wt underlyings", basket)]
print(f"MULTI-MARKET EQUITY-VOL BOOK  {cm.min().date()}..{cm.max().date()}  ({len(sleeves)} markets, k={K} measured)")
print(f"{'':<24}{'Sharpe':>8}{'t':>7}{'skew':>6}{'CAGR':>7}{'maxDD':>7}{'worst1y':>8}{'$1->':>7}")
for n_, r in rows:
    s = stats(r)
    print(f"{n_:<24}{s['sh']:>8.2f}{s['t']:>7.1f}{s['sk']:>6.1f}{s['cagr']*100:>6.1f}%{s['dd']*100:>6.0f}%"
          f"{s['w1y']*100:>7.0f}%{s['term']:>6.1f}x")
yx = pd.concat([book, static_bk], axis=1).dropna().values
y, x = yx[:,0], yx[:,1]; X = np.column_stack([np.ones(len(x)),x]); b,*_ = np.linalg.lstsq(X,y,rcond=None)
res = y-X@b; at = b[0]/np.sqrt((res@res/(len(y)-2))*np.linalg.inv(X.T@X)[0,0])
print(f"\nbook alpha-t vs static book {at:+.1f} (beta {b[1]:.2f}) | vol-target pinned@cap: book {pin_b*100:.0f}%, static {pin_s*100:.0f}%")
C = pd.DataFrame(sleeves).corr()
ut = C.values[np.triu_indices_from(C.values,1)]
print(f"sleeve correlations: mean {np.nanmean(ut):.2f}  max {np.nanmax(ut):.2f}")
print("\nyearly Sharpe (book | static | SPX):")
for yname, g in book.groupby(book.index.year):
    if len(g) < 60: continue
    s2 = static_bk.reindex(g.index).dropna(); s3 = spx.reindex(g.index).dropna()
    f = lambda p: p.mean()/p.std()*SQ if len(p)>60 and p.std()>0 else float("nan")
    print(f"  {yname}: {f(g):+5.1f} | {f(s2):+5.1f} | {f(s3):+5.1f}")

fig, ax = plt.subplots(3,1, figsize=(12,11), gridspec_kw={"height_ratios":[3,1,1]}, sharex=True)
for n_, r in rows:
    e = (1+r.fillna(0)).cumprod()
    ax[0].plot(e.index, e.values, lw=2.0 if "BOOK" in n_ else 1.1, label=f"{n_} (Sh {stats(r)['sh']:.2f})")
ax[0].set_yscale("log"); ax[0].legend(loc="upper left"); ax[0].set_ylabel("growth of $1 (log)")
ax[0].set_title(f"Multi-market equity-vol book ({len(sleeves)} markets, straddle instrument, k={K} measured, walk-forward)")
e = (1+book).cumprod(); dd = e/e.cummax()-1
ax[1].fill_between(dd.index, dd.values*100, 0, color="firebrick", alpha=.5); ax[1].set_ylabel("book DD %")
rs = book.rolling(252).mean()/book.rolling(252).std()*SQ
ax[2].plot(rs.index, rs.values, color="navy"); ax[2].axhline(0, color="grey", lw=.5); ax[2].set_ylabel("rolling 1y Sharpe")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"multimarket_book.png"), dpi=110); plt.close(fig)
pd.DataFrame({"book_ret":book}).to_csv(os.path.join(OUT,"multimarket_book_ledger.csv"), index_label="Date")
print(f"\nchart -> multimarket_book.png | ledger -> multimarket_book_ledger.csv")
