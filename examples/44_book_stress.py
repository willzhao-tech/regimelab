"""Example 44 - STRESS THE MULTI-MARKET BOOK: friction ladder, spread sweep, leave-one-out, OOS split.
Frictions modeled (cumulative ladder):
 L0 baseline: proportional spread (2.5% US / 4% non-US of premium per in-market day)
 L1 + stress widening: spread *= (1 + max(0, VolIdx/median63 - 1))  (2x median vol => 2x spread)
 L2 + long-vol penalty: LONG legs pay 2x the (widened) spread  (bid-ask widens when buying vol)
 L3 + discrete strikes: strike off-ATM by alternating +/-0.125%; payoff |ret - eps| (premium unchanged = conservative)
 L4 + floors/assignment: cost floor 1.5bp of notional per in-market day; EEM (American-style ETF) +0.5% premium penalty
Then: spread MULTIPLIER sweep (x1/x1.5/x2/x3) on top of L4; leave-one-out; 2004-2015 vs 2015-2026 split."""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
from math import erf

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0); DT = 1/252.0; TRAIN, TEST = 1260, 252; K = 0.82
PAIRS = [("SPX","SPX","VIX",.025),("NQ","NQ_F","VXN",.025),("EEM","EEM","VXEEM",.025),
         ("DAX","DAX","VDAX",.04),("SX5E","SX5E","VSTOXX",.04),("N225","N225","JNIV",.04),
         ("HSI","HSI","VHSI",.04),("NIFTY","NSEI","INDIAVIX",.04)]
GA = [(a,b,c) for a in (2.,4.,6.) for b in (0.,-2.) for c in (0.,1.,2.)]
GB = [(b1,b2) for b1 in (.8,1.,1.2) for b2 in (1.3,1.6,2.)]
Nrm = lambda x: 0.5*(1.0+np.vectorize(erf)(x/np.sqrt(2.0)))

DATA = {}
for name, uf, vf, sp in PAIRS:
    df = pd.read_csv(os.path.join(OUT,uf+"_all_history.csv"),parse_dates=["Date"]).set_index("Date").sort_index()
    vi = pd.read_csv(os.path.join(OUT,vf+"_all_history.csv"),parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    ret = df["Close"].pct_change(fill_method=None)
    idx = ret.index.intersection(vi.index)
    DATA[name] = (df.loc[idx], ret.loc[idx], vi.loc[idx], sp)

def market(name, level, mult=1.0):
    df, ret, vi, sp0 = DATA[name]
    idx = ret.index
    if len(idx) < TRAIN+TEST+50: return None, None
    prem = pd.Series(2*(2*Nrm(0.5*(K*vi.values/100)*np.sqrt(DT))-1), index=idx)
    pk = lambda w: (np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(w).mean()/(4*np.log(2)))*SQ*100).shift(1)
    rich = vi - pk(21); trend = pk(10) - pk(42)
    rng = np.log(df["High"]/df["Low"])*100; be = vi/SQ
    spread = pd.Series(sp0, index=idx)
    if level >= 1:
        spread = spread*(1 + (vi/vi.rolling(63).median().shift(1) - 1).clip(lower=0)).fillna(sp0)
    if level >= 4 and name == "EEM":
        spread = spread + 0.005
    spread = spread*mult
    eps = pd.Series(0.00125*np.where(np.arange(len(idx))%2, 1, -1), index=idx) if level >= 3 else pd.Series(0., index=idx)
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
        payoff = prem.shift(1) - (ret - eps).abs()
        c = spread.shift(1)*prem.shift(1)*pos.abs()
        if level >= 2:
            c = c + spread.shift(1)*prem.shift(1)*pos.abs().where(pos<0, 0.)   # long legs pay 2x
        if level >= 4:
            c = c.where(~(pos.abs()>0), np.maximum(c, 0.00015))                # 1.5bp floor
        return (pos*payoff - c).dropna()
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
    if pA is None or pB is None: return None, None
    cm = pA.index.intersection(pB.index)
    return 0.5*pA.loc[cm]+0.5*pB.loc[cm], pnl(pd.Series(1.,index=idx)).reindex(cm).dropna()

def book_of(d):
    sc = {k: (p/(p.rolling(63).std().shift(1))) for k, p in d.items()}
    bk = pd.DataFrame(sc).mean(axis=1, skipna=True).dropna()
    lev = (0.10/(bk.rolling(63).std().shift(1)*SQ)).clip(upper=4.0)
    return (lev*bk).dropna()

def sh(r):
    r = r.dropna(); return r.mean()/r.std()*SQ if len(r) > 100 and r.std() > 0 else np.nan
def dd(r):
    e = (1+r.dropna()).cumprod(); return float((e/e.cummax()-1).min())

print("(1) FRICTION LADDER (cumulative) — book Sharpe / maxDD / alpha-t vs same-friction static")
lvl_names = ["L0 baseline","L1 +stress-widen","L2 +longvol 2x","L3 +strike offset","L4 +floors/assign"]
sleeves_L4 = {}
for lv in range(5):
    sl = {}; st_ = {}
    for name,_,_,_ in PAIRS:
        a, b = market(name, lv)
        if a is not None: sl[name] = a; st_[name] = b
    bk = book_of(sl); sb = book_of(st_)
    if lv == 4: sleeves_L4 = sl
    yx = pd.concat([bk, sb], axis=1).dropna().values
    y, x = yx[:,0], yx[:,1]; X = np.column_stack([np.ones(len(x)),x]); bcoef,*_ = np.linalg.lstsq(X,y,rcond=None)
    res = y-X@bcoef; at = bcoef[0]/np.sqrt((res@res/(len(y)-2))*np.linalg.inv(X.T@X)[0,0])
    print(f"  {lvl_names[lv]:<20} Sharpe {sh(bk):5.2f}  maxDD {dd(bk)*100:4.0f}%  static {sh(sb):+5.2f}  alpha-t {at:+5.1f}")

print("\n(2) SPREAD MULTIPLIER SWEEP on top of FULL frictions (L4)")
for mult in (1.0, 1.5, 2.0, 3.0):
    sl = {}
    for name,_,_,_ in PAIRS:
        a, _ = market(name, 4, mult)
        if a is not None: sl[name] = a
    bk = book_of(sl)
    print(f"  spreads x{mult:<4} book Sharpe {sh(bk):5.2f}  maxDD {dd(bk)*100:4.0f}%")

print("\n(3) LEAVE-ONE-OUT (full frictions L4) + per-sleeve Sharpe")
for name in list(sleeves_L4):
    print(f"  sleeve {name:<6} own Sharpe {sh(sleeves_L4[name]):5.2f}   book WITHOUT it: "
          f"{sh(book_of({k:v for k,v in sleeves_L4.items() if k != name})):5.2f}")
print(f"  full book (L4): {sh(book_of(sleeves_L4)):.2f}   US-only(SPX,NQ,EEM): "
      f"{sh(book_of({k:sleeves_L4[k] for k in ('SPX','NQ','EEM') if k in sleeves_L4})):.2f}   "
      f"ex-US: {sh(book_of({k:v for k,v in sleeves_L4.items() if k not in ('SPX','NQ','EEM')})):.2f}")

print("\n(4) WINDOW SPLIT (full frictions L4)")
bk4 = book_of(sleeves_L4)
half = bk4.index[len(bk4)//2]
print(f"  first half  {bk4.index.min().date()}..{half.date()}:  Sharpe {sh(bk4[:half]):.2f}  maxDD {dd(bk4[:half])*100:.0f}%")
print(f"  second half {half.date()}..{bk4.index.max().date()}:  Sharpe {sh(bk4[half:]):.2f}  maxDD {dd(bk4[half:])*100:.0f}%")
