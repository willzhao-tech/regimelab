"""Example 42 - 4 NEW TRADEABLE pairs: N225/JNIV, HSI/VHSI, DAX/VDAX, EEM/VXEEM.
Identical methodology (signals/grids/walk-forward/proxy, 0.5vp cost). No re-tuning."""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0); TRAIN, TEST = 1260, 252
PAIRS = [("N225/JNIV","N225","JNIV"),("HSI/VHSI","HSI","VHSI"),
         ("DAX/VDAX","DAX","VDAX"),("EEM/VXEEM","EEM","VXEEM")]
GA = [(a,b,c) for a in (2.,4.,6.) for b in (0.,-2.) for c in (0.,1.,2.)]
GB = [(b1,b2) for b1 in (.8,1.,1.2) for b2 in (1.3,1.6,2.)]

def run(uf, vf):
    df = pd.read_csv(os.path.join(OUT,uf+"_all_history.csv"),parse_dates=["Date"]).set_index("Date").sort_index()
    vi = pd.read_csv(os.path.join(OUT,vf+"_all_history.csv"),parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    ret = df["Close"].pct_change(fill_method=None)
    idx = ret.index.intersection(vi.index)
    df, ret, vi = df.loc[idx], ret.loc[idx], vi.loc[idx]
    realized = ret.std()*SQ*100
    if len(idx) < TRAIN+TEST+50: return None
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
    if pA is None or pB is None: return None
    cm = pA.index.intersection(pB.index)
    bl = 0.5*pA.loc[cm]+0.5*pB.loc[cm]
    st = pnl(pd.Series(1.,index=idx)).reindex(cm).dropna()
    m = lambda p: (p.mean()/p.std()*SQ, p.mean()/(p.std()/np.sqrt(len(p))), p.skew())
    shB,tB,skB = m(bl); shS,tS,skS = m(st)
    yx = pd.concat([bl,st],axis=1).dropna().values; y,x = yx[:,0],yx[:,1]
    X = np.column_stack([np.ones(len(x)),x]); b,*_ = np.linalg.lstsq(X,y,rcond=None)
    r = y-X@b; at = b[0]/np.sqrt((r@r/(len(y)-2))*np.linalg.inv(X.T@X)[0,0])
    return dict(n=len(bl), start=str(bl.index.min().date()), vol_idx=vi.mean(), realized=realized,
                blend_sh=shB, blend_t=tB, blend_skew=skB, static_sh=shS, static_skew=skS, alpha_t=at)

print("4 NEW TRADEABLE PAIRS — identical strategy (proxy instrument)\n")
print(f"{'pair':<11}{'OOS n':>6}{'start':>11}{'volIdx':>7}{'realzd':>7} | {'static Sh/skew':>15} | {'blend Sh (t)':>13}{'skew':>6}{'alpha-t':>8}")
rows = []
for name, uf, vf in PAIRS:
    res = run(uf, vf)
    if res is None: print(f"{name:<11} insufficient data"); continue
    rows.append({**res, "pair": name})
    print(f"{name:<11}{res['n']:>6}{res['start']:>11}{res['vol_idx']:>7.1f}{res['realized']:>7.1f} | "
          f"{res['static_sh']:>6.2f} /{res['static_skew']:>6.1f} | {res['blend_sh']:>6.2f} ({res['blend_t']:>+4.1f})"
          f"{res['blend_skew']:>6.1f}{res['alpha_t']:>8.1f}")
pd.DataFrame(rows).to_csv(os.path.join(OUT,"p1_tradeable_pairs.csv"), index=False)
print(f"\nsaved -> {os.path.join(OUT,'p1_tradeable_pairs.csv')}")
print("(volIdx vs realzd column = sanity check that the vol index is on the right scale)")
