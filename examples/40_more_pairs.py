"""
Example 40 — universality test on 5 NEW underlying/vol-index pairs (3 non-equity).

Pairs: DJI/VXD (2005+), WTI/OVX (2007+, OIL), XAU/GVZ (2008+, GOLD),
       EURUSD/EVZ (2008-2025, FX), NSEI/INDIAVIX (2008+, India).
IDENTICAL methodology to the validated NQ/VXN & SPX/VIX runs: same signals, same grids,
same walk-forward (1260d train -> 252d test), proxy instrument, 0.5 vol-pt cost.
NO re-tuning — pure replication. Thresholds are in absolute vol points, so on low-vol
assets (EVZ ~10) the gates may rarely fire: breadth is reported, that is a finding too.
"""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0); TRAIN, TEST = 1260, 252

PAIRS = [
    ("DJI/VXD",        "DJI_all_history.csv",    "VXD_all_history.csv"),
    ("WTI/OVX (oil)",  "WTI_all_history.csv",    "OVX_all_history.csv"),
    ("XAU/GVZ (gold)", "XAU_all_history.csv",    "GVZ_all_history.csv"),
    ("EUR/EVZ (fx)",   "EURUSD_all_history.csv", "EVZ_all_history.csv"),
    ("NSEI/INDIAVIX",  "NSEI_all_history.csv",   "INDIAVIX_all_history.csv"),
]
GRID_A = [(a, b, c) for a in (2.0, 4.0, 6.0) for b in (0.0, -2.0) for c in (0.0, 1.0, 2.0)]
GRID_B = [(b1, b2) for b1 in (0.8, 1.0, 1.2) for b2 in (1.3, 1.6, 2.0)]

def run_pair(name, und_file, vol_file):
    df = pd.read_csv(os.path.join(OUT, und_file), parse_dates=["Date"]).set_index("Date").sort_index()
    vi = pd.read_csv(os.path.join(OUT, vol_file), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
    ret = df["Close"].pct_change()
    idx = ret.index.intersection(vi.index)
    df, ret, vi = df.loc[idx], ret.loc[idx], vi.loc[idx]
    if len(idx) < TRAIN + TEST + 50:
        return None
    park = lambda w: (np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(w).mean()/(4*np.log(2)))*SQ*100).shift(1)
    fc21, fc10, fc42 = park(21), park(10), park(42)
    rich = vi - fc21; trend = fc10 - fc42
    rng = np.log(df["High"]/df["Low"])*100.0; be = vi/SQ
    iv = (vi.shift(1)/100.0)**2/252.0; rvar = ret**2

    def sig_A(g):
        r_hi, r_lo, d = g
        s = pd.Series(0.0, index=idx)
        s[((rich >= r_hi) & (trend <= -d)).fillna(False)] = 1.0
        s[((rich <= r_lo) & (trend >= d)).fillna(False)] = -1.0
        return s

    def sig_B(g):
        b1, b2 = g
        s = pd.Series(0.0, index=idx); ok = rng.notna() & be.notna()
        s[ok & (rng < b1*be)] = 1.0
        s[ok & (rng > b2*be)] = -1.0
        return s

    def pnl_of(s):
        pos = s.clip(-1, 1).shift(1).fillna(0.0)
        cost = (2*vi.shift(1)*0.5/1e4/252).fillna(0.0)*pos.diff().abs().fillna(0.0)
        return (pos*(iv - rvar) - cost).dropna(), pos

    def wf(sig_fn, grid):
        series = [pnl_of(sig_fn(g))[0] for g in grid]
        bidx = series[0].index; parts = []; pparts = []; start = TRAIN
        while start + TEST <= len(bidx):
            tr = bidx[start-TRAIN:start]; te = bidx[start:start+TEST]
            def sh(p):
                x = p.reindex(tr).dropna()
                return x.mean()/x.std()*SQ if len(x) > 60 and x.std() > 0 else -9.0
            bi = max(range(len(grid)), key=lambda i: sh(series[i]))
            parts.append(series[bi].reindex(te))
            pparts.append(pnl_of(sig_fn(grid[bi]))[1].reindex(te))
            start += TEST
        if not parts:
            return None, None
        return pd.concat(parts).dropna(), pd.concat(pparts)

    pA, posA = wf(sig_A, GRID_A)
    pB, posB = wf(sig_B, GRID_B)
    if pA is None or pB is None:
        return None
    common = pA.index.intersection(pB.index)
    blend = 0.5*pA.loc[common] + 0.5*pB.loc[common]
    static = pnl_of(pd.Series(1.0, index=idx))[0].reindex(common).dropna()
    pos_b = (0.5*posA.reindex(common).fillna(0) + 0.5*posB.reindex(common).fillna(0))

    def m(p):
        p = p.dropna()
        if len(p) < 100 or p.std() == 0:
            return np.nan, np.nan, np.nan
        return p.mean()/p.std()*SQ, p.mean()/(p.std()/np.sqrt(len(p))), p.skew()

    shS, tS, skS = m(static); shX, tX, skX = m(blend)
    yx = pd.concat([blend, static], axis=1).dropna().values
    y, x = yx[:, 0], yx[:, 1]; X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X@b
    at = b[0]/np.sqrt((r@r/(len(y)-2))*np.linalg.inv(X.T@X)[0, 0])
    return dict(pair=name, n=len(common), start=str(common.min().date()), end=str(common.max().date()),
                static_sh=shS, static_skew=skS, blend_sh=shX, blend_t=tX, blend_skew=skX,
                alpha_t=at, beta=b[1],
                pct_short=100*(pos_b > 0).mean(), pct_long=100*(pos_b < 0).mean(),
                vol_level=vi.mean())

rows = []
print("UNIVERSALITY TEST — identical strategy on 5 new pairs (proxy instrument)\n")
print(f"{'pair':<17}{'OOS n':>6}{'volIdx':>7} | {'static Sh/skew':>15} | {'blend Sh (t)':>14}{'skew':>6}{'alpha-t':>8}{'beta':>6}{'%shrt':>6}{'%lng':>5}")
for name, uf, vf in PAIRS:
    res = run_pair(name, uf, vf)
    if res is None:
        print(f"{name:<17} insufficient data"); continue
    rows.append(res)
    print(f"{res['pair']:<17}{res['n']:>6}{res['vol_level']:>7.1f} | {res['static_sh']:>6.2f} /{res['static_skew']:>6.1f} | "
          f"{res['blend_sh']:>6.2f} ({res['blend_t']:>+4.1f}){res['blend_skew']:>6.1f}{res['alpha_t']:>8.1f}"
          f"{res['beta']:>6.2f}{res['pct_short']:>6.0f}{res['pct_long']:>5.0f}")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "p1_more_pairs.csv"), index=False)
print(f"\ncriterion (same as before): alpha-t >= 2 AND blend skew > static skew")
print(f"saved -> {os.path.join(OUT, 'p1_more_pairs.csv')}")
print("NOTE: identical grids in ABSOLUTE vol pts — on low-vol assets (EVZ) gates may rarely fire;")
print("breadth columns show this. No re-tuning anywhere; deflation unchanged (same 27 combos).")
