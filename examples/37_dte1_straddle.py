"""
Example 37 — the max-gamma / min-vega instrument: 1-DTE delta-hedged ATM straddle.

P0-1 showed the vol-arb signals predict next-day REALIZED variance but know nothing about
implied-vol changes — so they die on vega-dominated variance swaps. The instrument that
isolates exactly what they predict: sell (or buy) a 1-day ATM straddle at close t, expire
at close t+1. Held to expiry => NO vega mark-to-market. Black-Scholes, zero rates:

  premium_frac(t) = 2*(2*N(0.5*sigma_1d*sqrt(dt)) - 1)        (ATM straddle / spot)
  short-side pnl_(t+1) = premium_frac(t) - |ret_(t+1)|         (delta-hedged at inception;
                                                                ATM straddle delta ~ 0)
  sigma_1d = k * VXN_t / 100   (k = 1-DTE IV vs 30-day index — UNKNOWN without real
             options data; swept over {0.85, 1.00}. Both strategy AND static share k,
             so the timing-alpha test is robust to k even though the carry level is not.)

Costs: a FRESH straddle is traded every in-market day (unlike the swap, you pay every day):
  cost_t = spread_pct * premium_frac * |pos_t|,  spread swept over {1%, 2.5%, 5%}.

Same signals/grids/walk-forward as the validated families. Compare vs static short-straddle.
Run:  python examples/37_dte1_straddle.py
"""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
from math import erf, sqrt

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)
DT = 1.0/252.0
TRAIN, TEST = 1260, 252

df = pd.read_csv(os.path.join(OUT, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
vxn = pd.read_csv(os.path.join(OUT, "VXN_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
ret = df["Close"].pct_change()
idx = ret.index.intersection(vxn.index)
df, ret, vxn = df.loc[idx], ret.loc[idx], vxn.loc[idx]

N = lambda x: 0.5*(1.0 + np.vectorize(erf)(x/np.sqrt(2.0)))

def straddle_premium_frac(k):
    sig = k * vxn / 100.0
    return pd.Series(2.0*(2.0*N(0.5*sig*np.sqrt(DT)) - 1.0), index=idx)

# ---- signals (identical to the validated families) ----
park = lambda w: (np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(w).mean()/(4*np.log(2)))*SQ*100).shift(1)
fc21, fc10, fc42 = park(21), park(10), park(42)
rich = vxn - fc21; trend = fc10 - fc42
rng = np.log(df["High"]/df["Low"])*100.0; be = vxn/SQ

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

GRID_A = [(a, b, c) for a in (2.0, 4.0, 6.0) for b in (0.0, -2.0) for c in (0.0, 1.0, 2.0)]
GRID_B = [(b1, b2) for b1 in (0.8, 1.0, 1.2) for b2 in (1.3, 1.6, 2.0)]

def pnl_of(s, prem, spread_pct):
    pos = s.clip(-1, 1).shift(1).fillna(0.0)                 # decide close t, trade straddle expiring t+1
    short_pnl = prem.shift(1) - ret.abs()                     # premium struck at t, settled t+1
    cost = spread_pct * prem.shift(1) * pos.abs()             # pay spread EVERY day in market
    return (pos*short_pnl - cost).dropna()

def walk_forward(sig_fn, grid, prem, spread_pct):
    series = [pnl_of(sig_fn(g), prem, spread_pct) for g in grid]
    bidx = series[0].index; parts = []; picks = []; start = TRAIN
    while start + TEST <= len(bidx):
        tr = bidx[start-TRAIN:start]; te = bidx[start:start+TEST]
        def sh(p):
            x = p.reindex(tr).dropna()
            return x.mean()/x.std()*SQ if len(x) > 60 and x.std() > 0 else -9.0
        bi = max(range(len(grid)), key=lambda i: sh(series[i]))
        picks.append(grid[bi]); parts.append(series[bi].reindex(te))
        start += TEST
    out = pd.concat(parts).dropna(); out.attrs["picks"] = picks
    return out

def metrics(p):
    p = p.dropna()
    sh = p.mean()/p.std()*SQ; t = p.mean()/(p.std()/np.sqrt(len(p)))
    return sh, t, p.skew(), abs(p.min())/abs(p.mean()) if p.mean() != 0 else np.inf

def alpha_vs(p, base):
    yx = pd.concat([p, base], axis=1).dropna().values
    y, x = yx[:, 0], yx[:, 1]
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X@b
    return b[0]/np.sqrt((r@r/(len(y)-2))*np.linalg.inv(X.T@X)[0, 0]), b[1]

print("1-DTE DELTA-HEDGED STRADDLE (held to expiry: pure gamma, ZERO vega)")
print(f"data {idx.min().date()}..{idx.max().date()}  |  E[premium k=1] {straddle_premium_frac(1.0).mean()*1e4:.0f} bps/day  vs  E|ret| {ret.abs().mean()*1e4:.0f} bps/day")
print(f"\n{'k':>5}{'spread':>8} | {'static':>14} | {'A':>16} | {'B':>16} | {'blend':>16}{'alpha-t(blend)':>15}")
rows = []
for k in (0.85, 1.00):
    prem = straddle_premium_frac(k)
    for sp in (0.01, 0.025, 0.05):
        stat = pnl_of(pd.Series(1.0, index=idx), prem, sp)
        pA = walk_forward(sig_A, GRID_A, prem, sp)
        pB = walk_forward(sig_B, GRID_B, prem, sp)
        common = pA.index.intersection(pB.index)
        blend = 0.5*pA.loc[common] + 0.5*pB.loc[common]
        statc = stat.reindex(common).dropna()
        shS, tS, skS, _ = metrics(statc)
        shA, tA, skA, _ = metrics(pA.loc[common]); shB, tB, skB, _ = metrics(pB.loc[common])
        shX, tX, skX, _ = metrics(blend)
        at, beta = alpha_vs(blend, statc)
        rows.append(dict(k=k, spread=sp, static=shS, A=shA, B=shB, blend=shX, alpha_t=at, skew_blend=skX, skew_static=skS))
        print(f"{k:>5.2f}{sp*100:>7.1f}% | Sh {shS:>5.2f} sk {skS:>4.1f} | Sh {shA:>5.2f} sk {skA:>4.1f} | "
              f"Sh {shB:>5.2f} sk {skB:>4.1f} | Sh {shX:>5.2f} sk {skX:>4.1f}{at:>12.1f}")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "p0_dte1_results.csv"), index=False)
print(f"\nsaved -> {os.path.join(OUT, 'p0_dte1_results.csv')}")
print("""
READ: the carry LEVEL depends on k (1-DTE IV vs VXN — unknowable without real options data).
The robust question is the TIMING test: blend alpha-t vs static on the SAME instrument, and
whether the skew inversion reappears. If alpha survives here (it died on the rolled swap),
the gamma instrument is the right home for the signal — pending real 0-1DTE quotes.""")
