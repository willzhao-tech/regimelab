# -*- coding: utf-8 -*-
"""
AUDIT of p0_robust.py — independent recompute, NO import of volarb_harness.
Everything rebuilt from the written spec:
  park(w) = sqrt(rolling_w mean of ln(H/L)^2 / (4 ln 2)) * sqrt(252) * 100, shifted 1d
  A: richness = VXN - park21f; volTrend = park10f - park42f
     s=+1 if richness>=r_hi & volTrend<=-d ; s=-1 if richness<=r_lo & volTrend>=d
  B: range = ln(H/L)*100 ; be = VXN/sqrt(252); s=+1 if range<b1*be ; s=-1 if range>b2*be
  P&L: pos = s.shift(1); iv=(VXN.shift(1)/100)^2/252; rvar=ret^2
       cost = 2*VXN.shift(1)*0.5/1e4/252 * |dpos| ; pnl = pos*(iv-rvar) - cost
  WF: trailing train window, pick best train Sharpe (first-index tie-break), step=test.
Checks:
  1. all 9 cells recomputed independently
  2. truncate-BEFORE vs truncate-AFTER comparison for one cell (leak test)
  3. worst-cell teardown (1260/252 start 2013): per-block, per-year, concentration
"""
import numpy as np
import pandas as pd

DATA = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

nq_raw = pd.read_csv(DATA + r"\NQ_F_all_history.csv", parse_dates=["Date"]).set_index("Date").sort_index()
vx_raw = pd.read_csv(DATA + r"\VXN_all_history.csv", parse_dates=["Date"]).set_index("Date").sort_index()["Close"].dropna()
common = nq_raw.index.intersection(vx_raw.index)
NQ, VX = nq_raw.loc[common], vx_raw.loc[common]
print(f"aligned: {NQ.index[0].date()}..{NQ.index[-1].date()} n={len(NQ)}")

GRID_A = [(a, b, c) for a in (2.0, 4.0, 6.0) for b in (0.0, -2.0) for c in (0.0, 1.0, 2.0)]
GRID_B = [(b1, b2) for b1 in (0.8, 1.0, 1.2) for b2 in (1.3, 1.6, 2.0)]


def skew_adj(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    s = x.std(ddof=1)
    return n / ((n - 1.0) * (n - 2.0)) * np.sum(((x - m) / s) ** 3)


def build_pnls(d, v):
    """All 27 combo P&L series + static, on (possibly truncated) sample d/v."""
    ret = d["Close"].pct_change()
    lhl2 = np.log(d["High"] / d["Low"]) ** 2

    def park_f(w):
        return (np.sqrt(lhl2.rolling(w).mean() / (4 * np.log(2))) * SQ * 100).shift(1)

    rich = v - park_f(21)
    trend = park_f(10) - park_f(42)
    rng = np.log(d["High"] / d["Low"]) * 100.0
    be = v / SQ
    iv = (v.shift(1) / 100.0) ** 2 / 252.0
    rvar = ret ** 2
    edge = iv - rvar
    cunit = (2 * v.shift(1) * 0.5 / 1e4 / 252).fillna(0.0)

    def pnl_of(s):
        pos = s.shift(1).fillna(0.0)
        return (pos * edge - cunit * pos.diff().abs().fillna(0.0)).dropna()

    pA = []
    for rh, rl, dd in GRID_A:
        s = pd.Series(0.0, index=d.index)
        s[((rich >= rh) & (trend <= -dd)).fillna(False)] = 1.0
        s[((rich <= rl) & (trend >= dd)).fillna(False)] = -1.0
        pA.append(pnl_of(s))
    pB = []
    ok = rng.notna() & be.notna()
    for b1, b2 in GRID_B:
        s = pd.Series(0.0, index=d.index)
        s[ok & (rng < b1 * be)] = 1.0
        s[ok & (rng > b2 * be)] = -1.0
        pB.append(pnl_of(s))
    static = pnl_of(pd.Series(1.0, index=d.index))
    return pA, pB, static


def wf(pnls, grid, train, test):
    idx = pnls[0].index
    parts, picks = [], []
    st = train
    while st + test <= len(idx):
        trw, tew = idx[st - train:st], idx[st:st + test]
        best, bs = 0, -1e18
        for i, p in enumerate(pnls):
            x = p.reindex(trw).dropna()
            sh = x.mean() / x.std() * SQ if (len(x) > 60 and x.std() > 0) else -9.0
            if sh > bs:
                bs, best = sh, i
        picks.append(grid[best])
        parts.append(pnls[best].reindex(tew))
        st += test
    return (pd.concat(parts).dropna() if parts else pd.Series(dtype=float)), picks


def alpha_t(y_s, x_s):
    yx = pd.concat([y_s, x_s], axis=1).dropna().values
    y, x = yx[:, 0], yx[:, 1]
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return float(b[0] / np.sqrt((r @ r / (len(y) - 2)) * np.linalg.inv(X.T @ X)[0, 0]))


def run_cell(start, train, test):
    d = NQ if start is None else NQ.loc[NQ.index >= pd.Timestamp(start)]
    v = VX.reindex(d.index)
    pA, pB, static = build_pnls(d, v)
    oA, picksA = wf(pA, GRID_A, train, test)
    oB, picksB = wf(pB, GRID_B, train, test)
    cm = oA.index.intersection(oB.index)
    bl = 0.5 * oA.loc[cm] + 0.5 * oB.loc[cm]
    stt = static.reindex(cm).dropna()
    sh = bl.mean() / bl.std() * SQ
    shs = stt.mean() / stt.std() * SQ
    return dict(sharpe=sh, skew=skew_adj(bl), alpha_t=alpha_t(bl, stt),
                static_sh=shs, static_skew=skew_adj(stt),
                n=len(bl), nb=len(picksA),
                o0=str(cm.min().date()), o1=str(cm.max().date()),
                blend=bl, static=stt, picksA=picksA, picksB=picksB)


print("\n=== 1) INDEPENDENT RECOMPUTE, all 9 cells ===")
REPORTED = {  # (train,test,start) -> (sharpe, skew, alpha_t)
    (1000, 200, None): (1.71, 16.2, 11.0), (1000, 200, "2008-01-01"): (1.85, 12.2, 8.4),
    (1000, 200, "2013-01-01"): (1.72, 10.6, 6.0),
    (1260, 252, None): (1.73, 15.9, 11.0), (1260, 252, "2008-01-01"): (1.76, 11.6, 7.6),
    (1260, 252, "2013-01-01"): (1.71, 10.0, 5.7),
    (1500, 252, None): (1.73, 15.6, 10.8), (1500, 252, "2008-01-01"): (1.76, 11.2, 7.2),
    (1500, 252, "2013-01-01"): (1.92, 10.1, 6.1),
}
cells = {}
for (train, test) in [(1000, 200), (1260, 252), (1500, 252)]:
    for start in [None, "2008-01-01", "2013-01-01"]:
        r = run_cell(start, train, test)
        cells[(train, test, start)] = r
        rep = REPORTED[(train, test, start)]
        ok = (abs(r["sharpe"] - rep[0]) < 0.03 and abs(r["skew"] - rep[1]) < 0.3
              and abs(r["alpha_t"] - rep[2]) < 0.3)
        print(f"{train}/{test} start={str(start):<10} OOS {r['o0']}..{r['o1']} "
              f"({r['n']}d,{r['nb']}b)  Sh {r['sharpe']:.3f}  skew {r['skew']:+.2f}  "
              f"a-t {r['alpha_t']:+.2f}  static {r['static_sh']:.2f}/{r['static_skew']:+.1f}  "
              f"reported {rep}  {'MATCH' if ok else '** MISMATCH **'}")

print("\n=== 2) LEAK TEST: truncate-BEFORE (cold rebuild) vs truncate-AFTER (slice full-WF OOS) ===")
r_full = cells[(1260, 252, None)]
for start in ["2008-01-01", "2013-01-01"]:
    r_cold = cells[(1260, 252, start)]
    sliced = r_full["blend"].loc[r_cold["o0"]:r_cold["o1"]]
    shs = sliced.mean() / sliced.std() * SQ
    same_days = r_cold["blend"].index.equals(sliced.index)
    diff = (r_cold["blend"] - sliced.reindex(r_cold["blend"].index)).abs().max()
    print(f"start={start}: cold Sh {r_cold['sharpe']:.3f} vs full-WF-sliced Sh {shs:.3f} "
          f"(same index: {same_days}; max |pnl diff| {diff:.2e}) -> "
          f"{'series DIFFER => truncation genuinely re-runs the pipeline' if diff > 1e-12 or not same_days else 'IDENTICAL => suspicious'}")

print("\n=== 3) WORST-CELL TEARDOWN: 1260/252 start=2013 (lowest alpha-t 5.7) ===")
w = cells[(1260, 252, "2013-01-01")]
bl = w["blend"]
print("picks A per block:", w["picksA"])
print("picks B per block:", w["picksB"])
yr = bl.groupby(bl.index.year)
print("\nper-year: n, Sharpe, sum")
for y, g in yr:
    print(f"  {y}: n={len(g):>3}  Sh {g.mean()/g.std()*SQ if g.std()>0 else float('nan'):+6.2f}  sum {g.sum():+9.6f}")
blocks = np.array_split(np.arange(len(bl)), max(1, len(bl) // 252))
print("\nper-block Sharpe:", [round(bl.iloc[b].mean() / bl.iloc[b].std() * SQ, 2) for b in blocks])
tot = bl.sum()
srt = bl.sort_values()
top5 = srt.iloc[-5:]
bot5 = srt.iloc[:5]
print(f"\ntotal P&L {tot:.6f}; top-5 days {top5.sum():.6f} ({100*top5.sum()/tot:.0f}% of total); "
      f"worst-5 days {bot5.sum():.6f}")
print("top-5 days:", [(str(i.date()), round(x, 6)) for i, x in top5.items()])
print("worst-5 days:", [(str(i.date()), round(x, 6)) for i, x in bot5.items()])
ex = bl.drop(top5.index)  # DIAGNOSTIC ONLY (tail-deletion forbidden for results)
print(f"DIAGNOSTIC ex-top5: Sh {ex.mean()/ex.std()*SQ:.2f}  skew {skew_adj(ex):+.1f}")
ex1 = bl.drop(srt.index[-1:])
print(f"DIAGNOSTIC ex-top1: Sh {ex1.mean()/ex1.std()*SQ:.2f}  skew {skew_adj(ex1):+.1f}")
# position occupancy: rebuild winning signal exposure share
nz = (bl != 0).mean()
print(f"share of OOS days with nonzero P&L: {nz:.2%}")
# COVID dependence: drop 2020-02-15..2020-04-30 (diagnostic)
mask = ~((bl.index >= "2020-02-15") & (bl.index <= "2020-04-30"))
exc = bl[mask]
print(f"DIAGNOSTIC ex-COVID(2020-02-15..04-30): Sh {exc.mean()/exc.std()*SQ:.2f}  skew {skew_adj(exc):+.1f}")
mask24 = ~((bl.index >= "2024-07-25") & (bl.index <= "2024-08-15"))
exc24 = bl[mask24]
print(f"DIAGNOSTIC ex-Aug2024 spike: Sh {exc24.mean()/exc24.std()*SQ:.2f}  skew {skew_adj(exc24):+.1f}")

print("\n=== 4) second-worst: 1000/200 start=2013 teardown (Sh 1.72, a-t 6.0) ===")
w2 = cells[(1000, 200, "2013-01-01")]
bl2 = w2["blend"]
srt2 = bl2.sort_values()
ex2 = bl2.drop(srt2.index[-5:])
print(f"total {bl2.sum():.6f}, top-5 share {100*srt2.iloc[-5:].sum()/bl2.sum():.0f}%, "
      f"DIAGNOSTIC ex-top5 Sh {ex2.mean()/ex2.std()*SQ:.2f} skew {skew_adj(ex2):+.1f}")
