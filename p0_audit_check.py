# -*- coding: utf-8 -*-
"""
p0_audit_check.py - ADVERSARIAL AUDIT of p0_instrument.py.
Fully independent reimplementation: no import of volarb_harness or p0_instrument.
Recomputes rolled-21d walk-forward OOS for A, B, blend, static; decomposes pnl
into gamma/carry leg vs vega MTM leg; tests VXN-momentum-disguise hypothesis.
"""
import numpy as np
import pandas as pd
import os

DATA = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

# ---------------- independent data load ----------------
nq = pd.read_csv(os.path.join(DATA, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
vxn = pd.read_csv(os.path.join(DATA, "VXN_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna().sort_index()
ret = nq["Close"].pct_change()
idx = ret.index.intersection(vxn.index)
nq, ret, vxn = nq.loc[idx], ret.loc[idx], vxn.loc[idx]
print("rows after intersect:", len(idx), idx[0].date(), "->", idx[-1].date())

# ---------------- instrument ----------------
V = (vxn / 100.0) ** 2
rv1 = 252.0 * ret ** 2
GAMMA = (1.0 / 21.0) * (V.shift(1) - rv1)       # carry/gamma leg (short side)
VEGA = (20.0 / 21.0) * (V.shift(1) - V)         # MTM vega leg (short side)
PNL_SHORT = GAMMA + VEGA
SPREAD = 2.0 * (vxn.shift(1) / 100.0) * 0.005


def bt_rolled(s):
    pos = s.clip(-1, 1).shift(1).fillna(0.0)
    cost = (SPREAD * pos.diff().abs().fillna(0.0)).fillna(0.0)
    return (pos * PNL_SHORT - cost).dropna(), pos


# ---------------- independent signals ----------------
def park(w):
    return (np.sqrt((np.log(nq["High"] / nq["Low"]) ** 2).rolling(w).mean() / (4 * np.log(2))) * SQ * 100).shift(1)


richness = vxn - park(21)
trend = park(10) - park(42)


def sig_A(g):
    r_hi, r_lo, d = g
    s = pd.Series(0.0, index=idx)
    s[((richness >= r_hi) & (trend <= -d)).fillna(False)] = 1.0
    s[((richness <= r_lo) & (trend >= d)).fillna(False)] = -1.0
    return s


rng = np.log(nq["High"] / nq["Low"]) * 100.0
be = vxn / SQ


def sig_B(g):
    b1, b2 = g
    s = pd.Series(0.0, index=idx)
    ok = rng.notna() & be.notna()
    s[ok & (rng < b1 * be)] = 1.0
    s[ok & (rng > b2 * be)] = -1.0
    return s


GRID_A = [(a, b, c) for a in (2.0, 4.0, 6.0) for b in (0.0, -2.0) for c in (0.0, 1.0, 2.0)]
GRID_B = [(b1, b2) for b1 in (0.8, 1.0, 1.2) for b2 in (1.3, 1.6, 2.0)]


# ---------------- independent walk-forward ----------------
def walk_forward(sig_fn, grid, train=1260, test=252):
    runs = {g: bt_rolled(sig_fn(g)) for g in grid}
    pidx = runs[grid[0]][0].index
    parts, posparts, picks = [], [], []
    start = train
    while start + test <= len(pidx):
        tr = pidx[start - train:start]
        te = pidx[start:start + test]
        best, bsh = None, -1e9
        for g in grid:
            x = runs[g][0].reindex(tr).dropna()
            sh = x.mean() / x.std() * SQ if len(x) > 60 and x.std() > 0 else -9.0
            if sh > bsh:
                bsh, best = sh, g
        picks.append(best)
        parts.append(runs[best][0].reindex(te))
        posparts.append(runs[best][1].reindex(te))
        start += test
    return pd.concat(parts).dropna(), pd.concat(posparts), picks


def met(p):
    p = p.dropna()
    return (p.mean() / p.std() * SQ, p.mean() / (p.std() / np.sqrt(len(p))), p.skew(), len(p))


def nw_alpha(y, x, lag=5):
    z = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    n = len(z)
    X = np.column_stack([np.ones(n), z["x"].values])
    yv = z["y"].values
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    e = yv - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    t_ols = b[0] / np.sqrt(((e @ e) / (n - 2) * XtXi)[0, 0])
    u = X * e[:, None]
    S = u.T @ u
    for L in range(1, lag + 1):
        S += (1 - L / (lag + 1)) * (u[L:].T @ u[:-L] + u[:-L].T @ u[L:])
    cnw = XtXi @ S @ XtXi
    return b[0], t_ols, b[0] / np.sqrt(cnw[0, 0]), b[1]


oosA, posA, picksA = walk_forward(sig_A, GRID_A)
oosB, posB, picksB = walk_forward(sig_B, GRID_B)
common = oosA.index.intersection(oosB.index)
blend = 0.5 * oosA.reindex(common) + 0.5 * oosB.reindex(common)
static, _ = bt_rolled(pd.Series(1.0, index=idx))
static_oos = static.reindex(common)

print("\n=== INDEPENDENT recomputation, rolled 21d, OOS %s..%s n=%d ===" % (common[0].date(), common[-1].date(), len(common)))
for nm, p in [("A", oosA), ("B", oosB), ("blend", blend), ("static", static_oos)]:
    sh, t, sk, n = met(p)
    line = "%-7s sharpe %+.3f  t %+.2f  skew %+.2f  n %d" % (nm, sh, t, sk, n)
    if nm != "static":
        a, to, tn, bb = nw_alpha(p, static.reindex(p.index))
        line += "   alpha-t OLS %+.2f NW5 %+.2f beta %+.3f" % (to, tn, bb)
    print(line)

from collections import Counter
print("\npicks A:", Counter(picksA).most_common(3))
print("picks B:", Counter(picksB).most_common(3))

# ---------------- decomposition: gamma leg vs vega leg ----------------
print("\n=== pnl decomposition on OOS days (pre-cost, position-weighted) ===")
for nm, pos in [("A", posA), ("B", posB), ("static", pd.Series(1.0, index=common))]:
    pos = pos.reindex(common).fillna(0.0)
    pg = (pos * GAMMA.reindex(common)).dropna()
    pv = (pos * VEGA.reindex(common)).dropna()
    tot = (pg + pv)
    print("%-7s gamma-leg: ann mean %+.5f sharpe %+.2f | vega-leg: ann mean %+.5f sharpe %+.2f | total pre-cost sharpe %+.2f"
          % (nm, pg.mean() * 252, pg.mean() / pg.std() * SQ, pv.mean() * 252, pv.mean() / pv.std() * SQ,
             tot.mean() / tot.std() * SQ))
    print("        share of total pre-cost pnl: gamma %.0f%%  vega %.0f%%"
          % (100 * pg.sum() / tot.sum(), 100 * pv.sum() / tot.sum()))

# ---------------- VXN-momentum disguise test ----------------
print("\n=== VXN-momentum / short-VXN-change disguise tests (OOS days) ===")
naive_static_vega = (V.shift(1) - V).reindex(common)              # always-short-VXN-change pnl
dV = V.diff()
mom_pos = np.sign(-dV).shift(1).fillna(0.0)                       # short vol after VXN rose? no: -sign(dV)= -1 after rise
mom_pnl = (mom_pos * (V.shift(1) - V)).reindex(common)            # VXN-momentum strategy (yesterday's dV continues)
rev_pnl = (-mom_pos * (V.shift(1) - V)).reindex(common)           # VXN-reversal strategy
for nm, p in [("A", oosA.reindex(common)), ("B", oosB.reindex(common)), ("blend", blend)]:
    print("%-7s corr w/ always-short-dVXN %+.3f | corr w/ VXN-momentum %+.3f | corr w/ VXN-reversal %+.3f"
          % (nm, p.corr(naive_static_vega), p.corr(mom_pnl), p.corr(rev_pnl)))
print("ref sharpe: always-short-dVXN %+.2f, VXN-momentum %+.2f, VXN-reversal %+.2f (pre-cost, OOS days)"
      % (naive_static_vega.mean() / naive_static_vega.std() * SQ,
         mom_pnl.mean() / mom_pnl.std() * SQ, rev_pnl.mean() / rev_pnl.std() * SQ))

# conditional: what does A earn when in market, split by leg sign of position
inA = posA.reindex(common).fillna(0.0)
print("\nA position days: short=%d long=%d flat=%d" % ((inA > 0).sum(), (inA < 0).sum(), (inA == 0).sum()))
shrt = oosA.reindex(common)[inA > 0]
lng = oosA.reindex(common)[inA < 0]
for nm, p in [("A short-vol days", shrt), ("A long-vol days", lng)]:
    if len(p) > 60:
        print("  %-18s n=%d  ann mean %+.5f  sharpe %+.2f  skew %+.2f" % (nm, len(p), p.mean() * 252, p.mean() / p.std() * SQ, p.skew()))
inB = posB.reindex(common).fillna(0.0)
print("B position days: short=%d long=%d flat=%d" % ((inB > 0).sum(), (inB < 0).sum(), (inB == 0).sum()))
shrtB = oosB.reindex(common)[inB > 0]
lngB = oosB.reindex(common)[inB < 0]
for nm, p in [("B short-vol days", shrtB), ("B long-vol days", lngB)]:
    if len(p) > 60:
        print("  %-18s n=%d  ann mean %+.5f  sharpe %+.2f  skew %+.2f" % (nm, len(p), p.mean() * 252, p.mean() / p.std() * SQ, p.skew()))

# B long-vol days decomposed by leg
posBc = inB
pgB = (posBc * GAMMA.reindex(common))[inB < 0]
pvB = (posBc * VEGA.reindex(common))[inB < 0]
print("  B long-vol days legs: gamma ann mean %+.5f, vega ann mean %+.5f" % (pgB.mean() * 252, pvB.mean() * 252))
pgBs = (posBc * GAMMA.reindex(common))[inB > 0]
pvBs = (posBc * VEGA.reindex(common))[inB > 0]
print("  B short-vol days legs: gamma ann mean %+.5f, vega ann mean %+.5f" % (pgBs.mean() * 252, pvBs.mean() * 252))
