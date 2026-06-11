# -*- coding: utf-8 -*-
"""p0_audit_costs.py - quantify cost drag vs leg pnl in the rolled run (audit follow-up)."""
import numpy as np
import pandas as pd
import os

DATA = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)
nq = pd.read_csv(os.path.join(DATA, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
vxn = pd.read_csv(os.path.join(DATA, "VXN_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna().sort_index()
ret = nq["Close"].pct_change()
idx = ret.index.intersection(vxn.index)
nq, ret, vxn = nq.loc[idx], ret.loc[idx], vxn.loc[idx]
V = (vxn / 100.0) ** 2
rv1 = 252.0 * ret ** 2
GAMMA = (1.0 / 21.0) * (V.shift(1) - rv1)
VEGA = (20.0 / 21.0) * (V.shift(1) - V)
PNL_SHORT = GAMMA + VEGA
SPREAD = 2.0 * (vxn.shift(1) / 100.0) * 0.005


def park(w):
    return (np.sqrt((np.log(nq["High"] / nq["Low"]) ** 2).rolling(w).mean() / (4 * np.log(2))) * SQ * 100).shift(1)


richness, trend = vxn - park(21), park(10) - park(42)
rng, be = np.log(nq["High"] / nq["Low"]) * 100.0, vxn / SQ


def sig_A(g):
    r_hi, r_lo, d = g
    s = pd.Series(0.0, index=idx)
    s[((richness >= r_hi) & (trend <= -d)).fillna(False)] = 1.0
    s[((richness <= r_lo) & (trend >= d)).fillna(False)] = -1.0
    return s


def sig_B(g):
    b1, b2 = g
    s = pd.Series(0.0, index=idx)
    ok = rng.notna() & be.notna()
    s[ok & (rng < b1 * be)] = 1.0
    s[ok & (rng > b2 * be)] = -1.0
    return s


def bt(s):
    pos = s.clip(-1, 1).shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(0.0)
    cost = (SPREAD * turn).fillna(0.0)
    return (pos * PNL_SHORT - cost).dropna(), pos, turn, cost


GRID_A = [(a, b, c) for a in (2.0, 4.0, 6.0) for b in (0.0, -2.0) for c in (0.0, 1.0, 2.0)]
GRID_B = [(b1, b2) for b1 in (0.8, 1.0, 1.2) for b2 in (1.3, 1.6, 2.0)]


def wf(sig_fn, grid, train=1260, test=252):
    runs = {g: bt(sig_fn(g)) for g in grid}
    pidx = runs[grid[0]][0].index
    parts, pos_p, turn_p, cost_p = [], [], [], []
    start = train
    while start + test <= len(pidx):
        tr, te = pidx[start - train:start], pidx[start:start + test]
        best = max(grid, key=lambda g: (lambda x: x.mean() / x.std() * SQ if len(x) > 60 and x.std() > 0 else -9.0)(runs[g][0].reindex(tr).dropna()))
        parts.append(runs[best][0].reindex(te)); pos_p.append(runs[best][1].reindex(te))
        turn_p.append(runs[best][2].reindex(te)); cost_p.append(runs[best][3].reindex(te))
        start += test
    return (pd.concat(parts).dropna(), pd.concat(pos_p), pd.concat(turn_p), pd.concat(cost_p))


for nm, fn, grid in [("A", sig_A, GRID_A), ("B", sig_B, GRID_B)]:
    pnl, pos, turn, cost = wf(fn, grid)
    cm = pnl.index
    pre = (pos * PNL_SHORT).reindex(cm)
    print("%s: net ann mean %+.5f (sharpe %+.2f) | pre-cost ann mean %+.5f (sharpe %+.2f) | cost drag ann %+.5f"
          % (nm, pnl.mean() * 252, pnl.mean() / pnl.std() * SQ, pre.mean() * 252, pre.mean() / pre.std() * SQ,
             cost.reindex(cm).mean() * 252))
    print("   turnover/day %.3f, mean spread per unit %.5f (ann var units), cost as %% of pre-cost gross |pnl|: drag/premean = %.0f%%"
          % (turn.reindex(cm).mean(), SPREAD.reindex(cm).mean(), 100 * cost.reindex(cm).mean() / abs(pre.mean()) if pre.mean() != 0 else float("nan")))

# static for reference
s_pnl, s_pos, s_turn, s_cost = bt(pd.Series(1.0, index=idx))
cm2 = s_pnl.index
print("static: net ann mean %+.5f, cost ann %+.6f (one-time entry only)" % (s_pnl.mean() * 252, s_cost.reindex(cm2).mean() * 252))

# what would B be at HALF the spread (0.25 volpt) and at zero cost?
for mult, lab in [(0.5, "half spread 0.25vp"), (0.0, "zero cost")]:
    SP2 = SPREAD * mult
    def bt2(s):
        pos = s.clip(-1, 1).shift(1).fillna(0.0)
        cost = (SP2 * pos.diff().abs().fillna(0.0)).fillna(0.0)
        return (pos * PNL_SHORT - cost).dropna()
    runs = {g: bt2(sig_B(g)) for g in GRID_B}
    pidx = runs[GRID_B[0]].index
    parts, start = [], 1260
    while start + 252 <= len(pidx):
        tr, te = pidx[start - 1260:start], pidx[start:start + 252]
        best = max(GRID_B, key=lambda g: (lambda x: x.mean() / x.std() * SQ if len(x) > 60 and x.std() > 0 else -9.0)(runs[g].reindex(tr).dropna()))
        parts.append(runs[best].reindex(te)); start += 252
    p = pd.concat(parts).dropna()
    print("B walk-forward at %s: sharpe %+.2f (t %+.2f)" % (lab, p.mean() / p.std() * SQ, p.mean() / (p.std() / np.sqrt(len(p)))))
