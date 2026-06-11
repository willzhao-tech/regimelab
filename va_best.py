# -*- coding: utf-8 -*-
"""
va_best.py -- combination of the ONLY two families that survived audit:
  A) range_forecast : s=+1 if range_t < b1*be_t, s=-1 if range_t > b2*be_t (be = VXN/sqrt252)
  B) regime_combo   : s=+1 if rich & vol-falling, s=-1 if cheap & vol-rising

Combination rule (declared a priori, NOT optimized): fixed 50/50 average of the
two family positions, s_best = 0.5*s_A + 0.5*s_B.  Each family's parameters are
picked INDEPENDENTLY by the same trailing-train walk-forward (1260d train ->
252d test) used in the audited single-family runs.  No joint selection, no new
free parameters, no caps, no winsorizing, harness P&L untouched.

Outputs: blend OOS metrics, alpha vs static short-vol (OLS + Newey-West 10),
tail table, cost sweep at 0.5 / 1.0 / 2.0 vol-pts with FULL re-selection,
family-stream correlation, yearly-block consistency.
"""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
import volarb_harness as H

SQ = np.sqrt(252.0)
df, ret, vxn = H.load()

# ---------- family A: range_forecast ----------
rng = np.log(df["High"] / df["Low"]) * 100.0
be = vxn / SQ

def sig_A(g):
    b1, b2 = g
    s = pd.Series(0.0, index=ret.index)
    ok = rng.notna() & be.notna()
    s[ok & (rng < b1 * be)] = 1.0
    s[ok & (rng > b2 * be)] = -1.0
    return s

GRID_A = [(b1, b2) for b1 in (0.8, 1.0, 1.2) for b2 in (1.3, 1.6, 2.0)]

# ---------- family B: regime_combo ----------
fc21 = H.fcast_vol(df, ret, "park21")
fc10 = H.fcast_vol(df, ret, "park10")
fc42 = H.fcast_vol(df, ret, "park42")
richness = vxn - fc21
trend = fc10 - fc42

def sig_B(g):
    r_hi, r_lo, d = g
    s = pd.Series(0.0, index=ret.index)
    s[((richness >= r_hi) & (trend <= -d)).fillna(False)] = 1.0
    s[((richness <= r_lo) & (trend >= d)).fillna(False)] = -1.0
    return s

GRID_B = [(a, b, c) for a in (2.0, 4.0, 6.0) for b in (0.0, -2.0) for c in (0.0, 1.0, 2.0)]

TRAIN, TEST = 1260, 252

def tr_sharpe(s, tr):
    x = s.reindex(tr).dropna()
    return (x.mean() / x.std() * SQ) if len(x) > 60 and x.std() > 0 else -9.0

def run_blend(cost):
    """Per-family trailing-train selection, then fixed 50/50 position blend."""
    pnls_A = [H.backtest(sig_A(g), ret, vxn, cost) for g in GRID_A]
    pnls_B = [H.backtest(sig_B(g), ret, vxn, cost) for g in GRID_B]
    idx = pnls_A[0].index
    assert idx.equals(pnls_B[0].index)
    parts, parts_A, parts_B, pos_parts = [], [], [], []
    start = TRAIN
    while start + TEST <= len(idx):
        tr = idx[start - TRAIN:start]
        te = idx[start:start + TEST]
        ai = max(range(len(GRID_A)), key=lambda i: tr_sharpe(pnls_A[i], tr))
        bi = max(range(len(GRID_B)), key=lambda i: tr_sharpe(pnls_B[i], tr))
        s_blend = 0.5 * sig_A(GRID_A[ai]) + 0.5 * sig_B(GRID_B[bi])
        pnl_blend = H.backtest(s_blend, ret, vxn, cost)
        parts.append(pnl_blend.reindex(te))
        parts_A.append(pnls_A[ai].reindex(te))
        parts_B.append(pnls_B[bi].reindex(te))
        pos_parts.append(s_blend.clip(-1, 1).shift(1).reindex(te))
        start += TEST
    oos = pd.concat(parts).dropna()
    oA = pd.concat(parts_A).dropna()
    oB = pd.concat(parts_B).dropna()
    pos = pd.concat(pos_parts).reindex(oos.index)
    return oos, oA, oB, pos

def nw_t_mean(p, L=10):
    p = np.asarray(p, dtype=float)
    u = p - p.mean()
    n = len(p)
    S = float(u @ u)
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        S += 2 * w * float(u[l:] @ u[:-l])
    return p.mean() / np.sqrt(S / n / n)

def reg_alpha(y, x, L=10):
    """y = a + b x ; returns alpha, beta, t_alpha OLS, t_alpha NW(L)."""
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ b
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    s2 = float(u @ u) / (n - k)
    t_ols = b[0] / np.sqrt(s2 * XtXi[0, 0])
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    t_nw = b[0] / np.sqrt(V[0, 0])
    return b[0], b[1], t_ols, t_nw

# ---------- main run at default cost 0.5 vp ----------
oos, oA, oB, pos = run_blend(0.5)
static = H.backtest(pd.Series(1.0, index=ret.index), ret, vxn, 0.5)
base = static.reindex(oos.index).dropna()

m = H.metrics(oos)
mb = H.metrics(base)
print("=== va_best (50/50 range_forecast + regime_combo), walk-forward OOS ===")
print("OOS span: %s .. %s  n=%d" % (oos.index[0].date(), oos.index[-1].date(), len(oos)))
for k, v in m.items():
    print("  %-12s %s" % (k, v))
print("NW(10) t of mean: %.2f" % nw_t_mean(oos))
print("\n--- static s=+1 on SAME dates ---")
for k, v in mb.items():
    print("  %-12s %s" % (k, v))

# sub-family reproduction sanity
print("\nfamily A (range_forecast) standalone OOS Sharpe: %.4f" % H.metrics(oA)["sharpe"])
print("family B (regime_combo)  standalone OOS Sharpe: %.4f" % H.metrics(oB)["sharpe"])
print("corr(daily pnl A, B) on OOS: %.3f" % oA.corr(oB))

# positions
pl = float((pos < 0).mean() * 100); ps = float((pos > 0).mean() * 100); pf = float((pos == 0).mean() * 100)
print("\nOOS position: short %.1f%% | long %.1f%% | flat %.1f%% | mean %.3f" % (ps, pl, pf, pos.mean()))

# alpha vs static
yx = pd.concat([oos, base], axis=1, keys=["y", "x"]).dropna()
a, bcoef, t_ols, t_nw = reg_alpha(yx["y"].values, yx["x"].values)
print("\nTiming regression oos = a + b*static (same dates):")
print("  alpha=%.3e/day  beta=%.3f  t(alpha) OLS=%.2f  NW(10)=%.2f" % (a, bcoef, t_ols, t_nw))
print("  ann. alpha (var-units/yr): %.4f" % (a * 252))

# tail table
w = oos.nsmallest(5); bst = oos.nlargest(3)
mu = oos.mean()
print("\nWorst 5 OOS days (x mean daily pnl):")
for d, v in w.items():
    print("  %s  %.4e  (%.0fx)  pos=%.2f  NQ ret=%+.2f%%" % (d.date(), v, v / mu, pos.loc[d], ret.loc[d] * 100))
print("Best 3 OOS days:")
for d, v in bst.items():
    print("  %s  %.4e  (%.0fx)  pos=%.2f  NQ ret=%+.2f%%" % (d.date(), v, v / mu, pos.loc[d], ret.loc[d] * 100))
print("static worst day on same dates: %.4e (%.0fx its mean)" % (base.min(), base.min() / base.mean()))

# crisis windows
for lab, a0, a1 in [("2008H2", "2008-07-01", "2008-12-31"), ("2020Q1", "2020-01-01", "2020-04-30"),
                    ("2022", "2022-01-01", "2022-12-31")]:
    s_b = oos.loc[a0:a1].sum(); s_s = base.loc[a0:a1].sum()
    print("%s: blend %+0.4f vs static %+0.4f" % (lab, s_b, s_s))

# robustness: ex-best-days Sharpe (tail-fluke check, NOT used for the headline)
for ntop in (3, 10, 20):
    ex = oos.drop(oos.nlargest(ntop).index)
    print("Sharpe ex-top-%d winners: %.3f" % (ntop, ex.mean() / ex.std() * SQ))

# yearly blocks
blocks = [oos.iloc[i:i + TEST] for i in range(0, len(oos), TEST)]
bs = [b.mean() / b.std() * SQ for b in blocks if len(b) > 60]
print("\nYearly OOS blocks: %d | positive Sharpe: %d | median %.2f | min %.2f" %
      (len(bs), sum(1 for s in bs if s > 0), np.median(bs), min(bs)))
beats = sum(1 for b in blocks if len(b) > 60 and b.sum() > base.reindex(b.index).sum())
print("blocks where blend total pnl beats static: %d/%d" % (beats, len(bs)))

# ---------- cost sweep with FULL re-selection ----------
print("\n=== cost sweep (selection re-run at each cost) ===")
for cost in (1.0, 2.0):
    o2, _, _, _ = run_blend(cost)
    m2 = H.metrics(o2)
    b2 = H.backtest(pd.Series(1.0, index=ret.index), ret, vxn, cost).reindex(o2.index).dropna()
    yx2 = pd.concat([o2, b2], axis=1, keys=["y", "x"]).dropna()
    a2, bb2, to2, tn2 = reg_alpha(yx2["y"].values, yx2["x"].values)
    print("cost %.1f vp: Sharpe %.4f  t %.2f  skew %.2f  | alpha t OLS %.2f NW %.2f" %
          (cost, m2["sharpe"], m2["tstat"], m2["skew"], to2, tn2))

# turnover
to = pos.diff().abs().sum() / (len(pos) / 252.0)
print("\nblend turnover: %.1f unit-trades/yr" % to)
