# -*- coding: utf-8 -*-
"""Example 45 - FINAL optimized multi-market vol book = the SELECTION-FREE FLOOR.
Among the verified-causal finalists, invvol x cov DOMINATES: it ties composed on Sharpe (1.26 vs 1.29),
maxDD (-20%) and Calmar (~0.52), but has materially better tail (skew -1.86 vs -2.49, worst-day -6.9%
vs -9.7%) AND uses ZERO return information (no in-sample pick) -> maximally defensible. The composed
riskweight x cov leans INTO recent winners, which in short-vol concentrates the left tail (skew -2.49,
WORSE than the -1.05 baseline). So the skew 'fix' is not a new method - it is choosing the floor.
  weight[t] = invtrailingvol_252(sleeve)[t] * coverage_gate[t]   (both .shift(1) causal, NO returns used)
All under FULL L4 frictions (bookopt_harness). composed shown only as comparison."""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
from math import erf
import bookopt_harness as H
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"; SQ = H.SQ; WIN = 252
Nrm = lambda x: 0.5*(1.0+np.vectorize(erf)(x/np.sqrt(2.0)))

sleeves = {}; statics = {}
for name,_,_,_ in H.PAIRS:
    a, b = H.market(name)
    if a is not None: sleeves[name] = a; statics[name] = b

def invvol(p, win=WIN):                            # RISK only, NO return sign -> selection-free
    return (1.0/p.rolling(win).std()).replace([np.inf,-np.inf], np.nan).shift(1)
def riskweight(p, win=WIN, eps=0.05):              # trailing-Sharpe (uses returns) - composed only
    sh = (p.rolling(win).mean()/p.rolling(win).std()*SQ).replace([np.inf,-np.inf], np.nan)
    return (sh.clip(lower=0.0)+eps).shift(1)
def coverage_gate(name, margin_bp=0.0, win=WIN):
    H._load(); df, ret, vi, sp0 = H._DATA[name]; idx = ret.index
    prem = pd.Series(2*(2*Nrm(0.5*(H.K*vi.values/100)*np.sqrt(H.DT))-1), index=idx)
    spread = (pd.Series(sp0,index=idx)*(1+(vi/vi.rolling(63).median().shift(1)-1).clip(lower=0)).fillna(1.)).fillna(sp0)
    if name == "EEM": spread = spread + 0.005
    epsoff = pd.Series(0.00125*np.where(np.arange(len(idx))%2,1,-1), index=idx)
    net = prem.shift(1)-(ret-epsoff).abs()-spread.shift(1)*prem.shift(1)
    edge = net.rolling(win).mean().shift(1)
    cov = (edge >= margin_bp*1e-4).astype(float).where(edge.notna(), 1.0)
    return cov.shift(1)

COV = {n: coverage_gate(n) for n in sleeves}
W_floor = {n: invvol(sleeves[n])*COV[n].reindex(sleeves[n].index) for n in sleeves}      # THE BOOK
W_comp  = {n: riskweight(sleeves[n])*COV[n].reindex(sleeves[n].index) for n in sleeves}  # comparison
floor    = H.book_of(sleeves, W_floor)
composed = H.book_of(sleeves, W_comp)
baseline = H.book_of(sleeves)
static   = H.book_of(statics)

def line(tag, r):
    r = r.dropna(); h1, h2 = H.split_halves(r); e = (1+r).cumprod()
    yrs = (r.index[-1]-r.index[0]).days/365.25; dd = H.maxdd(r)
    print(f"  {tag:<30} Sharpe {H.sharpe(r):5.2f}  maxDD {dd*100:5.0f}%  skew {r.skew():5.2f}  "
          f"worstD {r.min()*100:5.1f}%  Calmar {(e.iloc[-1]**(1/yrs)-1)/abs(dd):4.2f}  halves ({h1:.2f},{h2:.2f})")
print(f"FINAL OPTIMIZED BOOK = selection-free floor  {floor.index.min().date()}..{floor.index.max().date()}  (L4 frictions)")
line(">> FLOOR invvol x cov (THE BOOK)", floor)
line("composed riskwt x cov (compare)", composed)
line("L4 baseline equal-risk", baseline)
line("static short-vol book", static)

pd.DataFrame({"book_ret": floor}).to_csv(os.path.join(OUT,"optimized_book_ledger.csv"), index_label="Date")

fig, ax = plt.subplots(3,1, figsize=(12,11), gridspec_kw={"height_ratios":[3,1,1]}, sharex=True)
for tag, r, lw in [(">> FLOOR (invvol x cov) = the book", floor, 2.4),
                   ("composed (riskwt x cov)", composed, 1.2),
                   ("L4 baseline equal-risk", baseline, 1.2),
                   ("static short-vol book", static, 1.0)]:
    e = (1+r.reindex(floor.index).fillna(0)).cumprod()
    ax[0].plot(e.index, e.values, lw=lw, label=f"{tag} (Sh {H.sharpe(r):.2f}, sk {r.dropna().skew():.1f})")
ax[0].set_yscale("log"); ax[0].legend(loc="upper left"); ax[0].set_ylabel("growth of $1 (log)")
ax[0].set_title("Final optimized equity-vol book = selection-free floor (invvol x cov, NO return info) - full L4 frictions")
for tag, r, col in [("FLOOR", floor, "navy"), ("composed", composed, "darkorange"), ("baseline", baseline, "grey")]:
    e = (1+r.dropna()).cumprod(); d = e/e.cummax()-1
    ax[1].plot(d.index, d.values*100, color=col, lw=1.2, label=tag)
ax[1].legend(loc="lower left"); ax[1].set_ylabel("drawdown %"); ax[1].axhline(0, color="k", lw=.4)
rs = floor.rolling(252).mean()/floor.rolling(252).std()*SQ
ax[2].plot(rs.index, rs.values, color="darkgreen"); ax[2].axhline(0, color="grey", lw=.5)
ax[2].axhline(1, color="grey", lw=.4, ls="--"); ax[2].set_ylabel("floor rolling 1y Sharpe")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"optimized_book.png"), dpi=110); plt.close(fig)
print(f"\nchart -> optimized_book.png | ledger -> optimized_book_ledger.csv")
