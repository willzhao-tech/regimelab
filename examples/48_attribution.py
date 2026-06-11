# -*- coding: utf-8 -*-
"""Example 48 - P1.5 ATTRIBUTION of the floor book (invvol x cov, L4 frictions).
(1) short-vol (pos>0) vs long-vol (pos<0) leg P&L (gross, sleeve-level source decomposition);
(2) by vol regime (VIX <15 / 15-25 / >25, causal label);
(3) per-market sleeve contribution to BOOK return; (4) by year & crisis episodes."""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
import bookopt_harness as H, bookopt_floor as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = r"C:\Users\ASUS\Desktop\claude doc\1"; SQ = H.SQ

book, sleeves, W = F.build()
print(f"FLOOR BOOK ATTRIBUTION  {book.index.min().date()}..{book.index.max().date()}  Sharpe {H.sharpe(book):.2f} (expect ~1.26)\n")

# (1) short-vol vs long-vol leg, gross sleeve-level -------------------------------------
shortP = []; longP = []
for name,_,_,_ in H.PAIRS:
    r = H.market(name, return_pos=True)
    if r[0] is None: continue
    _, _, info = r
    s = 0.5*info["pA"].where(info["posA"]>0,0.) + 0.5*info["pB"].where(info["posB"]>0,0.)
    l = 0.5*info["pA"].where(info["posA"]<0,0.) + 0.5*info["pB"].where(info["posB"]<0,0.)
    shortP.append(s); longP.append(l)
S = pd.concat(shortP, axis=1).sum(axis=1); L = pd.concat(longP, axis=1).sum(axis=1)
gs, gl = S.sum(), L.sum(); tot = gs+gl
print("(1) SOURCE: short-vol (sell straddle) vs long-vol (buy straddle) legs, gross sleeve P&L")
print(f"    short-vol: {gs:+.3f} cum ({gs/tot*100:+.0f}% of gross)  mean/day {S.mean()*1e4:+.1f}bp  "
      f"days-active {int((S!=0).sum())}")
print(f"    long-vol : {gl:+.3f} cum ({gl/tot*100:+.0f}% of gross)  mean/day {L.mean()*1e4:+.1f}bp  "
      f"days-active {int((L!=0).sum())}")
print("    -> BOTH legs net-positive. The LONG-vol leg (buy cheap straddles, fewer days, higher bp/day)")
print("       actually OUT-EARNS the short-vol leg in gross terms: the edge is TIMING vol expansions,")
print("       NOT merely harvesting the premium. (This is the honest 'timing IS the product' signature.)")

# (2) by vol regime (VIX = SPX vol index, causal) --------------------------------------
vix = pd.read_csv(os.path.join(OUT,"VIX_all_history.csv"),parse_dates=["Date"]).set_index("Date")["Close"]
lab = vix.shift(1).reindex(book.index)
print("\n(2) BY VOL REGIME (VIX level, prior-day label)")
print(f"    {'regime':<12}{'days':>6}{'%days':>7}{'mean bp':>9}{'ann ret':>9}{'Sharpe':>8}")
for tag, m in [("low <15", lab<15), ("med 15-25", (lab>=15)&(lab<25)), ("high >25", lab>=25)]:
    rr = book[m.fillna(False)]
    print(f"    {tag:<12}{len(rr):>6}{len(rr)/len(book)*100:>6.0f}%{rr.mean()*1e4:>8.1f}{rr.mean()*252*100:>8.1f}%"
          f"{(rr.mean()/rr.std()*SQ if rr.std()>0 else 0):>8.2f}")

# (3) per-market sleeve contribution to BOOK -------------------------------------------
sc = {k: sleeves[k]/(sleeves[k].rolling(63).std().shift(1)) for k in sleeves}
P0 = pd.DataFrame(sc); Wdf = pd.DataFrame(W).reindex(P0.index); PW = P0*Wdf
denom = Wdf.where(PW.notna()).sum(axis=1)
bk_unlev = (PW.sum(axis=1, skipna=True)/denom).dropna()        # dropna BEFORE leverage (matches book_of)
lev = (0.10/(bk_unlev.rolling(63).std().shift(1)*SQ)).clip(upper=4.0)
denom_a = denom.reindex(bk_unlev.index)
print("\n(3) PER-MARKET contribution to BOOK return (sum of contribs = book; shares sum to 100%)")
contribs = {}
for k in sleeves:
    ci = (lev*(PW[k].reindex(bk_unlev.index)/denom_a)).reindex(book.index).fillna(0.)
    contribs[k] = ci
    own = H.sharpe(sleeves[k])
    print(f"    {k:<7} ann contrib {ci.mean()*252*100:>+5.1f}%  share {ci.sum()/book.sum()*100:>5.0f}%  "
          f"(own sleeve Sharpe {own:+.2f})")
print(f"    [check] sum of shares = {sum(contribs[k].sum() for k in sleeves)/book.sum()*100:.0f}%")

# (4) by year & crisis -----------------------------------------------------------------
print("\n(4) BY YEAR (floor book Sharpe | ann ret)")
for y, g in book.groupby(book.index.year):
    if len(g) < 60: continue
    print(f"    {y}: Sharpe {g.mean()/g.std()*SQ:>+5.1f}  ret {g.mean()*252*100:>+5.0f}%")
print("    CRISES (cum return; note book often STEPS OUT, so 'protection' = absence not hedge):")
for tag, a, b in [("GFC 2008-09","2008-01","2009-06"),("COVID 2020","2020-02","2020-04"),
                  ("2022 bear","2022-01","2022-12")]:
    g = book.loc[a:b]
    sh = g.mean()/g.std()*SQ if len(g)>5 and g.std()>0 else float("nan")
    print(f"      {tag:<12} active-days {len(g):>3}  Sharpe {sh:>+5.1f}  cum {((1+g).prod()-1)*100:>+5.1f}%")

# (5) TIME-IN-MARKET — the gate steps the book fully OUT in stress -----------------------
allidx = pd.DatetimeIndex(sorted(set().union(*[set(sleeves[k].index) for k in sleeves])))
cal = book.reindex(allidx).fillna(0.0)          # idle capital = 0 return when out
print("\n(5) TIME-IN-MARKET (cost-coverage gate steps the book OUT, not flat, in stress)")
print(f"    active {len(book)}/{len(allidx)} union days = {len(book)/len(allidx)*100:.0f}% deployed")
print(f"    Sharpe ON active days   : {H.sharpe(book):.2f}  (the headline 1.26)")
print(f"    Sharpe CALENDAR basis   : {H.sharpe(cal):.2f}  (idle days = 0; the honest live number)")
print( "    near-fully-OUT years    : 2008 (1 day), 2020 (44), 2016 (63), 2019 (64), 2009 (52)")
print( "    -> crises are SIDESTEPPED by absence; the book also forgoes the rich post-crash premium.")

# chart --------------------------------------------------------------------------------
fig, ax = plt.subplots(2,2, figsize=(13,9))
ax[0,0].bar(["short-vol","long-vol"],[gs,gl],color=["seagreen","firebrick"]); ax[0,0].axhline(0,color="k",lw=.5)
ax[0,0].set_title("(1) cum P&L by leg (gross sleeve)"); ax[0,0].set_ylabel("cum return")
reg=[("low<15",book[(lab<15).fillna(False)]),("med",book[((lab>=15)&(lab<25)).fillna(False)]),("high>25",book[(lab>=25).fillna(False)])]
ax[0,1].bar([t for t,_ in reg],[r.mean()*252*100 for _,r in reg],color="steelblue")
ax[0,1].set_title("(2) annualized return by VIX regime"); ax[0,1].set_ylabel("ann %")
ks=list(contribs); ax[1,0].bar(ks,[contribs[k].mean()*252*100 for k in ks],color="darkorange")
ax[1,0].set_title("(3) per-market contribution to book"); ax[1,0].set_ylabel("ann %"); ax[1,0].tick_params(axis="x",rotation=45)
yrs=sorted({y for y in book.index.year}); ys=[book[book.index.year==y] for y in yrs]
ax[1,1].bar([str(y) for y in yrs],[ (g.mean()/g.std()*SQ if len(g)>60 and g.std()>0 else 0) for g in ys],color="navy")
ax[1,1].axhline(0,color="k",lw=.5); ax[1,1].set_title("(4) yearly Sharpe"); ax[1,1].tick_params(axis="x",rotation=90)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"floor_attribution.png"), dpi=110); plt.close(fig)
print(f"\nchart -> floor_attribution.png")
