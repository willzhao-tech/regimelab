# -*- coding: utf-8 -*-
"""P3.14 - interactive dashboard for the FLOOR BOOK (volbook_dashboard.html).
Equity / drawdown / rolling-Sharpe, attribution (legs, regime, per-market), k- & spread-sensitivity,
time-in-market, and an honest stats+caveats header. Reuses bookopt_floor (single source of truth)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import bookopt_harness as H, bookopt_floor as F
import plotly.graph_objects as go
import plotly.io as pio

OUT = H.OUT; SQ = H.SQ
CDN = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

def fig_html(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displayModeBar": False})

# ---- core book + comparisons --------------------------------------------------------
book, sleeves, W = F.build()
baseline = H.book_of(sleeves)
statics = {}
for n,_,_,_ in H.PAIRS:
    _, st = H.market(n)[:2] if False else (None, None)
for n in list(sleeves):
    _, st = H.market(n)
    statics[n] = st
static = H.book_of(statics)
allidx = pd.DatetimeIndex(sorted(set().union(*[set(sleeves[k].index) for k in sleeves])))
cal = book.reindex(allidx).fillna(0.0)

def eq(r): return (1+r.reindex(book.index).fillna(0)).cumprod()
f_eq = go.Figure()
for tag, r, c, w in [("floor (the book)", book, "#1f4ea1", 3), ("baseline equal-risk", baseline, "#999", 1.5),
                     ("static short-vol", static, "#c0392b", 1.5)]:
    e = eq(r); f_eq.add_trace(go.Scatter(x=e.index, y=e.values, name=f"{tag} (Sh {H.sharpe(r):.2f})",
                                         line=dict(color=c, width=w)))
f_eq.update_yaxes(type="log", title="growth of $1 (log)")
f_eq.update_layout(title="Equity curve — full L4 frictions, walk-forward", height=430,
                   legend=dict(orientation="h", y=-0.18), margin=dict(l=50,r=20,t=40,b=40))

e = (1+book).cumprod(); dd = (e/e.cummax()-1)*100
eb = (1+baseline).cumprod(); ddb = (eb/eb.cummax()-1)*100
f_dd = go.Figure()
f_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name="floor", fill="tozeroy", line=dict(color="#1f4ea1")))
f_dd.add_trace(go.Scatter(x=ddb.index, y=ddb.values, name="baseline", line=dict(color="#bbb")))
f_dd.update_layout(title="Drawdown %", height=250, margin=dict(l=50,r=20,t=40,b=30))

rs = book.rolling(252).mean()/book.rolling(252).std()*SQ
f_rs = go.Figure(go.Scatter(x=rs.index, y=rs.values, line=dict(color="#27773b")))
f_rs.add_hline(y=1, line=dict(dash="dot", color="#aaa")); f_rs.add_hline(y=0, line=dict(color="#ddd"))
f_rs.update_layout(title="Rolling 1-year Sharpe", height=250, margin=dict(l=50,r=20,t=40,b=30))

# ---- attribution: legs / regime / per-market ----------------------------------------
shortP, longP = [], []
for n,_,_,_ in H.PAIRS:
    r = H.market(n, return_pos=True)
    if r[0] is None: continue
    _,_,info = r
    shortP.append(0.5*info["pA"].where(info["posA"]>0,0.)+0.5*info["pB"].where(info["posB"]>0,0.))
    longP.append(0.5*info["pA"].where(info["posA"]<0,0.)+0.5*info["pB"].where(info["posB"]<0,0.))
gs = pd.concat(shortP,axis=1).sum(axis=1).sum(); gl = pd.concat(longP,axis=1).sum(axis=1).sum()
f_leg = go.Figure(go.Bar(x=["short-vol","long-vol"], y=[gs,gl], marker_color=["#27773b","#c0392b"]))
f_leg.update_layout(title=f"Gross P&L by leg (long-vol = {gl/(gs+gl)*100:.0f}%)", height=300, margin=dict(l=50,r=20,t=40,b=30))

vix = pd.read_csv(os.path.join(OUT,"VIX_all_history.csv"),parse_dates=["Date"]).set_index("Date")["Close"].shift(1).reindex(book.index)
reg = [("VIX<15", book[(vix<15).fillna(False)]), ("15-25", book[((vix>=15)&(vix<25)).fillna(False)]), ("VIX>25", book[(vix>=25).fillna(False)])]
f_reg = go.Figure(go.Bar(x=[t for t,_ in reg], y=[r.mean()*252*100 for _,r in reg], marker_color="#2c6fb5"))
f_reg.update_layout(title="Annualized return by VIX regime (%)", height=300, margin=dict(l=50,r=20,t=40,b=30))

sc = {k: sleeves[k]/(sleeves[k].rolling(63).std().shift(1)) for k in sleeves}
P0 = pd.DataFrame(sc); Wdf = pd.DataFrame(W).reindex(P0.index); PW = P0*Wdf
denom = Wdf.where(PW.notna()).sum(axis=1); bku = (PW.sum(axis=1,skipna=True)/denom).dropna()
lev = (0.10/(bku.rolling(63).std().shift(1)*SQ)).clip(upper=4.0); dn = denom.reindex(bku.index)
contrib = {k: (lev*(PW[k].reindex(bku.index)/dn)).reindex(book.index).fillna(0.).mean()*252*100 for k in sleeves}
f_mkt = go.Figure(go.Bar(x=list(contrib), y=list(contrib.values()), marker_color="#e08a1e"))
f_mkt.update_layout(title="Per-market contribution to book (ann %; SPX carries ~60%)", height=300, margin=dict(l=50,r=20,t=40,b=30))

# ---- sensitivities (verified numbers, recomputed) -----------------------------------
ORIG_K = H.K; ks=[0.75,0.79,0.82,0.85,0.90]; ka=[]; kc=[]
for kk in ks:
    H.K=kk; bk,sl,_=F.build(); ai=pd.DatetimeIndex(sorted(set().union(*[set(sl[m].index) for m in sl])))
    ka.append(H.sharpe(bk)); kc.append(H.sharpe(bk.reindex(ai).fillna(0.)))
H.K=ORIG_K
f_k = go.Figure()
f_k.add_trace(go.Scatter(x=ks,y=ka,name="active",line=dict(color="#1f4ea1")))
f_k.add_trace(go.Scatter(x=ks,y=kc,name="calendar",line=dict(color="#7aa3d8")))
f_k.add_vline(x=0.82, line=dict(dash="dot",color="#888")); f_k.add_hline(y=0,line=dict(color="#eee"))
f_k.update_layout(title="k-sensitivity (measured k=0.82; the #1 model risk)", height=300, xaxis_title="k", yaxis_title="Sharpe", margin=dict(l=50,r=20,t=40,b=40))

ms=[1.0,1.25,1.5,2.0,3.0]; sa=[]; sc2=[]
for m in ms:
    bk,sl,_=F.build(mult=m); ai=pd.DatetimeIndex(sorted(set().union(*[set(sl[x].index) for x in sl])))
    sa.append(H.sharpe(bk)); sc2.append(H.sharpe(bk.reindex(ai).fillna(0.)))
f_sp = go.Figure()
f_sp.add_trace(go.Scatter(x=ms,y=sa,name="floor active",line=dict(color="#1f4ea1")))
f_sp.add_trace(go.Scatter(x=ms,y=sc2,name="floor calendar",line=dict(color="#7aa3d8")))
f_sp.add_hline(y=0,line=dict(color="#eee"))
f_sp.update_layout(title="Spread-sensitivity (stays positive; gate self-protects)", height=300, xaxis_title="spread x", yaxis_title="Sharpe", margin=dict(l=50,r=20,t=40,b=40))

dep = {}
for y in range(2005,2027):
    u = int((allidx.year==y).sum()); b = int((book.index.year==y).sum())
    if u: dep[y] = b/u*100
f_tim = go.Figure(go.Bar(x=[str(y) for y in dep], y=list(dep.values()), marker_color="#6b4ea1"))
f_tim.update_layout(title="Time-in-market by year (% deployed; 2008≈out → sidesteps by absence)", height=300, margin=dict(l=50,r=20,t=40,b=40))

# ---- assemble -----------------------------------------------------------------------
S = lambda r: H.sharpe(r)
stats = f"""
<div class='cards'>
  <div class='card'><div class='v'>1.26</div><div class='l'>Sharpe (active days)</div></div>
  <div class='card'><div class='v'>1.01</div><div class='l'>Sharpe (calendar, idle=0)</div></div>
  <div class='card'><div class='v'>-20%</div><div class='l'>max drawdown</div></div>
  <div class='card'><div class='v'>64%</div><div class='l'>days deployed</div></div>
  <div class='card'><div class='v'>-0.97</div><div class='l'>static short-vol (always on)</div></div>
  <div class='card'><div class='v'>0.6-1.0</div><div class='l'>prudent band (k-risk priced)</div></div>
</div>"""
caveats = """
<ul class='cav'>
  <li><b>Timing is the product.</b> Always-on short-vol loses (-0.97); all value is conditional. Long-vol legs out-earn short (71%).</li>
  <li><b>k is the #1 risk.</b> +-0.03 in k swings Sharpe 0.88&harr;1.98; calm-regime effective k may sit near 0.79 &rarr; plan on calendar ~0.6-1.0.</li>
  <li><b>Spread-fragile but self-protecting.</b> Degrades gracefully, stays positive to 3x (gate steps out); naive book goes negative at 2x.</li>
  <li><b>SPX-carried (~60%); crisis "protection" is absence</b> (2008 traded 1 day) &mdash; sidesteps, doesn't hedge, forgoes post-crash premium.</li>
  <li><b>Premium is a model, not fills.</b> Real per-strike quotes (paid data) would settle the k-band. No look-ahead is machine-checked.</li>
</ul>"""

def row(*figs): return "<div class='row'>" + "".join(f"<div class='cell'>{fig_html(f)}</div>" for f in figs) + "</div>"
html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>{CDN}
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1a1a1a}}
 header{{background:#13294b;color:#fff;padding:20px 28px}} header h1{{margin:0;font-size:20px}}
 header p{{margin:6px 0 0;color:#b9c6df;font-size:13px}}
 .wrap{{max-width:1280px;margin:0 auto;padding:18px}}
 .cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
 .card{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:14px 18px;flex:1;min-width:150px;text-align:center}}
 .card .v{{font-size:26px;font-weight:700;color:#13294b}} .card .l{{font-size:12px;color:#667;margin-top:4px}}
 .row{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px}} .cell{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:8px;flex:1;min-width:380px}}
 .cav{{background:#fff;border:1px solid #e3e6ea;border-left:4px solid #c0392b;border-radius:8px;padding:12px 22px;font-size:13.5px;line-height:1.7}}
 h3{{margin:18px 4px 6px;color:#13294b}}
</style></head><body>
<header><h1>Multi-Market Equity-Vol Book &mdash; the floor book</h1>
<p>1-DTE straddle &middot; selection-free (invvol &times; cost-coverage) &middot; full L4 frictions &middot; walk-forward 2005&ndash;2026 &middot; paper/educational, not advice</p></header>
<div class='wrap'>
 {stats}
 <div class='cav'><b>Honest caveats &mdash; read these</b>{caveats}</div>
 <h3>Performance</h3>
 {row(f_eq)}
 {row(f_dd, f_rs)}
 <h3>Attribution</h3>
 {row(f_leg, f_reg, f_mkt)}
 <h3>Sensitivity &amp; robustness</h3>
 {row(f_k, f_sp)}
 {row(f_tim)}
 <p style='color:#889;font-size:12px;margin-top:18px'>Regenerate: <code>python p3_volbook_dash.py</code>. Source: bookopt_floor.py / bookopt_harness.py. Reproduce numbers via examples/44&ndash;49.</p>
</div></body></html>"""

path = os.path.join(OUT, "volbook_dashboard.html")
with open(path, "w", encoding="utf-8") as fh: fh.write(html)
sz = os.path.getsize(path)/1024
print(f"dashboard -> {path}  ({sz:.0f} KB)")
print(f"  floor active {S(book):.2f} | calendar {S(cal):.2f} | baseline {S(baseline):.2f} | static {S(static):.2f}")
