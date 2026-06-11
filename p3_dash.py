# -*- coding: utf-8 -*-
"""
P3-14: Interactive vol-arb dashboard (self-contained HTML, no server).

Reads volarb_ledger_[A|B|blend].csv (+ p1_attribution.csv if present) from
C:\\Users\\ASUS\\Desktop\\claude doc\\1 and writes volarb_dashboard.html there.

Tabs:
  1. Cumulative P&L (A / B / blend / static short-vol) with range slider
  2. Drawdown panel
  3. Yearly Sharpe bars (blend vs static; A/B toggleable)
  4. Attribution (position-state, VXN regime, crisis windows blend vs static)
  5. Headline metrics table + proxy caveat

plotly include: try CDN tag + inline fallback is fragile in one file, so we
embed plotly.js INLINE (plotly.offline.get_plotlyjs) -> truly self-contained,
opens offline (user is behind China geo-block; CDN may not load).
"""
import io
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly import offline as pyo

ART = r"C:\Users\ASUS\Desktop\claude doc\1"
OUT = os.path.join(ART, "volarb_dashboard.html")
ANN = np.sqrt(252.0)

# ---------------------------------------------------------------- load ledgers
def load_ledger(tag):
    df = pd.read_csv(os.path.join(ART, f"volarb_ledger_{tag}.csv"),
                     parse_dates=["Date"]).set_index("Date").sort_index()
    return df

led = {t: load_ledger(t) for t in ("A", "B", "blend")}

# static short-vol on the same dates, same proxy: pos=+1 always.
# pnl = (iv - rvar) - cost; only position change is the day-1 entry.
base = led["A"]
static_pnl = (base["iv_strike_daily_var"] - base["realized_var"]).copy()
entry_cost = 2.0 * base["VXN"].iloc[0] * 0.5 / 1e4 / 252.0  # |dpos|=1 once
static_pnl.iloc[0] -= entry_cost

pnl = pd.DataFrame({
    "A": led["A"]["daily_pnl"],
    "B": led["B"]["daily_pnl"],
    "blend": led["blend"]["daily_pnl"],
    "static": static_pnl,
})
cum = pnl.cumsum()
dd = cum - cum.cummax()

# ---------------------------------------------------------------- metrics
def sharpe(x):
    s = x.std()
    return float(x.mean() / s * ANN) if s > 0 else np.nan

def metrics_row(name, x, pos=None):
    c = x.cumsum()
    m = {
        "Strategy": name,
        "Ann. Sharpe": sharpe(x),
        "Ann. mean P&L": float(x.mean() * 252),
        "Ann. vol": float(x.std() * ANN),
        "Daily skew": float(x.skew()),
        "Total P&L": float(x.sum()),
        "Max drawdown": float((c - c.cummax()).min()),
        "Worst day": float(x.min()),
        "Best day": float(x.max()),
    }
    if pos is not None:
        m["% days short-vol"] = float((pos > 0).mean() * 100)
        m["% days long-vol"] = float((pos < 0).mean() * 100)
        m["% days flat"] = float((pos == 0).mean() * 100)
    else:
        m["% days short-vol"], m["% days long-vol"], m["% days flat"] = 100.0, 0.0, 0.0
    return m

def alpha_t(y, x):
    """OLS y = a + b*x; return (a*252, t-stat of a)."""
    y = y.values; x = x.values; n = len(y)
    xb, yb = x.mean(), y.mean()
    sxx = ((x - xb) ** 2).sum()
    b = ((x - xb) * (y - yb)).sum() / sxx
    a = yb - b * xb
    resid = y - a - b * x
    s2 = (resid ** 2).sum() / (n - 2)
    se_a = np.sqrt(s2 * (1.0 / n + xb ** 2 / sxx))
    return a * 252, a / se_a

rows = [
    metrics_row("A (regime_combo)", pnl["A"], led["A"]["position"]),
    metrics_row("B (range_forecast)", pnl["B"], led["B"]["position"]),
    metrics_row("Blend (0.5A+0.5B)", pnl["blend"], led["blend"]["position"]),
    metrics_row("Static short-vol", pnl["static"], None),
]
for r, tag in zip(rows[:3], ("A", "B", "blend")):
    a_ann, t = alpha_t(pnl[tag], pnl["static"])
    r["Ann. alpha vs static"] = a_ann
    r["Alpha t-stat (OLS)"] = t
rows[3]["Ann. alpha vs static"] = 0.0
rows[3]["Alpha t-stat (OLS)"] = np.nan

met = pd.DataFrame(rows).set_index("Strategy")
print("=== headline metrics (verify vs spec) ===")
print(met[["Ann. Sharpe", "Daily skew", "Alpha t-stat (OLS)"]].round(2).to_string())

# ---------------------------------------------------------------- figure 1: cum P&L
COL = {"A": "#1f77b4", "B": "#ff7f0e", "blend": "#2ca02c", "static": "#888888"}
LBL = {"A": "A regime_combo", "B": "B range_forecast",
       "blend": "Blend 0.5A+0.5B", "static": "Static short-vol"}

fig1 = go.Figure()
for k in ("A", "B", "blend", "static"):
    fig1.add_trace(go.Scatter(
        x=cum.index, y=cum[k], name=LBL[k],
        line=dict(color=COL[k], width=2.2 if k == "blend" else 1.4,
                  dash="dot" if k == "static" else "solid")))
fig1.update_layout(
    title="Cumulative P&L per unit variance-notional — walk-forward OOS "
          f"{cum.index[0].date()} .. {cum.index[-1].date()} (PROXY instrument)",
    yaxis_title="Cumulative P&L (variance units)",
    hovermode="x unified", template="plotly_white", height=560,
    legend=dict(orientation="h", y=1.06, x=0),
    margin=dict(t=90, l=60, r=30, b=30))
fig1.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(buttons=[
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(count=5, label="5y", step="year", stepmode="backward"),
        dict(count=10, label="10y", step="year", stepmode="backward"),
        dict(step="all", label="all")]))

# ---------------------------------------------------------------- figure 2: drawdown
fig2 = go.Figure()
for k in ("blend", "static", "A", "B"):
    fig2.add_trace(go.Scatter(
        x=dd.index, y=dd[k], name=LBL[k], fill="tozeroy" if k == "blend" else None,
        line=dict(color=COL[k], width=2.0 if k == "blend" else 1.0,
                  dash="dot" if k == "static" else "solid"),
        visible=True if k in ("blend", "static") else "legendonly"))
fig2.update_layout(
    title="Drawdown of cumulative P&L (variance units) — note static's deep "
          "left-tail drawdowns vs the strategies",
    yaxis_title="Drawdown", hovermode="x unified", template="plotly_white",
    height=480, legend=dict(orientation="h", y=1.08, x=0),
    margin=dict(t=80, l=60, r=30, b=30))

# ---------------------------------------------------------------- figure 3: yearly Sharpe
yr = pnl.groupby(pnl.index.year).agg(lambda x: x.mean() / x.std() * ANN if x.std() > 0 else np.nan)
fig3 = go.Figure()
for k, vis in (("blend", True), ("static", True), ("A", "legendonly"), ("B", "legendonly")):
    fig3.add_trace(go.Bar(x=yr.index.astype(str), y=yr[k], name=LBL[k],
                          marker_color=COL[k], visible=vis))
fig3.update_layout(
    title="Yearly annualised Sharpe — strategy vs static short-vol "
          "(2026 is a 48-day stub)",
    yaxis_title="Sharpe (ann.)", barmode="group", template="plotly_white",
    height=480, legend=dict(orientation="h", y=1.08, x=0),
    margin=dict(t=80, l=60, r=30, b=30))
fig3.add_hline(y=0, line_color="#999", line_width=1)

# ---------------------------------------------------------------- figure 4: attribution
attr_path = os.path.join(ART, "p1_attribution.csv")
if os.path.exists(attr_path):
    attr = pd.read_csv(attr_path)
    attr_src = "p1_attribution.csv"
    a_pos = attr[attr["section"] == "a_position_state"]
    b_vxn = attr[attr["section"] == "b_vxn_regime"]
    d_cri = attr[attr["section"] == "d_crisis"]
else:  # fallback: compute from ledgers
    attr_src = "computed from ledgers (p1_attribution.csv not found)"
    bl = led["blend"]
    def bucket(mask, label):
        x = bl.loc[mask, "daily_pnl"]
        return dict(bucket=label, n_days=int(mask.sum()), total_pnl=x.sum(),
                    pnl_share_pct=100 * x.sum() / bl["daily_pnl"].sum(),
                    ann_sharpe_in_bucket=sharpe(x))
    a_pos = pd.DataFrame([
        bucket(bl["position"] > 0, "short_vol (pos>0)"),
        bucket(bl["position"] < 0, "long_vol (pos<0)"),
        bucket(bl["position"] == 0, "flat (pos==0)")])
    b_vxn = pd.DataFrame([
        bucket(bl["VXN"] < 15, "VXN<15"),
        bucket((bl["VXN"] >= 15) & (bl["VXN"] <= 25), "VXN 15-25"),
        bucket(bl["VXN"] > 25, "VXN>25")])
    wins = [("2008-09-01", "2009-03-31", "2008-09..2009-03 (GFC)"),
            ("2011-08-01", "2011-10-31", "2011-08..10 (US dgrade)"),
            ("2018-02-01", "2018-02-28", "2018-02 (Volmageddon)"),
            ("2018-10-01", "2018-12-31", "2018-10..12 (Q4 selloff)"),
            ("2020-02-01", "2020-04-30", "2020-02..04 (COVID)"),
            ("2022-01-01", "2022-12-31", "2022 (full year)"),
            ("2025-04-01", "2025-04-30", "2025-04 (tariff shock)")]
    d_rows = []
    for s, e, lab in wins:
        m = (bl.index >= s) & (bl.index <= e)
        d_rows.append(dict(bucket=lab, total_pnl=bl.loc[m, "daily_pnl"].sum(),
                           static_total_pnl=float(pnl.loc[m, "static"].sum())))
    d_cri = pd.DataFrame(d_rows)

def share_txt(df):
    return [f"{s:.1f}% of P&L" if pd.notna(s) else "" for s in df["pnl_share_pct"]]

fig4a = go.Figure(go.Bar(
    x=a_pos["bucket"], y=a_pos["total_pnl"], marker_color=["#2ca02c", "#1f77b4", "#bbbbbb"],
    text=share_txt(a_pos), textposition="outside"))
fig4a.update_layout(title=f"Blend P&L by position state ({attr_src})",
                    yaxis_title="Total P&L", template="plotly_white", height=400,
                    margin=dict(t=70, l=60, r=30, b=30))

fig4b = go.Figure(go.Bar(
    x=b_vxn["bucket"], y=b_vxn["total_pnl"], marker_color="#1f77b4",
    text=share_txt(b_vxn), textposition="outside"))
fig4b.update_layout(title="Blend P&L by VXN regime",
                    yaxis_title="Total P&L", template="plotly_white", height=400,
                    margin=dict(t=70, l=60, r=30, b=30))

fig4c = go.Figure()
fig4c.add_trace(go.Bar(x=d_cri["bucket"], y=d_cri["total_pnl"],
                       name="Blend", marker_color=COL["blend"]))
fig4c.add_trace(go.Bar(x=d_cri["bucket"], y=d_cri["static_total_pnl"],
                       name="Static short-vol", marker_color=COL["static"]))
fig4c.update_layout(title="Crisis windows: blend vs static short-vol "
                          "(the skew-inversion claim, window by window)",
                    yaxis_title="Total P&L in window", barmode="group",
                    template="plotly_white", height=460,
                    legend=dict(orientation="h", y=1.1, x=0),
                    margin=dict(t=90, l=60, r=30, b=60))
fig4c.add_hline(y=0, line_color="#999", line_width=1)

# ---------------------------------------------------------------- metrics table html
fmt = {
    "Ann. Sharpe": "{:.2f}", "Ann. mean P&L": "{:.4f}", "Ann. vol": "{:.4f}",
    "Daily skew": "{:+.1f}", "Total P&L": "{:.4f}", "Max drawdown": "{:.4f}",
    "Worst day": "{:.5f}", "Best day": "{:.5f}", "% days short-vol": "{:.1f}",
    "% days long-vol": "{:.1f}", "% days flat": "{:.1f}",
    "Ann. alpha vs static": "{:.4f}", "Alpha t-stat (OLS)": "{:.1f}",
}
cols = list(fmt.keys())
thead = "<tr><th>Strategy</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
tbody = ""
for name, r in met.iterrows():
    cells = "".join(
        f"<td>{fmt[c].format(r[c]) if pd.notna(r[c]) else '&mdash;'}</td>" for c in cols)
    cls = ' class="hl"' if "Blend" in name else ""
    tbody += f"<tr{cls}><td><b>{name}</b></td>{cells}</tr>"
metrics_table = f'<table id="metrics_table">{thead}{tbody}</table>'

caveat_html = f"""
<div id="caveat_box">
<h3>PROXY caveat &amp; provenance (read before quoting any number)</h3>
<p><b>Instrument is a proxy, not a tradable.</b> P&amp;L_t = pos_t &times; (iv_t &minus; rvar_t) &minus; cost_t with
iv_t = (VXN_(t-1)/100)&sup2;/252 (strike at prior close), rvar_t = ret_t&sup2;,
cost_t = 2&times;VXN_(t-1)&times;0.5/10&#8308;/252 &times; |&Delta;pos| (0.5 vol-pt spread).
A 1-day variance swap struck at VXN does <b>not</b> trade; this inflates Sharpe LEVELS for strategy and
baseline alike. A real NDX var-swap / option implementation is expected around <b>0.8&ndash;1.2 Sharpe</b>.</p>
<p><b>Where the Sharpe comes from:</b> walk-forward out-of-sample only &mdash; train 1260 trading days,
test 252, parameters picked by best TRAIN Sharpe, TEST blocks concatenated
({cum.index[0].date()} .. {cum.index[-1].date()}, {len(pnl)} days). Signals are causal
(Parkinson forecasts shifted 1 day; pos_t = s_(t-1)). Grids searched: family A 18 combos
(r_hi&times;r_lo&times;d = 3&times;2&times;3), family B 9 combos (b1&times;b2 = 3&times;3) &mdash; 27 total, so deflate
single-test p-values accordingly.</p>
<p><b>The robust claims are relative, not absolute:</b> (1) alpha of the regime strategies over static
short-vol (blend OLS alpha t &asymp; {met.loc['Blend (0.5A+0.5B)', 'Alpha t-stat (OLS)']:.0f});
(2) skew inversion &mdash; daily skew {met.loc['Blend (0.5A+0.5B)', 'Daily skew']:+.0f} for the blend vs
{met.loc['Static short-vol', 'Daily skew']:+.0f} for static. <b>What breaks if the favorable assumption
fails:</b> if VXN is not a fair 1-day variance strike (term/risk premia at the 30-day tenor, no daily
mark), absolute Sharpes fall toward ~1 and the long-vol leg's crisis convexity shrinks; wider spreads
than 0.5 vol-pt hit families A/B (which trade) more than static. No tail deletion or clipping was applied.</p>
</div>
"""

# ---------------------------------------------------------------- assemble html
config = {"responsive": True, "displaylogo": False}
def div(fig, did):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       div_id=did, config=config)

divs = {
    "fig_cumpnl": div(fig1, "fig_cumpnl"),
    "fig_drawdown": div(fig2, "fig_drawdown"),
    "fig_yearly_sharpe": div(fig3, "fig_yearly_sharpe"),
    "fig_attr_pos": div(fig4a, "fig_attr_pos"),
    "fig_attr_vxn": div(fig4b, "fig_attr_vxn"),
    "fig_attr_crisis": div(fig4c, "fig_attr_crisis"),
}

plotlyjs = pyo.get_plotlyjs()  # inline -> self-contained, no CDN needed

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Vol-arb dashboard — NQ/VXN walk-forward (proxy)</title>
<style>
 body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f5f6f8; color: #222; }}
 header {{ background: #1b2a41; color: #fff; padding: 14px 24px; }}
 header h1 {{ margin: 0; font-size: 20px; }}
 header p {{ margin: 4px 0 0; font-size: 12.5px; color: #c7d0dd; }}
 .tabbar {{ display: flex; gap: 4px; background: #1b2a41; padding: 0 16px; }}
 .tabbtn {{ background: #2c3e58; color: #dde5ef; border: none; padding: 10px 18px;
            cursor: pointer; font-size: 14px; border-radius: 8px 8px 0 0; }}
 .tabbtn.active {{ background: #f5f6f8; color: #1b2a41; font-weight: 600; }}
 .tabpane {{ display: none; padding: 18px 24px 40px; }}
 .tabpane.show {{ display: block; }}
 .card {{ background: #fff; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
          padding: 12px; margin-bottom: 18px; }}
 #metrics_table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
 #metrics_table th, #metrics_table td {{ border: 1px solid #dfe3e8; padding: 6px 8px; text-align: right; }}
 #metrics_table th {{ background: #eef1f5; }}
 #metrics_table td:first-child {{ text-align: left; }}
 #metrics_table tr.hl {{ background: #eaf6ea; }}
 #caveat_box {{ background: #fff8e6; border: 1px solid #e6cf8b; border-radius: 10px;
                padding: 14px 18px; font-size: 13.5px; line-height: 1.5; }}
 #caveat_box h3 {{ margin-top: 2px; color: #8a6d1a; }}
</style></head><body>
<header>
 <h1>Long-short variance on NQ/VXN — walk-forward OOS dashboard</h1>
 <p>Proxy instrument (1-day variance at VXN strike, NOT directly tradable) &middot;
    {cum.index[0].date()} .. {cum.index[-1].date()} &middot; {len(pnl)} trading days &middot;
    families A (regime_combo) + B (range_forecast), blend = 0.5A+0.5B &middot;
    attribution source: {attr_src}</p>
</header>
<div class="tabbar">
 <button class="tabbtn active" onclick="showTab('tab1',this)">1 &middot; Cumulative P&amp;L</button>
 <button class="tabbtn" onclick="showTab('tab2',this)">2 &middot; Drawdowns</button>
 <button class="tabbtn" onclick="showTab('tab3',this)">3 &middot; Yearly Sharpe</button>
 <button class="tabbtn" onclick="showTab('tab4',this)">4 &middot; Attribution</button>
 <button class="tabbtn" onclick="showTab('tab5',this)">5 &middot; Metrics &amp; caveats</button>
</div>
<div id="tab1" class="tabpane show"><div class="card">{divs['fig_cumpnl']}</div></div>
<div id="tab2" class="tabpane"><div class="card">{divs['fig_drawdown']}</div></div>
<div id="tab3" class="tabpane"><div class="card">{divs['fig_yearly_sharpe']}</div></div>
<div id="tab4" class="tabpane">
 <div class="card">{divs['fig_attr_pos']}</div>
 <div class="card">{divs['fig_attr_vxn']}</div>
 <div class="card">{divs['fig_attr_crisis']}</div>
</div>
<div id="tab5" class="tabpane">
 <div class="card">{metrics_table}</div>
 {caveat_html}
</div>
<script>{plotlyjs}</script>
<script>
function showTab(id, btn) {{
  document.querySelectorAll('.tabpane').forEach(p => p.classList.remove('show'));
  document.querySelectorAll('.tabbtn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('show');
  btn.classList.add('active');
  requestAnimationFrame(() =>
    document.getElementById(id).querySelectorAll('.js-plotly-plot')
            .forEach(el => Plotly.Plots.resize(el)));
}}
window.addEventListener('resize', () =>
  document.querySelectorAll('.tabpane.show .js-plotly-plot')
          .forEach(el => Plotly.Plots.resize(el)));
</script>
</body></html>"""

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write(html)

# ---------------------------------------------------------------- verify
size = os.path.getsize(OUT)
need = ["fig_cumpnl", "fig_drawdown", "fig_yearly_sharpe", "fig_attr_pos",
        "fig_attr_vxn", "fig_attr_crisis", "metrics_table", "caveat_box"]
with io.open(OUT, "r", encoding="utf-8") as f:
    body = f.read()
missing = [d for d in need if f'id="{d}"' not in body]
assert size > 100_000, f"file too small: {size}"
assert not missing, f"missing div ids: {missing}"
print(f"\nOK: {OUT}")
print(f"size: {size:,} bytes ({size/1e6:.2f} MB)")
print(f"div ids present: {need}")
