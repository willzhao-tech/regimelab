# -*- coding: utf-8 -*-
"""MASTER CALENDAR generator — platform-native rebuild of the big-move catalyst calendar.
Big-move days are COMPUTED from our gated CSVs (full history, per-instrument self-calibrating
thresholds: 92nd pct of |open-to-close| and of range, the ancestor study's ~13-20% flag rate);
catalyst NARRATIVES are merged from catalyst_narratives.json (1,142 days hand-attributed in the
original research, 2018-2026 — preserved, never fabricated); scheduled events (FOMC/NFP/CPI)
overlay from market_calendar. Output: <DATA_DIR>/master_calendar.html (standalone, no deps).

    python build_master_calendar.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from market_calendar import calendar_df

DATA = os.environ.get("REGIMELAB_DATA_DIR", r"C:\Users\ASUS\Desktop\claude doc\1")
INSTS = [("NQ", "NQ_F", "#5aa9ff"), ("A50", "A50", "#ff5d6c"), ("EURUSD", "EURUSD", "#ffb648"),
         ("US10Y", "US10Y", "#c792ea"), ("WTI", "WTI", "#3ddc97"), ("XAU", "XAU", "#f4d35e"),
         ("VIX", "VIX", "#ff7b9c")]
Q = 0.92

recs = {}
for iid, stem, _ in INSTS:
    df = pd.read_csv(os.path.join(DATA, stem + "_all_history.csv"), parse_dates=["Date"]
                     ).set_index("Date").sort_index()
    if iid == "VIX":                                   # dual trigger: big level change OR stress level
        chg = df["Close"].pct_change() * 100
        thr = chg.abs().quantile(Q)
        flag = (chg.abs() >= thr) | (df["Close"] >= 30)
        rng = (df["High"] - df["Low"]) / df["Close"].shift(1) * 100
    else:
        chg = (df["Close"] - df["Open"]) / df["Open"] * 100
        rng = (df["High"] - df["Low"]) / df["Open"] * 100
        ok = df["Open"] > 0
        chg, rng = chg.where(ok), rng.where(ok)
        flag = (chg.abs() >= chg.abs().quantile(Q)) | (rng >= rng.quantile(Q))
    n = int(flag.sum())
    print(f"  {iid:<7} {len(df):>6} rows  flagged {n} ({n/len(df)*100:.1f}%)")
    for d in df.index[flag.fillna(False)]:
        k = d.date().isoformat()
        recs.setdefault(k, {})[iid] = [round(float(chg.loc[d]), 2) if chg.loc[d] == chg.loc[d] else 0.0,
                                        round(float(rng.loc[d]), 2) if rng.loc[d] == rng.loc[d] else 0.0]

cats = {}
cn = os.path.join(DATA, "catalyst_narratives.json")
if os.path.exists(cn):
    raw = json.load(open(cn, encoding="utf-8"))
    for k, v in raw.items():
        for iid, txt in v.get("cats", {}).items():
            cats.setdefault(k, {})[iid] = txt
print(f"  narratives merged: {sum(len(v) for v in cats.values())} instrument-days")

ev = {r["date"].date().isoformat(): r["event"] + ("~" if r["source"] == "projected" else "")
      for _, r in calendar_df(start="1999-01-01").iterrows()}

years = sorted({int(k[:4]) for k in recs})
payload = dict(insts=[[i, c] for i, _, c in INSTS], recs=recs, cats=cats, ev=ev, years=years)
J = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Master Calendar - Big-Move Days - regimelab</title><style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;900&family=Spline+Sans+Mono:wght@400;600&display=swap');
:root{--bg:#070a10;--panel:#0e131c;--p2:#0b0f17;--line:#1c2531;--ink:#e9eef6;--mut:#7e8ca3;--dim:#4a5667;
--disp:'Archivo',sans-serif;--mono:'Spline Sans Mono',monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:var(--disp);padding:28px 20px 60px}
.wrap{max-width:1200px;margin:0 auto}
.tag{font-family:var(--mono);font-size:10.5px;letter-spacing:.22em;color:#3ddc97;text-transform:uppercase}
h1{font-weight:900;font-size:28px;margin-top:8px}h1 .em{color:var(--mut);font-weight:600}
.sub{color:var(--mut);font-family:var(--mono);font-size:11.5px;margin-top:9px;line-height:1.7;max-width:920px}
.legend{display:flex;gap:7px;margin-top:14px;flex-wrap:wrap}
.pill{font-family:var(--mono);font-size:10.5px;padding:4px 9px;border-radius:5px;border:1px solid var(--line);
display:flex;align-items:center;gap:6px;color:var(--mut);cursor:pointer;user-select:none}
.pill .sw{width:8px;height:8px;border-radius:2px}.pill.off{opacity:.32}
.yrnav{display:inline-flex;gap:2px;padding:4px;background:var(--p2);border:1px solid var(--line);border-radius:10px;flex-wrap:wrap;margin:16px 0 10px}
.yrnav button{font-family:var(--mono);font-size:11px;padding:6px 10px;border-radius:7px;border:none;background:transparent;color:var(--mut);cursor:pointer}
.yrnav button.on{background:#5aa9ff;color:#04121f;font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:8px 0 14px}
.kpi{background:var(--p2);border:1px solid var(--line);border-radius:11px;padding:11px 13px}
.kpi .big{font-family:var(--mono);font-size:20px;font-weight:600}
.kpi .lab{font-family:var(--mono);font-size:9px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin-top:5px}
.layout{display:grid;grid-template-columns:1fr 330px;gap:16px}
@media(max-width:900px){.layout{grid-template-columns:1fr}}
.cal{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px}
.months{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:1100px){.months{grid-template-columns:repeat(2,1fr)}}
.mo h3{font-family:var(--mono);font-size:10.5px;color:var(--mut);margin-bottom:6px}
.dow,.days{display:grid;grid-template-columns:repeat(7,1fr);gap:3px}
.dow span{font-family:var(--mono);font-size:8px;color:var(--dim);text-align:center}
.day{aspect-ratio:1;border-radius:4px;background:#0c1119;display:flex;align-items:center;justify-content:center;
position:relative;font-family:var(--mono);font-size:9.5px;color:var(--dim)}
.day.empty{background:transparent}.day.has{cursor:pointer;color:#06090f;font-weight:600}
.day.has:hover,.day.sel{outline:1.5px solid var(--ink);outline-offset:-1px;z-index:2}
.day .ev{position:absolute;top:1px;right:2px;width:4px;height:4px;border-radius:50%;background:#fff}
.side{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;position:sticky;top:16px;align-self:start;max-height:calc(100vh - 32px);overflow-y:auto}
.side .ph{font-family:var(--mono);font-size:11px;color:var(--dim);text-align:center;padding:28px 8px;line-height:1.6}
.side .dt{font-family:var(--mono);font-size:13.5px;font-weight:600}
.side .br{font-family:var(--mono);font-size:10px;color:var(--mut);margin:3px 0 10px}
.row{border-top:1px solid var(--line);padding:10px 0}
.row .top{display:flex;align-items:center;gap:8px}
.row .nm{font-family:var(--mono);font-size:11.5px;font-weight:600}
.row .mv{font-family:var(--mono);font-size:11.5px;margin-left:auto}
.row .mv.up{color:#2bd9a8}.row .mv.dn{color:#ff5d6c}
.row .cat{font-size:11.5px;color:var(--mut);line-height:1.5;margin-top:5px}
.row .sw{width:9px;height:9px;border-radius:2px}
.foot{margin-top:24px;border-top:1px solid var(--line);padding-top:14px;font-family:var(--mono);font-size:9.5px;color:var(--dim);line-height:1.7}
</style></head><body><div class="wrap">
<div class="tag">regimelab - master calendar</div>
<h1>Big-move days <span class="em">- 7 instruments - full history</span></h1>
<div class="sub">Flags computed from the platform's gated data (per-instrument 92nd-percentile open-to-close and range
thresholds; VIX dual-trigger incl. close&ge;30). Catalyst narratives 2018-2026 preserved from the original attribution
research. White dot = scheduled event day (FOMC / NFP / CPI; ~ = projected). Click a day.</div>
<div class="legend" id="pills"></div>
<div class="yrnav" id="yrs"></div>
<div class="kpis" id="kpis"></div>
<div class="layout"><div class="cal"><div class="months" id="months"></div></div>
<div class="side" id="side"><div class="ph">click a flagged day<br>for moves + catalyst</div></div></div>
<div class="foot">Generated by build_master_calendar.py - regimelab. Narratives: catalyst_narratives.json (hand-attributed
in the original macro-calendar research; days without text show the scheduled-event tag or 'no attributed catalyst').
Paper research only - not investment advice.</div>
</div><script>
const M=__PAYLOAD__;
const ON=Object.fromEntries(M.insts.map(x=>[x[0],true]));
let YEAR=M.years[M.years.length-1],SEL=null;
const CMAP=Object.fromEntries(M.insts.map(x=>[x[0],x[1]]));
const SCALE=["#1d3a5c","#1f5c8a","#2bd9a8","#e8a33d","#ff7b4d","#ff5d6c","#ff2d55"];
function active(k){const r=M.recs[k];if(!r)return[];return Object.keys(r).filter(i=>ON[i]);}
function render(){
  document.getElementById('yrs').innerHTML=M.years.map(y=>`<button class="${y===YEAR?'on':''}" onclick="YEAR=${y};render()">${y}</button>`).join('');
  document.getElementById('pills').innerHTML=M.insts.map(([i,c])=>
    `<div class="pill ${ON[i]?'':'off'}" onclick="ON['${i}']=!ON['${i}'];render()"><span class="sw" style="background:${c}"></span>${i}</div>`).join('');
  let mh='';let tot=0,sys=0,onev=0;
  for(let m=0;m<12;m++){
    const first=new Date(Date.UTC(YEAR,m,1)),days=new Date(Date.UTC(YEAR,m+1,0)).getUTCDate(),start=(first.getUTCDay()+6)%7;
    let cells='';for(let i=0;i<start;i++)cells+='<div class="day empty"></div>';
    for(let d=1;d<=days;d++){
      const k=`${YEAR}-${String(m+1).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
      const a=active(k),n=a.length,evd=M.ev[k];
      if(n){tot++;if(n>=3)sys++;if(evd)onev++;}
      const bg=n?SCALE[Math.min(n-1,SCALE.length-1)]:'';
      cells+=`<div class="day ${n?'has':''} ${SEL===k?'sel':''}" ${n?`style="background:${bg}" onclick="show('${k}')"`:''}>`+
             `${n?d:'<span style=\\'opacity:.35\\'>'+d+'</span>'}${evd&&n?'<span class="ev"></span>':''}</div>`;
    }
    mh+=`<div class="mo"><h3>${first.toLocaleString('en-US',{month:'short',timeZone:'UTC'})} ${YEAR}</h3>
      <div class="dow">${'Mo Tu We Th Fr Sa Su'.split(' ').map(x=>`<span>${x}</span>`).join('')}</div>
      <div class="days">${cells}</div></div>`;
  }
  document.getElementById('months').innerHTML=mh;
  document.getElementById('kpis').innerHTML=
    `<div class="kpi"><div class="big">${tot}</div><div class="lab">big-move days ${YEAR}</div></div>`+
    `<div class="kpi"><div class="big">${sys}</div><div class="lab">systemic (3+ instruments)</div></div>`+
    `<div class="kpi"><div class="big">${tot?Math.round(onev/tot*100):0}%</div><div class="lab">on scheduled event days</div></div>`+
    `<div class="kpi"><div class="big">${Object.keys(M.cats).filter(k=>k.startsWith(YEAR)).length}</div><div class="lab">days with attributed catalyst</div></div>`;
  if(SEL)show(SEL,true);
}
window.show=function(k,keep){
  SEL=k;if(!keep)render();
  const r=M.recs[k]||{},c=M.cats[k]||{},evd=M.ev[k];
  const rows=Object.entries(r).filter(([i])=>ON[i]).map(([i,[chg,rng]])=>{
    const cat=c[i]||(evd?`Scheduled event day (${evd}). No hand-attributed narrative.`:'No attributed catalyst (pre-2018 or idiosyncratic).');
    return `<div class="row"><div class="top"><span class="sw" style="background:${CMAP[i]}"></span>
      <span class="nm">${i}</span><span class="mv ${chg>=0?'up':'dn'}">${chg>=0?'+':''}${chg}% / rng ${rng}%</span></div>
      <div class="cat">${cat}</div></div>`;}).join('');
  document.getElementById('side').innerHTML=
    `<div class="dt">${k}</div><div class="br">${evd?('scheduled: '+evd):'no scheduled event'} - ${Object.keys(r).length} instrument(s) flagged</div>${rows}`;
};
render();
</script></body></html>"""

out = os.path.join(DATA, "master_calendar.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML.replace("__PAYLOAD__", J))
print(f"\nmaster_calendar.html -> {out}  ({os.path.getsize(out)/1024:.0f} KB, "
      f"{len(recs)} big-move days, {len(years)} years)")
