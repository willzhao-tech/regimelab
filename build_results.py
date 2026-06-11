# -*- coding: utf-8 -*-
"""RENDER-FROM-DATA: compute EVERY headline number for the vol-book and write results.json.
The single source of truth. The working paper, README, and dashboard should cite from this file --
never hand-type a statistic into prose. (Ported discipline from the semiconductor study, which hit
the prose/code drift bug twice.) Regenerate: python build_results.py  ->  <DATA_DIR>/results.json"""
import os, sys, json
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
import bookopt_harness as H, bookopt_floor as F, bookopt_stats as S

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"; SQ = H.SQ
def r2(x): return None if x is None or (isinstance(x,float) and np.isnan(x)) else round(float(x), 4)

def build_floor():
    H._DATA.clear()
    book, sleeves, W = F.build()
    allidx = pd.DatetimeIndex(sorted(set().union(*[set(sleeves[k].index) for k in sleeves])))
    cal = book.reindex(allidx).fillna(0.0)
    return book, cal, sleeves, W, allidx

book, cal, sleeves, W, allidx = build_floor()
st = F.stat_line(book)
R = {}
R["meta"] = {"data_through": str(book.index.max().date()), "start": str(book.index.min().date()),
             "n_markets": len(sleeves), "markets": list(sleeves), "k": H.K,
             "instrument": "1-DTE delta-hedged ATM straddle", "frictions": "L4 (stress-spread, 2x long-leg, "
             "discrete strikes, floors, EEM assignment, optional margin funding)"}

# ---- data vintage + code version: pin every result to its exact inputs (P1 discipline) ----
import hashlib, subprocess
def _sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
vintage = {}
for _, uf, vf, _ in H.PAIRS:
    for stem in (uf, vf):
        p = os.path.join(OUT, stem + "_all_history.csv")
        if os.path.exists(p) and stem not in vintage:
            vintage[stem] = _sha16(p)
try:
    code_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
except Exception:
    code_rev = None
R["vintage"] = {"code_commit": code_rev, "data_sha256_16": vintage}

# ---- headline floor ----
R["floor"] = {"sharpe_active": r2(S.sharpe_ann(book)), "sharpe_calendar": r2(S.sharpe_ann(cal)),
              "maxdd": r2(st["maxdd"]), "skew": r2(st["skew"]), "worst_day": r2(st["worst"]),
              "calmar": r2(st["calmar"]), "cagr": r2(st["cagr"]),
              "deployed_pct": r2(len(book)/len(allidx)*100), "n_active": len(book), "n_union": len(allidx)}

# ---- comparisons ----
baseline = H.book_of(sleeves)
statics = {n: H.market(n)[1] for n in sleeves}
static = H.book_of(statics)
R["comparison"] = {"baseline_equalrisk_sharpe": r2(S.sharpe_ann(baseline)),
                   "static_shortvol_sharpe": r2(S.sharpe_ann(static))}

# ---- inference ----
p0a,*_ = S.psr(book,0.0); p0c,*_ = S.psr(cal,0.0); p5c,*_ = S.psr(cal,0.5/SQ)
cia = S.block_bootstrap_sharpe(book, 20, 3000, 11); cic = S.block_bootstrap_sharpe(cal, 20, 3000, 11)
names = list(sleeves); pv = [S.sharpe_pvalue(sleeves[k]) for k in names]
keep, crit = S.bh_fdr(pv, 0.05)
tvals = [S.sharpe_ann(sleeves[k])/SQ*np.sqrt(len(sleeves[k])) for k in names]
dsr = {str(N): {"null": r2(S.deflated_sharpe(cal,N)["dsr"]),
                "emp": r2(S.deflated_sharpe(cal,N,[float(sleeves[k].mean()/sleeves[k].std()) for k in names])["dsr"])}
       for N in (10, 50, 250)}
R["inference"] = {"psr_active_gt0": r2(p0a), "psr_cal_gt0": r2(p0c), "psr_cal_gt0p5": r2(p5c),
                  "boot_ci_active": [r2(cia["lo"]), r2(cia["hi"])], "boot_ci_calendar": [r2(cic["lo"]), r2(cic["hi"])],
                  "fdr_survive": int(keep.sum()), "fdr_t_gt3": int(sum(t>3 for t in tvals)),
                  "fdr_critical_p": r2(crit), "dsr": dsr}

# ---- per-sleeve ----
R["sleeves"] = {k: {"sharpe": r2(S.sharpe_ann(sleeves[k])), "t": r2(tvals[i]),
                    "p": r2(pv[i]), "fdr_keep": bool(keep[i])} for i, k in enumerate(names)}

# ---- robustness ----
def floor_at(tr, te, fund=0.0):
    o1,o2 = H.TRAIN, H.TEST; H.TRAIN, H.TEST = tr, te
    bk,_,_ = F.build(fund_rf=fund); H.TRAIN, H.TEST = o1,o2; return S.sharpe_ann(bk)
R["robustness"] = {
    "window_1000_200": r2(floor_at(1000,200)), "window_1260_252": r2(floor_at(1260,252)),
    "window_1500_252": r2(floor_at(1500,252)),
    "start_2008": r2(S.sharpe_ann(book.loc["2008":])), "start_2013": r2(S.sharpe_ann(book.loc["2013":])),
    "funding_5pct": r2(S.sharpe_ann(F.build(fund_rf=0.05)[0])), "adverse_corner": r2(floor_at(1000,200,0.05))}

# ---- spread & k sweeps ----
R["spread_sweep_active"] = {f"x{m}": r2(S.sharpe_ann(F.build(mult=m)[0])) for m in (1.0,1.5,2.0,3.0)}
ksw = {}
o = H.K
for k in (0.79, 0.82, 0.85):
    H.K = k; bk,sl,_ = F.build(); ai = pd.DatetimeIndex(sorted(set().union(*[set(sl[x].index) for x in sl])))
    ksw[str(k)] = {"active": r2(S.sharpe_ann(bk)), "calendar": r2(S.sharpe_ann(bk.reindex(ai).fillna(0.0)))}
H.K = o
R["k_sweep"] = ksw

# ---- placebo ----
rng = np.random.default_rng(7); real = S.sharpe_ann(cal)
def placebo(use_cov, Nrep=800):
    out = np.empty(Nrep)
    for i in range(Nrep):
        Wr = {k: (pd.Series(rng.uniform(0.2,1.0), index=sleeves[k].index) *
                  (F.coverage_gate(k).reindex(sleeves[k].index) if use_cov else 1.0)) for k in sleeves}
        out[i] = S.sharpe_ann(H.book_of(sleeves, Wr).reindex(allidx).fillna(0.0))
    return float((out < real).mean()*100)
R["placebo"] = {"vs_random_x_coverage_pct": r2(placebo(True)), "vs_random_no_coverage_pct": r2(placebo(False))}

path = os.path.join(OUT, "results.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(R, f, indent=2)
print(f"results.json -> {path}")
print(f"  floor active {R['floor']['sharpe_active']} | calendar {R['floor']['sharpe_calendar']} "
      f"| boot CI cal {R['inference']['boot_ci_calendar']} | FDR {R['inference']['fdr_survive']}/8 "
      f"| deployed {R['floor']['deployed_pct']}%")
