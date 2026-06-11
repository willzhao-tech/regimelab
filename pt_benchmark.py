# -*- coding: utf-8 -*-
"""
BENCHMARK BATTLE: Does the 200d-TREND overlay add anything over just vol-controlling NQ?

Attack: Compare Sleeve A vs voltarget_nq (no trend) head-to-head.
Quantify: Sharpe, maxDD, Calmar, worst-1y, time-to-recover from worst DD.
Question: Is trend's contribution a ROBUST drawdown improvement, or within noise?
          Is it worth the whipsaw drag in non-crash years?

We isolate the TREND effect cleanly by holding everything else fixed and toggling
ONLY use_trend. We also compare the full Sleeve A (trend+fomc) vs voltarget_nq so we
know what the "package" claim is.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import sleeveA_harness as H

SQ = np.sqrt(252.0)
df, fomc = H.load()

# ---------- helpers ----------
def equity(r):
    r = pd.Series(r).dropna()
    return (1.0 + r).cumprod()

def drawdown_analysis(r):
    """Return maxDD, the trough date, peak date, recovery date, and time-to-recover (calendar days & trading days)."""
    r = pd.Series(r).dropna()
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = eq/peak - 1.0
    trough_date = dd.idxmin()
    maxdd = float(dd.min())
    # peak just before trough
    pre = eq.loc[:trough_date]
    peak_val = pre.cummax().iloc[-1]
    # peak date = last time equity == running peak before trough
    peak_date = pre[pre >= peak_val * (1-1e-12)].index
    peak_date = peak_date[peak_date <= trough_date]
    peak_date = peak_date[-1] if len(peak_date) else pre.index[0]
    # recovery: first date after trough where eq >= peak_val
    post = eq.loc[trough_date:]
    rec = post[post >= peak_val]
    if len(rec):
        rec_date = rec.index[0]
        cal_days = (rec_date - peak_date).days
        # trading days from trough to recovery
        td_trough_rec = int(eq.index.get_indexer([rec_date])[0] - eq.index.get_indexer([trough_date])[0])
        td_peak_rec = int(eq.index.get_indexer([rec_date])[0] - eq.index.get_indexer([peak_date])[0])
        recovered = True
    else:
        rec_date = None
        cal_days = (eq.index[-1] - peak_date).days  # underwater so far
        td_trough_rec = int(len(eq) - 1 - eq.index.get_indexer([trough_date])[0])
        td_peak_rec = int(len(eq) - 1 - eq.index.get_indexer([peak_date])[0])
        recovered = False
    return dict(maxdd=maxdd, peak_date=peak_date, trough_date=trough_date, rec_date=rec_date,
                recovered=recovered, recover_cal_days=cal_days,
                td_trough_to_rec=td_trough_rec, td_peak_to_rec=td_peak_rec)

def calendar_year_returns(r):
    r = pd.Series(r).dropna()
    return r.groupby(r.index.year).apply(lambda x: (1+x).prod()-1.0)

def rolling_1y_dd(r):
    eq = equity(r)
    return float(eq.pct_change(252).min())

def full_metrics(name, r):
    m = H.metrics(r)
    da = drawdown_analysis(r)
    m["recover_td"] = da["td_peak_to_rec"]
    m["recovered"] = da["recovered"]
    m["dd_peak"] = da["peak_date"].date()
    m["dd_trough"] = da["trough_date"].date()
    m["dd_rec"] = da["rec_date"].date() if da["rec_date"] is not None else "UNDERWATER"
    m["name"] = name
    return m

# ---------- build the contestants (matched params, vol_kind/win/target/max_lev/cost identical) ----------
COMMON = dict(vol_kind="parkinson", vol_win=21, trend_win=200, target=0.15,
              fomc_boost=0.5, cost=0.0005, max_lev=3.0)

# 1) voltarget_nq  : trend OFF, fomc OFF, voltarget ON   (the benchmark)
r_vt, p_vt = H.sleeve_a(df, fomc, use_trend=False, use_voltarget=True, use_fomc=False, **COMMON)

# 2) voltarget + trend (NO fomc) : isolates the pure TREND contribution vs benchmark
r_vt_tr, p_vt_tr = H.sleeve_a(df, fomc, use_trend=True, use_voltarget=True, use_fomc=False, **COMMON)

# 3) full Sleeve A : trend ON, fomc ON, voltarget ON  (the advertised sleeve)
r_A, p_A = H.sleeve_a(df, fomc, use_trend=True, use_voltarget=True, use_fomc=True, **COMMON)

# 4) voltarget + fomc (NO trend) : control to see how much of A's edge is fomc not trend
r_vt_f, p_vt_f = H.sleeve_a(df, fomc, use_trend=False, use_voltarget=True, use_fomc=True, **COMMON)

# align all on common index
idx = r_vt.index.intersection(r_vt_tr.index).intersection(r_A.index).intersection(r_vt_f.index)
series = {
    "voltarget_nq (no trend)": r_vt.reindex(idx),
    "voltarget+trend (no fomc)": r_vt_tr.reindex(idx),
    "voltarget+fomc (no trend)": r_vt_f.reindex(idx),
    "FULL Sleeve A (trend+fomc)": r_A.reindex(idx),
}
pos = {
    "voltarget_nq (no trend)": p_vt.reindex(idx),
    "voltarget+trend (no fomc)": p_vt_tr.reindex(idx),
}

print("="*100)
print("HEAD-TO-HEAD METRICS  (full sample {} -> {}, n={})".format(idx[0].date(), idx[-1].date(), len(idx)))
print("="*100)
hdr = f"{'strategy':<30}{'Sharpe':>8}{'CAGR':>8}{'maxDD':>8}{'Calmar':>8}{'worst1y':>9}{'vol':>7}{'recTD':>7}{'recov':>7}"
print(hdr)
print("-"*100)
rows = {}
for name, r in series.items():
    m = full_metrics(name, r)
    rows[name] = m
    print(f"{name:<30}{m['sharpe']:>8.3f}{m['cagr']:>8.2%}{m['maxdd']:>8.1%}{m['calmar']:>8.2f}"
          f"{m['worst1y']:>9.1%}{m['vol']:>7.1%}{m['recover_td']:>7d}{str(m['recovered']):>7}")
print()
for name in series:
    m = rows[name]
    print(f"  {name:<30} worst-DD path: peak {m['dd_peak']} -> trough {m['dd_trough']} -> recover {m['dd_rec']}")

# ---------- the core attack: TREND contribution = (vt+trend) - (vt) ----------
print()
print("="*100)
print("ISOLATED TREND CONTRIBUTION  =  [voltarget+trend]  minus  [voltarget_nq]   (fomc held OFF in both)")
print("="*100)
base = rows["voltarget_nq (no trend)"]
trd  = rows["voltarget+trend (no fomc)"]
print(f"  Sharpe : {base['sharpe']:.3f} -> {trd['sharpe']:.3f}   (delta {trd['sharpe']-base['sharpe']:+.3f})")
print(f"  maxDD  : {base['maxdd']:.1%} -> {trd['maxdd']:.1%}   (delta {trd['maxdd']-base['maxdd']:+.1%}, "
      f"{'IMPROVED' if trd['maxdd']>base['maxdd'] else 'WORSE'} by {abs(trd['maxdd']-base['maxdd'])*100:.1f}pp)")
print(f"  Calmar : {base['calmar']:.2f} -> {trd['calmar']:.2f}   (delta {trd['calmar']-base['calmar']:+.2f})")
print(f"  worst1y: {base['worst1y']:.1%} -> {trd['worst1y']:.1%}   (delta {trd['worst1y']-base['worst1y']:+.1%})")
print(f"  recover(trading days, peak->recover of WORST dd): {base['recover_td']} -> {trd['recover_td']}")

# ---------- whipsaw drag in non-crash years ----------
print()
print("="*100)
print("WHIPSAW DRAG: per-calendar-year return, voltarget_nq vs voltarget+trend (no fomc)")
print("Positive 'trend-base' = trend HELPED that year; negative = trend DRAGGED (whipsaw).")
print("="*100)
cy_base = calendar_year_returns(series["voltarget_nq (no trend)"])
cy_trd  = calendar_year_returns(series["voltarget+trend (no fomc)"])
yrs = sorted(set(cy_base.index) | set(cy_trd.index))
print(f"{'year':<6}{'voltarget_nq':>14}{'vt+trend':>12}{'diff(trend-base)':>20}{'time-in-mkt(trend)':>20}")
help_yrs, drag_yrs = [], []
tim = pos["voltarget+trend (no fomc)"]  # exposure of trend version
# fraction of days trend is fully out (sig=0) per year -> approximate via positions==0
inmkt = (pos["voltarget+trend (no fomc)"] > 1e-9).astype(float)
inmkt_y = inmkt.groupby(inmkt.index.year).mean()
for y in yrs:
    b = cy_base.get(y, np.nan); t = cy_trd.get(y, np.nan); d = t-b
    im = inmkt_y.get(y, np.nan)
    flag = ""
    if not np.isnan(d):
        (help_yrs if d>0 else drag_yrs).append((y,d))
        flag = "  <== trend DRAG" if d < -0.02 else ("  <== trend HELP" if d > 0.02 else "")
    print(f"{y:<6}{b:>14.1%}{t:>12.1%}{d:>20.1%}{im:>19.0%}{flag}")
print()
help_sum = sum(d for _,d in help_yrs); drag_sum = sum(d for _,d in drag_yrs)
print(f"  Years trend HELPED: {len(help_yrs)}  (sum of positive diffs = {help_sum:+.1%})")
print(f"  Years trend DRAGGED: {len(drag_yrs)} (sum of negative diffs = {drag_sum:+.1%})")
print(f"  Net (help+drag): {help_sum+drag_sum:+.1%}  -> trend's edge concentrates where?")
# the big crash years
big = sorted([(y,d) for y,d in help_yrs+drag_yrs], key=lambda kv: kv[1])
print(f"  WORST 3 trend-drag years: {[(y, f'{d:+.1%}') for y,d in big[:3]]}")
print(f"  BEST 3 trend-help years : {[(y, f'{d:+.1%}') for y,d in big[-3:][::-1]]}")

# ---------- is the DD improvement within noise? bootstrap the difference in maxDD & sharpe ----------
print()
print("="*100)
print("IS IT NOISE?  Stationary block-bootstrap of the DAILY return DIFFERENCE (vt+trend) - (vt)")
print("Resample blocks of the PAIRED daily-return diffs; also bootstrap each path's maxDD separately.")
print("="*100)

rng = np.random.default_rng(7)
rb = series["voltarget_nq (no trend)"].values
rt = series["voltarget+trend (no fomc)"].values
n = len(rb)
BL = 21      # ~1 month blocks
NB = 4000

def boot_indices():
    idxs = []
    while len(idxs) < n:
        start = rng.integers(0, n)
        idxs.extend(range(start, min(start+BL, n)))
    return np.array(idxs[:n])

def mdd(arr):
    eq = np.cumprod(1.0+arr)
    return float((eq/np.maximum.accumulate(eq) - 1.0).min())

def shp(arr):
    s = arr.std()
    return float(arr.mean()/s*SQ) if s>0 else np.nan

d_sharpe, d_mdd, d_mean = [], [], []
for _ in range(NB):
    bi = boot_indices()
    bb, bt = rb[bi], rt[bi]
    d_sharpe.append(shp(bt)-shp(bb))
    d_mdd.append(mdd(bt)-mdd(bb))     # positive = trend has SHALLOWER dd (less negative)
    d_mean.append((bt-bb).mean()*252) # annualized mean return diff
d_sharpe = np.array(d_sharpe); d_mdd = np.array(d_mdd); d_mean = np.array(d_mean)

def ci(a): return np.percentile(a,2.5), np.percentile(a,50), np.percentile(a,97.5)
lo,md,hi = ci(d_sharpe)
print(f"  delta Sharpe (trend-base): median {md:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]   "
      f"P(delta>0)={np.mean(d_sharpe>0):.2%}")
lo,md,hi = ci(d_mdd)
print(f"  delta maxDD  (trend-base): median {md:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]   "
      f"P(trend shallower DD)={np.mean(d_mdd>0):.2%}")
lo,md,hi = ci(d_mean)
print(f"  delta ann mean ret       : median {md:+.2%}  95% CI [{lo:+.2%}, {hi:+.2%}]   "
      f"P(delta>0)={np.mean(d_mean>0):.2%}")

# ---------- sub-period: does trend ONLY earn its keep in 2000-02 / 2008 / 2020/2022 crashes? ----------
print()
print("="*100)
print("SUB-PERIOD ROBUSTNESS: split sample, recompute trend's Sharpe & maxDD edge")
print("="*100)
splits = [("2000-2009", "1999-01-01","2009-12-31"),
          ("2010-2019", "2010-01-01","2019-12-31"),
          ("2020-2026", "2020-01-01","2026-12-31"),
          ("ex-crashes (drop 2000-02,2008,2020,2022)", None, None)]
def sub(r, lo, hi):
    r = pd.Series(r)
    return r.loc[lo:hi]
crash_years = {2000,2001,2002,2008,2020,2022}
print(f"{'period':<42}{'Sh_base':>9}{'Sh_trend':>10}{'dSharpe':>9}{'dMaxDD':>9}")
for label, lo, hi in splits:
    if label.startswith("ex-crashes"):
        mask = ~series["voltarget_nq (no trend)"].index.year.isin(crash_years)
        b = series["voltarget_nq (no trend)"][mask]; t = series["voltarget+trend (no fomc)"][mask]
    else:
        b = sub(series["voltarget_nq (no trend)"], lo, hi)
        t = sub(series["voltarget+trend (no fomc)"], lo, hi)
    mb, mt = H.metrics(b), H.metrics(t)
    print(f"{label:<42}{mb['sharpe']:>9.3f}{mt['sharpe']:>10.3f}"
          f"{mt['sharpe']-mb['sharpe']:>9.3f}{mt['maxdd']-mb['maxdd']:>9.1%}")

print()
print("="*100)
print("FULL vs BENCHMARK headline (what the sleeve actually advertises): FULL Sleeve A vs voltarget_nq")
print("="*100)
A = rows["FULL Sleeve A (trend+fomc)"]; B = rows["voltarget_nq (no trend)"]
print(f"  Sharpe {B['sharpe']:.3f} -> {A['sharpe']:.3f}  ({A['sharpe']-B['sharpe']:+.3f})")
print(f"  maxDD  {B['maxdd']:.1%} -> {A['maxdd']:.1%}  ({A['maxdd']-B['maxdd']:+.1%})")
print(f"  Calmar {B['calmar']:.2f} -> {A['calmar']:.2f}")
print(f"  worst1y {B['worst1y']:.1%} -> {A['worst1y']:.1%}")
print(f"  recover(td) {B['recover_td']} -> {A['recover_td']}")
