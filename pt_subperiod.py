# -*- coding: utf-8 -*-
"""
ATTACK: FEW-CRISIS ARTIFACT?
Is Sleeve A's edge over vol-matched NQ just a handful of crash dodges by the 200d trend brake?

Plan:
  1. Build Sleeve A (full config) and a VOL-MATCHED long-NQ benchmark.
     "vol-matched" = scale the benchmark daily-return series so its realized full-sample
     vol equals Sleeve A's, so any Sharpe difference is NOT a leverage/vol artifact.
  2. Sub-periods: 2001-2008, 2009-2016, 2017-2026 -> Sharpe, maxDD for both.
  3. KILLER TEST: drop the crash windows (2001-2002, 2008-09, 2020 Feb-Apr, 2022)
     from BOTH return streams and recompute. If Sleeve A's whole edge evaporates
     outside crashes, the "alpha" is just crash-dodging.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import sleeveA_harness as H

SQ = np.sqrt(252.0)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

df, fomc = H.load()

# --- Build the two return streams (both causal; harness shifts positions, monthly hold) ---
rA, posA = H.sleeve_a(df, fomc)                 # full Sleeve A: trend+voltarget+fomc
rNQ_vt   = H.voltarget_nq(df)                   # vol-controlled long NQ (no trend, no fomc)

# raw buy&hold NQ daily returns (for reference)
rNQ_raw  = df["Close"].pct_change().dropna()

# Align all three on common index
idx = rA.index.intersection(rNQ_vt.index).intersection(rNQ_raw.index)
rA      = rA.reindex(idx).dropna()
rNQ_vt  = rNQ_vt.reindex(idx).dropna()
rNQ_raw = rNQ_raw.reindex(idx).dropna()
idx = rA.index.intersection(rNQ_vt.index).intersection(rNQ_raw.index)
rA, rNQ_vt, rNQ_raw = rA.reindex(idx), rNQ_vt.reindex(idx), rNQ_raw.reindex(idx)

def vol_match(target_series, ref_series):
    """Scale target_series (constant multiplier) so its full-sample vol == ref_series vol.
    Constant scaling does NOT change Sharpe or maxDD-as-fraction; it just makes the
    level comparison honest. (Sharpe is scale-invariant, so vol-matching is for fairness
    of the 'is it beta' narrative, not for changing the Sharpe.)"""
    s = ref_series.std() / target_series.std()
    return target_series * s

# vol-match the NQ benchmarks to Sleeve A's full-sample vol
rNQ_vt_m  = vol_match(rNQ_vt,  rA)
rNQ_raw_m = vol_match(rNQ_raw, rA)

def m(r):
    d = H.metrics(r)
    return d

def show(label, r):
    d = m(r)
    print(f"  {label:26s} n={d['n']:5d}  Sharpe={d['sharpe']:6.3f}  CAGR={d['cagr']*100:7.2f}%  "
          f"maxDD={d['maxdd']*100:7.2f}%  vol={d['vol']*100:5.2f}%  worst1y={d['worst1y']*100:7.2f}%")

print("="*120)
print("FULL SAMPLE (vol-matched NQ benchmarks to Sleeve A vol):", idx.min().date(), "->", idx.max().date())
print("="*120)
show("Sleeve A (full)", rA)
show("NQ vol-target (matched)", rNQ_vt_m)
show("NQ buy&hold (matched)", rNQ_raw_m)
print("  [note: Sharpe is scale-invariant; vol-match only equalizes level for the maxDD/CAGR read]")

# ----------------------------------------------------------------------------
# 2) SUB-PERIODS
# ----------------------------------------------------------------------------
subperiods = [
    ("2001-2008", "2001-01-01", "2008-12-31"),
    ("2009-2016", "2009-01-01", "2016-12-31"),
    ("2017-2026", "2017-01-01", "2026-12-31"),
]
print()
print("="*120)
print("SUB-PERIODS  (Sleeve A vs vol-matched NQ-voltarget vs vol-matched NQ buy&hold)")
print("  Sharpe is computed WITHIN each sub-period on each sub-period's own returns")
print("="*120)
for name, a, b in subperiods:
    sl = slice(pd.Timestamp(a), pd.Timestamp(b))
    print(f"\n--- {name} ---")
    show("Sleeve A", rA.loc[sl])
    show("NQ vol-target (matched)", rNQ_vt_m.loc[sl])
    show("NQ buy&hold (matched)", rNQ_raw_m.loc[sl])

# ----------------------------------------------------------------------------
# 3) KILLER TEST: exclude crash windows from BOTH, recompute
# ----------------------------------------------------------------------------
crash_windows = [
    ("2001-2002 dotcom", "2001-01-01", "2002-12-31"),
    ("2008-09 GFC",      "2008-01-01", "2009-06-30"),
    ("2020 COVID",       "2020-02-15", "2020-04-30"),
    ("2022 bear",        "2022-01-01", "2022-12-31"),
]
crash_mask = pd.Series(False, index=idx)
for name, a, b in crash_windows:
    crash_mask |= (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
crash_mask = pd.Series(crash_mask.values, index=idx)

n_crash = int(crash_mask.sum())
n_total = len(idx)
print()
print("="*120)
print("KILLER TEST: EXCLUDE crash windows from BOTH streams, then recompute")
print(f"  crash windows: {[c[0] for c in crash_windows]}")
print(f"  crash days excluded: {n_crash} / {n_total} = {100*n_crash/n_total:.1f}%")
print("="*120)

# (a) crash days ONLY -- where does Sleeve A's outperformance come from?
print("\n[A] CRASH DAYS ONLY (the windows we will remove):")
show("Sleeve A", rA[crash_mask])
show("NQ vol-target (matched)", rNQ_vt_m[crash_mask])
show("NQ buy&hold (matched)", rNQ_raw_m[crash_mask])

# (b) NON-crash days -- the real question
keep = ~crash_mask
print("\n[B] NON-CRASH DAYS ONLY (crash windows removed from BOTH):")
show("Sleeve A", rA[keep])
show("NQ vol-target (matched)", rNQ_vt_m[keep])
show("NQ buy&hold (matched)", rNQ_raw_m[keep])

# Re-vol-match on the NON-crash subsample so the level comparison is honest there too
rNQ_vt_m2  = vol_match(rNQ_vt[keep],  rA[keep])
rNQ_raw_m2 = vol_match(rNQ_raw[keep], rA[keep])
print("\n[B'] NON-CRASH, benchmarks RE-vol-matched on the non-crash subsample:")
show("Sleeve A", rA[keep])
show("NQ vol-target (re-matched)", rNQ_vt_m2)
show("NQ buy&hold (re-matched)", rNQ_raw_m2)

# ----------------------------------------------------------------------------
# 4) Decompose: how much of Sleeve A's TOTAL outperformance lives in crash windows?
#    Compare cumulative log-excess return Sleeve A - NQ_vt(matched) in crash vs non-crash.
# ----------------------------------------------------------------------------
print()
print("="*120)
print("DECOMPOSITION: cumulative excess (Sleeve A - NQ vol-target matched)")
print("="*120)
excess = rA - rNQ_vt_m
tot_ex_all   = excess.sum()
tot_ex_crash = excess[crash_mask].sum()
tot_ex_keep  = excess[keep].sum()
print(f"  sum of daily excess, ALL days     : {tot_ex_all*100:8.2f}  (arith, %)")
print(f"  sum of daily excess, CRASH days   : {tot_ex_crash*100:8.2f}  ({100*tot_ex_crash/tot_ex_all:5.1f}% of total)")
print(f"  sum of daily excess, NON-CRASH    : {tot_ex_keep*100:8.2f}  ({100*tot_ex_keep/tot_ex_all:5.1f}% of total)")

# Also a Sharpe-of-the-excess (information-ratio style) in each bucket
def ir(x):
    x = pd.Series(x).dropna()
    if x.std() == 0 or len(x) < 30: return float('nan')
    return float(x.mean()/x.std()*SQ)
print(f"\n  IR of excess (Sleeve A - NQ_vt_matched):")
print(f"    all days   : {ir(excess):6.3f}")
print(f"    crash days : {ir(excess[crash_mask]):6.3f}")
print(f"    non-crash  : {ir(excess[keep]):6.3f}")

# ----------------------------------------------------------------------------
# 5) Robustness: vary the crash definition (widen / shrink) to make sure the
#    verdict isn't sensitive to exact boundaries.
# ----------------------------------------------------------------------------
print()
print("="*120)
print("ROBUSTNESS: vary crash-window padding; recompute NON-CRASH Sharpe gap")
print("="*120)
def build_mask(pad_days):
    msk = pd.Series(False, index=idx)
    for name, a, b in crash_windows:
        lo = pd.Timestamp(a) - pd.Timedelta(days=pad_days)
        hi = pd.Timestamp(b) + pd.Timedelta(days=pad_days)
        msk |= (idx >= lo) & (idx <= hi)
    return pd.Series(msk.values, index=idx)

print(f"  {'pad(d)':>7} {'%excl':>7} {'ShA_nc':>8} {'ShNQvt_nc':>10} {'gap':>7} {'ShNQbh_nc':>10}")
for pad in [0, 30, 90, 180]:
    msk = build_mask(pad)
    kp = ~msk
    sA = H.metrics(rA[kp])["sharpe"]
    vt = vol_match(rNQ_vt[kp], rA[kp])
    bh = vol_match(rNQ_raw[kp], rA[kp])
    sVT = H.metrics(vt)["sharpe"]
    sBH = H.metrics(bh)["sharpe"]
    print(f"  {pad:7d} {100*msk.sum()/len(idx):6.1f}% {sA:8.3f} {sVT:10.3f} {sA-sVT:7.3f} {sBH:10.3f}")

print("\nDONE.")
