# -*- coding: utf-8 -*-
"""
BREAK-IT attack on Sleeve A (vol-targeted, 200d-trend-braked, FOMC-tilted long NQ).
Three probes:
 (1) FAST-CRASH WHIPSAW in 2020: trend-exit/re-entry timing P&L decomposition.
 (2) REBAL-TIMING LUCK: shift monthly rebal across 21 day-of-month offsets, measure Sharpe swing.
 (3) 200d-MA arbitrariness: sweep trend_win, measure crash-protection sensitivity.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import sleeveA_harness as H

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
SQ = np.sqrt(252.0)

df, fomc = H.load()
df = df[~df.index.duplicated(keep="last")].sort_index()
print("DATA:", df.index.min().date(), "->", df.index.max().date(), "n=", len(df))

def base_metrics(r, label=""):
    m = H.metrics(r)
    print(f"  {label:28s} sharpe={m['sharpe']:.3f} cagr={m['cagr']*100:6.2f}% mdd={m['maxdd']*100:7.2f}% "
          f"calmar={m['calmar']:.2f} worst1y={m['worst1y']*100:7.2f}% vol={m['vol']*100:5.2f}% n={m['n']}")
    return m

# ------------------------------------------------------------------
# Baseline full Sleeve A
# ------------------------------------------------------------------
print("\n=== BASELINE (full Sleeve A) ===")
r_full, pos_full = H.sleeve_a(df, fomc)
m_full = base_metrics(r_full, "Sleeve A (full)")

# Decompose: vol-target only (no trend, no fomc) = the beta-ish benchmark
r_vt = H.voltarget_nq(df)
base_metrics(r_vt, "vol-target only (no trend)")

# trend+vol no fomc
r_tv, _ = H.sleeve_a(df, fomc, use_fomc=False)
base_metrics(r_tv, "vol-target + trend (no fomc)")

# ==================================================================
# PROBE 1: 2020 FAST-CRASH WHIPSAW
# ==================================================================
print("\n" + "="*70)
print("PROBE 1: 2020 V-SHAPED CRASH — does 200d trend whipsaw?")
print("="*70)

px = df["Close"]
ret = px.pct_change()
trend_ma = px.rolling(200).mean()
sig = (px > trend_ma).astype(float)        # causal trend signal (1=in, 0=out)

# Find trend transitions in 2020
y2020 = slice("2020-01-01", "2020-12-31")
sig20 = sig.loc["2019-12-01":"2020-12-31"]
trans = sig20.diff().fillna(0.0)
print("\nTrend signal transitions Dec2019-Dec2020 (signal value AFTER change):")
for d, v in trans[trans != 0].items():
    print(f"   {d.date()}  -> {'IN (price>200ma)' if sig.loc[d]==1 else 'OUT (price<200ma)'}   close={px.loc[d]:.1f}  200ma={trend_ma.loc[d]:.1f}")

# Crash low and pre-crash high
crash_win = px.loc["2020-02-01":"2020-06-30"]
print(f"\n2020 pre-crash high: {crash_win.loc['2020-02-01':'2020-02-29'].max():.1f} on {crash_win.loc['2020-02-01':'2020-02-29'].idxmax().date()}")
print(f"2020 crash LOW:      {crash_win.min():.1f} on {crash_win.idxmin().date()}")

# Where does the trend signal EXIT relative to the bottom?
sig_2020 = sig.loc["2020-01-01":"2020-07-31"]
exit_dates = sig_2020[(sig_2020.diff()==-1)].index
reentry_dates = sig_2020[(sig_2020.diff()==1)].index
low_date = crash_win.idxmin()
for ed in exit_dates:
    print(f"   EXIT on {ed.date()}: close={px.loc[ed]:.1f}  (bottom was {crash_win.min():.1f} on {low_date.date()}; "
          f"exit is {(ed-low_date).days} days {'AFTER' if ed>low_date else 'BEFORE'} the bottom)")
for rd in reentry_dates:
    print(f"   RE-ENTRY on {rd.date()}: close={px.loc[rd]:.1f}")

# Trend-timing P&L: compare full vs the SAME strategy with trend disabled, over 2020 crash window.
# Decompose the *trend-braking* contribution specifically in the crash window.
win2020 = slice("2020-02-15", "2020-09-30")  # crash + recovery
r_with_trend, _   = H.sleeve_a(df, fomc, use_trend=True,  use_fomc=False)
r_no_trend,   _   = H.sleeve_a(df, fomc, use_trend=False, use_fomc=False)
seg_with = r_with_trend.loc[win2020]
seg_no   = r_no_trend.loc[win2020]
def cum(s): return (1+s).prod()-1
print(f"\n2020 crash+recovery window [{win2020.start} .. {win2020.stop}]:")
print(f"   WITH 200d trend brake : total return = {cum(seg_with)*100:7.2f}%   (n={len(seg_with)})")
print(f"   WITHOUT trend (vol only): total return = {cum(seg_no)*100:7.2f}%")
print(f"   >>> TREND-TIMING P&L over crash+recovery = {(cum(seg_with)-cum(seg_no))*100:+.2f} pct-pts")

# Finer: split into DOWN-leg (Feb15-Mar23) and UP-leg (Mar23-Sep30)
down = slice("2020-02-15", "2020-03-23")
up   = slice("2020-03-24", "2020-09-30")
for name, sl in [("DOWN-leg (Feb15-Mar23)", down), ("UP-leg (Mar24-Sep30)", up)]:
    dw = cum(r_with_trend.loc[sl]); dn = cum(r_no_trend.loc[sl])
    print(f"   {name:26s}: trend={dw*100:7.2f}%  notrend={dn*100:7.2f}%  trend-edge={ (dw-dn)*100:+7.2f}pp")

# full-year 2020
for name, sl in [("FULL 2020", slice("2020-01-01","2020-12-31"))]:
    dw = cum(r_with_trend.loc[sl]); dn = cum(r_no_trend.loc[sl])
    print(f"   {name:26s}: trend={dw*100:7.2f}%  notrend={dn*100:7.2f}%  trend-edge={ (dw-dn)*100:+7.2f}pp")

# ==================================================================
# PROBE 2: REBAL-TIMING LUCK — shift monthly rebal day across offsets
# ==================================================================
print("\n" + "="*70)
print("PROBE 2: REBAL-TIMING LUCK — shift the monthly reset day")
print("="*70)
# The harness _monthly() resets on the FIRST trading day a new calendar month appears.
# We replicate sleeve_a but reset on the k-th trading day of each month, k=0..20.
def sleeve_a_offset(df, fomc, offset=0, vol_kind="parkinson", vol_win=21, trend_win=200,
                    target=0.15, fomc_boost=0.5, cost=0.0005, max_lev=3.0,
                    use_trend=True, use_voltarget=True, use_fomc=True):
    px=df["Close"]; ret=px.pct_change()
    vol=H._vol(df,vol_kind,vol_win)
    sig=(px>px.rolling(trend_win).mean()).astype(float) if use_trend else pd.Series(1.0,index=px.index)
    lev=(target/vol).clip(upper=max_lev) if use_voltarget else pd.Series(1.0,index=px.index)
    desired=(lev*sig).shift(1).fillna(0.0)
    # month-rebal with offset: hold last value; update only on the offset-th trading day of each month
    idx = desired.index
    # rank trading days within each month
    ym = pd.Series([(d.year, d.month) for d in idx], index=idx)
    day_in_month = ym.groupby(ym).cumcount()  # 0-based position within month
    out=[]; cur=0.0
    dvals = desired.values
    dim = day_in_month.values
    for i in range(len(idx)):
        if dim[i]==offset:
            cur=float(dvals[i])
        # if month shorter than offset, we just never update that month (keeps prior) - acceptable edge
        out.append(cur)
    base=pd.Series(out,index=idx)
    if use_fomc:
        win=pd.Series(False,index=ret.index)
        for p in ret.index.get_indexer(fomc, method="bfill"):
            if 1<=p<len(ret.index): win.iloc[p-1]=True; win.iloc[p]=True
        pos=base.where(~win, np.minimum(base+fomc_boost,4.0))
    else:
        pos=base
    r=(pos*ret - cost*pos.diff().abs().fillna(0.0)).dropna()
    return r, pos

sharpes=[]; cagrs=[]; mdds=[]
print("\n offset  sharpe   cagr%    mdd%    calmar")
for k in range(21):
    rk,_ = sleeve_a_offset(df, fomc, offset=k)
    mk = H.metrics(rk)
    sharpes.append(mk['sharpe']); cagrs.append(mk['cagr']); mdds.append(mk['maxdd'])
    if k in (0,1,2,5,10,15,20):
        print(f"   {k:3d}   {mk['sharpe']:.3f}  {mk['cagr']*100:6.2f}  {mk['maxdd']*100:7.2f}  {mk['calmar']:.2f}")
sharpes=np.array(sharpes)
print(f"\n  Sharpe across 21 offsets: min={sharpes.min():.3f} max={sharpes.max():.3f} "
      f"mean={sharpes.mean():.3f} std={sharpes.std():.3f} spread={sharpes.max()-sharpes.min():.3f}")
print(f"  CAGR  across 21 offsets: min={min(cagrs)*100:.2f}% max={max(cagrs)*100:.2f}% spread={(max(cagrs)-min(cagrs))*100:.2f}pp")
print(f"  MDD   across 21 offsets: min={min(mdds)*100:.2f}% max={max(mdds)*100:.2f}%")
# sanity: offset 0 should approx match harness monthly baseline
print(f"  (sanity) offset-0 sharpe {sharpes[0]:.3f} vs harness baseline {m_full['sharpe']:.3f}")

# ==================================================================
# PROBE 3: 200d-MA ARBITRARINESS — sweep trend window, crash protection
# ==================================================================
print("\n" + "="*70)
print("PROBE 3: 200d-MA ARBITRARY — sweep trend_win, measure crash protection")
print("="*70)
print("\n trend_win  sharpe   cagr%    mdd%    calmar   worst1y%  2020ret%  2008ret%")
sweep_sharpe={}
for tw in [50,75,100,125,150,175,200,225,250,300,400]:
    rtw,_ = H.sleeve_a(df, fomc, trend_win=tw)
    mtw = H.metrics(rtw)
    sweep_sharpe[tw]=mtw['sharpe']
    r2020 = (1+rtw.loc["2020-01-01":"2020-12-31"]).prod()-1
    r2008 = (1+rtw.loc["2008-01-01":"2008-12-31"]).prod()-1
    star = " <-200d" if tw==200 else ""
    print(f"   {tw:4d}    {mtw['sharpe']:.3f}  {mtw['cagr']*100:6.2f}  {mtw['maxdd']*100:7.2f}  "
          f"{mtw['calmar']:.2f}   {mtw['worst1y']*100:7.2f}  {r2020*100:7.2f}  {r2008*100:7.2f}{star}")

ss=np.array(list(sweep_sharpe.values()))
print(f"\n  Sharpe across trend_win sweep: min={ss.min():.3f} max={ss.max():.3f} std={ss.std():.3f}")
print(f"  Is 200d a peak or a cliff-edge? 200d sharpe={sweep_sharpe[200]:.3f}, "
      f"neighbors 175={sweep_sharpe[175]:.3f} 225={sweep_sharpe[225]:.3f}")

# Crash protection specifically: max drawdown during 2020 & 2008 for each window
print("\n  Crash-window max drawdown (within-period) per trend_win:")
print("  trend_win   2020-mdd%   2008-mdd%")
for tw in [50,100,150,200,250,300]:
    rtw,_ = H.sleeve_a(df, fomc, trend_win=tw)
    for yr,sl in [("2020",slice("2020-01-01","2020-12-31")),("2008",slice("2008-01-01","2008-12-31"))]:
        pass
    def wdd(s):
        eq=(1+s).cumprod(); return float((eq/eq.cummax()-1).min())
    dd20=wdd(rtw.loc["2020-01-01":"2020-12-31"])
    dd08=wdd(rtw.loc["2008-01-01":"2008-12-31"])
    print(f"     {tw:4d}      {dd20*100:7.2f}    {dd08*100:7.2f}")

# ==================================================================
# BONUS: where does Sharpe come from? subperiod stability
# ==================================================================
print("\n" + "="*70)
print("BONUS: subperiod Sharpe stability of full Sleeve A (is edge persistent?)")
print("="*70)
for sl in [("1999-2007",slice("1999","2007")),("2008-2012",slice("2008","2012")),
           ("2013-2019",slice("2013","2019")),("2020-2026",slice("2020","2026"))]:
    seg=r_full.loc[sl[1]]
    if len(seg)>30:
        m=H.metrics(seg)
        print(f"   {sl[0]:12s} sharpe={m['sharpe']:6.3f} cagr={m['cagr']*100:7.2f}% mdd={m['maxdd']*100:7.2f}% n={m['n']}")

print("\nDONE.")
