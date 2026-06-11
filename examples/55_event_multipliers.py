# -*- coding: utf-8 -*-
"""Example 55 - EVENT-DAY MOVE MULTIPLIERS on our 8-market panel (1999/2005-2026).
Re-measures the macro-calendar finding (FOMC/NFP ~2x big-move probability) on OUR panel before
building anything on it. Per market x event: mean|ret| multiplier vs non-event baseline, at
D0 (first trading day >= event; US markets react same-day) and D+1 (overnight landing for
Asia/Europe). Welch t-test on |ret|. Plus the E1 trap zone: PRE-event day stats.
Writes event_multipliers.csv (consumed by the interactive calendar + future lab families)."""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
from scipy import stats
import bookopt_harness as H
from market_calendar import event_flags

OUT = H.OUT
H._load()

def welch_t(a, b):
    if len(a) < 8 or len(b) < 8: return np.nan
    return float(stats.ttest_ind(a, b, equal_var=False).statistic)

rows = []
print("EVENT-DAY |MOVE| MULTIPLIERS (x baseline mean|ret|)  D0 = event trading day, D1 = next day")
print(f"{'mkt':<6}{'n':>6} | {'FOMC D0':>9}{'D1':>7} | {'NFP D0':>9}{'D1':>7} | {'CPI D0':>9}{'D1':>7} |{'preEvt':>8}")
for name, uf, vf, sp in H.PAIRS:
    ret = H._DATA[name][1].dropna()
    fl = event_flags(ret.index)
    ab = ret.abs()
    base = ab[(fl["any"] == 0) & (fl["any_next"] == 0) & (fl["pre_any"] == 0)]
    line = f"{name:<6}{len(ret):>6} |"
    for ev in ("fomc", "nfp", "cpi"):
        d0 = ab[fl[ev] == 1]
        d1 = ab[fl[ev].shift(1).fillna(0) == 1]
        m0, m1 = d0.mean()/base.mean(), d1.mean()/base.mean()
        t0, t1 = welch_t(d0, base), welch_t(d1, base)
        rows.append(dict(market=name, event=ev.upper(), mult_d0=round(m0, 3), t_d0=round(t0, 1),
                         mult_d1=round(m1, 3), t_d1=round(t1, 1), n_event=len(d0)))
        f0 = "*" if abs(t0) > 2.5 else " "
        f1 = "*" if abs(t1) > 2.5 else " "
        line += f"{m0:>8.2f}{f0}{m1:>6.2f}{f1} |"
    pre = ab[fl["pre_any"] == 1]
    mp = pre.mean()/base.mean()
    rows.append(dict(market=name, event="PRE_ANY", mult_d0=round(mp, 3),
                     t_d0=round(welch_t(pre, base), 1), mult_d1=np.nan, t_d1=np.nan, n_event=len(pre)))
    print(line + f"{mp:>7.2f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "event_multipliers.csv"), index=False)
print("\n* = Welch t > 2.5 vs non-event baseline")
print("READ: D0 = where US markets load; D1 = where Asia/Europe load FOMC (announced overnight")
print("for them). preEvt < 1 would mean pre-event days are QUIET (the calm before) — relevant to")
print("E1: family A reads pre-event IV premium as 'richness' exactly when realized vol is subdued.")
print(f"\nsaved -> event_multipliers.csv ({len(df)} rows)")
