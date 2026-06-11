# -*- coding: utf-8 -*-
"""Example 50 - does the vol-TIMING strategy work on NON-EQUITY underlyings? WTI/OVX, XAU/GVZ, EUR/EVZ.
IDENTICAL methodology to the equity book: 1-DTE straddle instrument, A/B regime signals, walk-forward
1260/252, FULL L4 frictions. (Earlier proxy-space finding: equity 9/9 pass, non-equity 0/3 fail -
re-test with the corrected straddle instrument.) Compares each to the equity floor, and builds a
3-market non-equity floor book. k=0.82 is equity-measured (extrapolated; scales blend & static alike,
so alpha-t vs same-friction static is the friction-robust 'does timing add value' read)."""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
import bookopt_harness as H, bookopt_floor as F

SQ = H.SQ
NONEQ = [("WTI","WTI","OVX",.03), ("XAU","XAU","GVZ",.03), ("EUR","EURUSD","EVZ",.02)]
H.PAIRS = H.PAIRS + NONEQ          # extend the universe; _load() will pick these up
H._DATA.clear()

def alpha_t(blend, static):
    yx = pd.concat([blend, static], axis=1).dropna().values
    if len(yx) < 100: return np.nan
    y, x = yx[:,0], yx[:,1]; X = np.column_stack([np.ones(len(x)), x])
    b,*_ = np.linalg.lstsq(X, y, rcond=None); r = y - X@b
    return float(b[0]/np.sqrt((r@r/(len(y)-2))*np.linalg.inv(X.T@X)[0,0]))

print("NON-EQUITY vol-timing (same harness as the 8-market equity book)\n")
print(f"{'mkt':<5}{'start':>11}{'n':>6}{'blendSh':>9}{'staticSh':>10}{'alpha-t':>9}{'days L/S':>10}  verdict")
neq_sleeves = {}
for name, uf, vf, sp in NONEQ:
    blend, static, info = H.market(name, return_pos=True)
    if blend is None:
        print(f"{name:<5}  insufficient data"); continue
    neq_sleeves[name] = blend
    sh = H.sharpe(blend); ssh = H.sharpe(static); at = alpha_t(blend, static)
    netpos = 0.5*info["posA"] + 0.5*info["posB"]
    dl = (netpos < 0).mean()*100; ds = (netpos > 0).mean()*100   # days net-long-vol / net-short-vol
    # PASS needs genuine TIMING alpha (alpha-t>2). A high Sharpe with alpha-t<=1 is a pricing
    # artifact (always-one-side winning because equity-k mis-prices this market's premium), NOT skill.
    if at > 2.0 and sh > 0.5:      verdict = "PASS (timing)"
    elif sh > 0.5 and at <= 1.0:   verdict = "ARTIFACT (no timing; k-mispriced)"
    else:                          verdict = "FAIL"
    print(f"{name:<5}{str(blend.index.min().date()):>11}{len(blend):>6}{sh:>9.2f}{ssh:>10.2f}{at:>9.1f}"
          f"{dl:>5.0f}/{ds:<4.0f}  {verdict}")

# reference: equity floor on its own (already known ~1.26)
print("\nfor reference, the 8-market EQUITY floor book: active Sharpe ~1.26 / calendar ~1.01")

# mini non-equity floor book (invvol x coverage across the 3) ----------------------------
if len(neq_sleeves) >= 2:
    Wn = {n: F.invvol(neq_sleeves[n])*F.coverage_gate(n).reindex(neq_sleeves[n].index) for n in neq_sleeves}
    nbook = H.book_of(neq_sleeves, Wn)
    ai = pd.DatetimeIndex(sorted(set().union(*[set(neq_sleeves[k].index) for k in neq_sleeves])))
    ncal = nbook.reindex(ai).fillna(0.0)
    st = F.stat_line(nbook)
    print(f"\n3-MARKET NON-EQUITY floor book: active Sharpe {H.sharpe(nbook):.2f}  calendar {H.sharpe(ncal):.2f}  "
          f"maxDD {st['maxdd']*100:.0f}%  n {len(nbook)}")

print("\nVERDICT: read the per-market alpha-t (timing value vs same-friction static) and Sharpe vs the")
print("equity ~1.26. PASS requires the regime signal to predict next-day realized variance in that market.")
