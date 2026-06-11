# -*- coding: utf-8 -*-
"""
p0_instrument.py - P0-1 TRADABLE-INSTRUMENT VALIDATION.

The old proxy P&L (pos * (iv - rvar), 1-day variance at VXN strike) is untradable.
Here both signal families are re-run, with IDENTICAL signals/grids/walk-forward,
on a ROLLED 21-trading-day VARIANCE SWAP rolled daily, computable from VXN + NQ:

    V_t   = (VXN_t/100)^2          (annualized implied variance, fair strike proxy)
    rv1_t = 252 * ret_t^2          (annualized 1-day realized variance)
    SHORT-side daily pnl:
    pnl_short_t = (1/21)*(V_{t-1} - rv1_t) + (20/21)*(V_{t-1} - V_t)

The (20/21) mark-to-market VEGA term dominates the (1/21) carry/gamma term; this
is the real risk the 1-day proxy hides.

    pos_t  = s_{t-1}   (signal computed at close t-1, held day t)
    pnl_t  = pos_t * pnl_short_t - cost_t
    cost_t = |pos_t - pos_{t-1}| * 2*(VXN_{t-1}/100)*(0.5/100)   (0.5 vol-pt strike spread)

Sensitivity (favorable-assumption check): the spec cost charges ONLY on signal
changes, i.e. assumes the daily 1/21 roll is free. A "rollcost" variant adds
(1/21)*|pos| daily turnover at the same spread.

Walk-forward identical to harness: train=1260d, test=252d, pick best TRAIN
Sharpe, concatenate TEST blocks only. No caps/clips/tail edits anywhere.
Grids: A regime_combo 18 combos, B range_forecast 9 combos (same as proxy run).

Outputs side-by-side proxy vs rolled metrics + alpha regressions vs the static
short on the SAME instrument/dates. Comparison table ->
C:\\Users\\ASUS\\Desktop\\claude doc\\1\\p0_instrument_comparison.csv
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import volarb_harness as H

ART = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

df, ret, vxn = H.load()

# ---------------------------------------------------------------- instrument
V = (vxn / 100.0) ** 2          # annualized implied variance at close t
rv1 = 252.0 * ret ** 2          # annualized 1-day realized variance
PNL_SHORT = (1.0 / 21.0) * (V.shift(1) - rv1) + (20.0 / 21.0) * (V.shift(1) - V)
SPREAD_VAR = 2.0 * (vxn.shift(1) / 100.0) * (0.5 / 100.0)   # variance units per unit traded


def backtest_rolled(s, roll_cost=False):
    """Rolled 21d var-swap backtest. s computed with data through close t; shifted here."""
    pos = s.reindex(ret.index).clip(-1, 1).shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(0.0)
    if roll_cost:
        turn = turn + pos.abs() / 21.0     # pay spread on the daily 1/21 roll too
    cost = (SPREAD_VAR * turn).fillna(0.0)
    return (pos * PNL_SHORT - cost).dropna()


# ---------------------------------------------------------------- signals (UNCHANGED)
fc21 = H.fcast_vol(df, ret, "park21")
fc10 = H.fcast_vol(df, ret, "park10")
fc42 = H.fcast_vol(df, ret, "park42")
richness = vxn - fc21
trend = fc10 - fc42


def sig_A(g):
    r_hi, r_lo, d = g
    s = pd.Series(0.0, index=ret.index)
    s[((richness >= r_hi) & (trend <= -d)).fillna(False)] = 1.0
    s[((richness <= r_lo) & (trend >= d)).fillna(False)] = -1.0
    return s


rng = np.log(df["High"] / df["Low"]) * 100.0
be = vxn / SQ


def sig_B(g):
    b1, b2 = g
    s = pd.Series(0.0, index=ret.index)
    ok = rng.notna() & be.notna()
    s[ok & (rng < b1 * be)] = 1.0
    s[ok & (rng > b2 * be)] = -1.0
    return s


GRID_A = [(a, b, c) for a in (2.0, 4.0, 6.0) for b in (0.0, -2.0) for c in (0.0, 1.0, 2.0)]
GRID_B = [(b1, b2) for b1 in (0.8, 1.0, 1.2) for b2 in (1.3, 1.6, 2.0)]
print("combos: family A=%d, family B=%d (deflation base; blend is fixed 50/50, no extra fit)"
      % (len(GRID_A), len(GRID_B)))

STATIC = pd.Series(1.0, index=ret.index)


# ---------------------------------------------------------------- regressions
def alpha_reg(y, x, nw_lag=5):
    """OLS y = a + b*x; returns alpha, t_alpha(OLS), t_alpha(NW), beta."""
    yx = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    n = len(yx)
    X = np.column_stack([np.ones(n), yx["x"].values])
    yv = yx["y"].values
    bet, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ bet
    XtX_inv = np.linalg.inv(X.T @ X)
    covb = (resid @ resid) / (n - 2) * XtX_inv
    t_ols = bet[0] / np.sqrt(covb[0, 0])
    # Newey-West (Bartlett) on the score
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, nw_lag + 1):
        w = 1.0 - L / (nw_lag + 1.0)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    covb_nw = XtX_inv @ S @ XtX_inv
    t_nw = bet[0] / np.sqrt(covb_nw[0, 0])
    return float(bet[0]), float(t_ols), float(t_nw), float(bet[1])


def run_instrument(name, bt):
    """Walk-forward A, B, blend, static for one instrument backtest function bt(signal)."""
    oosA = H.walk_forward(lambda g: bt(sig_A(g)), GRID_A)
    oosB = H.walk_forward(lambda g: bt(sig_B(g)), GRID_B)
    common = oosA.index.intersection(oosB.index)
    blend = 0.5 * oosA.reindex(common) + 0.5 * oosB.reindex(common)
    static_full = bt(STATIC)
    rows = []
    for fam, pnl in [("A_regime_combo", oosA), ("B_range_forecast", oosB),
                     ("blend_50_50", blend), ("static_short", static_full.reindex(common).dropna())]:
        m = H.metrics(pnl)
        if fam == "static_short":
            a, t_o, t_n, b = float("nan"), float("nan"), float("nan"), 1.0
        else:
            a, t_o, t_n, b = alpha_reg(pnl, static_full.reindex(pnl.index))
        rows.append(dict(instrument=name, family=fam,
                         sharpe=round(m["sharpe"], 3), tstat=round(m["tstat"], 2),
                         skew=round(m["skew"], 2), n=m["n"],
                         ann_mean=float(pnl.mean() * 252), ann_vol=float(pnl.std() * SQ),
                         alpha_daily=a, t_alpha_ols=round(t_o, 2) if t_o == t_o else float("nan"),
                         t_alpha_nw5=round(t_n, 2) if t_n == t_n else float("nan"),
                         beta=round(b, 3),
                         worst_ratio=round(m["worst_ratio"], 1),
                         oos_start=str(pnl.index[0].date()), oos_end=str(pnl.index[-1].date())))
    picks = dict(A=Counter(oosA.attrs["picks"]), B=Counter(oosB.attrs["picks"]))
    return rows, picks


# ---------------------------------------------------------------- run all three
all_rows = []
proxy_rows, proxy_picks = run_instrument("proxy_1d", lambda s: H.backtest(s, ret, vxn))
rolled_rows, rolled_picks = run_instrument("rolled_21d", lambda s: backtest_rolled(s, roll_cost=False))
rollc_rows, rollc_picks = run_instrument("rolled_21d_rollcost", lambda s: backtest_rolled(s, roll_cost=True))
all_rows = proxy_rows + rolled_rows + rollc_rows

tab = pd.DataFrame(all_rows)
out_csv = os.path.join(ART, "p0_instrument_comparison.csv")
tab.to_csv(out_csv, index=False)

pd.set_option("display.width", 250)
print("\n=== side-by-side comparison (walk-forward OOS only; train numbers never reported) ===")
print(tab[["instrument", "family", "sharpe", "tstat", "skew", "n",
           "t_alpha_ols", "t_alpha_nw5", "beta", "worst_ratio",
           "oos_start", "oos_end"]].to_string(index=False))
print("\nsaved -> %s" % out_csv)

# ---------------------------------------------------------------- diagnostics
print("\n=== instrument diagnostics (full sample, short side, before costs) ===")
gamma_term = ((1.0 / 21.0) * (V.shift(1) - rv1)).dropna()
vega_term = ((20.0 / 21.0) * (V.shift(1) - V)).dropna()
print("  std gamma/carry term (1/21):    %.3e (daily var units)" % gamma_term.std())
print("  std vega MTM term   (20/21):    %.3e" % vega_term.std())
print("  vega/gamma vol ratio:           %.2f" % (vega_term.std() / gamma_term.std()))
ps = PNL_SHORT.dropna()
print("  rolled short-side: ann Sharpe %.2f, skew %.1f"
      % (ps.mean() / ps.std() * SQ, ps.skew()))
prox = ((vxn.shift(1) / 100.0) ** 2 / 252.0 - ret ** 2).dropna()
ci = prox.index.intersection(ps.index)
print("  corr(rolled short pnl, proxy short pnl): %.3f" % ps.reindex(ci).corr(prox.reindex(ci)))

for nm, pk in [("proxy", proxy_picks), ("rolled", rolled_picks), ("rolled+rollcost", rollc_picks)]:
    print("\n  %s walk-forward picks:" % nm)
    for fam in ("A", "B"):
        tops = pk[fam].most_common(3)
        print("    %s top picks: %s   (blocks=%d)" % (fam, tops, sum(pk[fam].values())))

# ---------------------------------------------------------------- look-ahead audit
print("\n=== causality audit (rolled instrument) ===")
print("  PNL_SHORT_t uses V_{t-1}, V_t, ret_t -> pnl at t needs no future data: by construction.")
print("  pos_t = s_{t-1} via shift(1) inside backtest_rolled; signals identical to proxy run")
print("  (already truncation-tested in va_regime_combo.py / va_range_forecast.py).")
cut = len(ret) - 50
p_full = backtest_rolled(sig_A(GRID_A[0]))
df2, ret2, vxn2 = df.iloc[:cut], ret.iloc[:cut], vxn.iloc[:cut]
V2 = (vxn2 / 100.0) ** 2
rv12 = 252.0 * ret2 ** 2
ps2 = (1 / 21.0) * (V2.shift(1) - rv12) + (20 / 21.0) * (V2.shift(1) - V2)
s2 = pd.Series(0.0, index=ret2.index)
f21b = H.fcast_vol(df2, ret2, "park21"); f10b = H.fcast_vol(df2, ret2, "park10"); f42b = H.fcast_vol(df2, ret2, "park42")
r2 = vxn2 - f21b; t2 = f10b - f42b
r_hi, r_lo, d = GRID_A[0]
s2[((r2 >= r_hi) & (t2 <= -d)).fillna(False)] = 1.0
s2[((r2 <= r_lo) & (t2 >= d)).fillna(False)] = -1.0
pos2 = s2.clip(-1, 1).shift(1).fillna(0.0)
cost2 = (2.0 * (vxn2.shift(1) / 100.0) * 0.005 * pos2.diff().abs().fillna(0.0)).fillna(0.0)
p_trunc = (pos2 * ps2 - cost2).dropna()
ov = p_trunc.index[-30:]
same = bool(np.allclose(p_trunc.reindex(ov).values, p_full.reindex(ov).values, atol=1e-15))
print("  truncation test (pnl unchanged when future removed): %s" % ("PASS" if same else "FAIL"))

print("\nHARD-RULES note: all numbers above are concatenated TEST blocks (walk-forward);")
print("params only from trailing 1260d train Sharpe; no tail edits; costs as specified;")
print("rollcost rows show what breaks if the free-daily-roll assumption fails.")
