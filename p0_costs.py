# -*- coding: utf-8 -*-
"""
p0_costs.py -- TASK P0-2: refined cost model on the NQ/VXN variance proxy, BLEND strategy.

Cost scenarios (cumulative ladder):
  1. old_fixed      : spread = 0.5 vol-pt (original harness assumption)
  2. dynamic        : spread = 0.3 + 0.04*VXN_(t-1) vol-pts  (~1.1pt @ VXN 20, ~2.7pt @ VXN 60)
  3. dyn+slip       : + extra HALF-spread on turnover executed on stress days
                      (stress: execution-day range > 2 * trailing 252d median range, both causal)
  4. dyn+slip+fund  : + margin funding 4%/252 on 15% margin per unit notional HELD daily

Then a spread-multiplier sweep on scenario 4 to locate where alpha-t vs static < 2.

CAUSALITY / HONESTY NOTES
  - All cost inputs are trailing: VXN_(t-1) for spread width; stress flag at cost-day t uses
    range of day t-1 (the execution day, since pos_t = s_(t-1) is traded at close t-1) vs the
    rolling-252 median of range through day t-2 (median .shift(1)). No full-sample stats.
  - Costs change TRAIN Sharpes, so the walk-forward is RE-RUN per scenario per family
    (grid A = 18 combos, grid B = 9 combos, picks via trailing 1260d train -> 252d test only).
  - Static baseline (s=+1 always) is computed UNDER THE SAME cost scenario so the timing-alpha
    regression compares like with like.
  - No caps / clips / tail edits anywhere. OOS test blocks only are concatenated and reported.
  - Instrument is still the 1-day variance PROXY (not tradable); levels are inflated for
    strategy and baseline alike -- the robust read is the ladder SLOPE and alpha-t.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import volarb_harness as H

OUT = r"C:\Users\ASUS\Desktop\claude doc\1\p0_costs_ladder.csv"
SQ = np.sqrt(252.0)
TRAIN, TEST = 1260, 252
RF, MARGIN = 0.04, 0.15

df, ret, vxn = H.load()
iv, rvar = H.iv_rvar(ret, vxn)
v1 = vxn.shift(1)

# ---- causal stress flag for slippage ----
# range of day u vs trailing median through u-1; then shift(1) so the flag sits on cost day
# t = u+1, whose turnover |dpos_t| was executed at close of day u (pos_t = s_(t-1)).
rng_log = np.log(df["High"] / df["Low"])
med252 = rng_log.rolling(252, min_periods=126).median().shift(1)
stress_exec = (rng_log > 2.0 * med252).fillna(False)          # flag on execution day u
stress_cost = stress_exec.shift(1).fillna(False).astype(float)  # aligned to cost day t

print("stress-day frequency (executions flagged): %.2f%% of days"
      % (100.0 * stress_exec.mean()))


def cost_components(pos, spread_mode, slippage, funding, mult=1.0):
    """Return (total_cost, spread_cost, funding_cost) per day, all causal."""
    dpos = pos.diff().abs().fillna(0.0)
    if spread_mode == "fixed":
        sp = pd.Series(0.5, index=pos.index)
    else:
        sp = 0.3 + 0.04 * v1
    sp = sp * mult
    if slippage:
        sp = sp * (1.0 + 0.5 * stress_cost)   # extra half-spread on stress executions
    spread_cost = (2.0 * v1 * sp / 1e4 / 252.0).fillna(0.0) * dpos
    fund_cost = (RF / 252.0 * MARGIN) * pos.abs() if funding else pd.Series(0.0, index=pos.index)
    return spread_cost + fund_cost, spread_cost, fund_cost


def backtest_c(s, **cm):
    pos = s.reindex(ret.index).clip(-1, 1).shift(1).fillna(0.0)
    tot, _, _ = cost_components(pos, **cm)
    return (pos * (iv - rvar) - tot).dropna()


# ---- families (identical to va_regime_combo.py / va_range_forecast.py) ----
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


rng_pct = rng_log * 100.0
be = vxn / SQ


def sig_B(g):
    b1, b2 = g
    s = pd.Series(0.0, index=ret.index)
    ok = rng_pct.notna() & be.notna()
    s[ok & (rng_pct < b1 * be)] = 1.0
    s[ok & (rng_pct > b2 * be)] = -1.0
    return s


GRID_A = [(a, b, c) for a in (2.0, 4.0, 6.0) for b in (0.0, -2.0) for c in (0.0, 1.0, 2.0)]
GRID_B = [(b1, b2) for b1 in (0.8, 1.0, 1.2) for b2 in (1.3, 1.6, 2.0)]
print("combos: family A = %d, family B = %d (deflation base = %d per scenario; "
      "cost scenarios are fixed ex-ante by task spec, not searched)"
      % (len(GRID_A), len(GRID_B), len(GRID_A) + len(GRID_B)))


def collect_blocks(fn_of_g, picks, pnl_idx):
    """Reindex per-block full-series values of the picked param onto OOS test blocks."""
    parts, start, b = [], TRAIN, 0
    while start + TEST <= len(pnl_idx):
        te = pnl_idx[start:start + TEST]
        parts.append(fn_of_g(picks[b]).reindex(te))
        start += TEST
        b += 1
    return pd.concat(parts)


def alpha_t(y, x):
    yx = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    Y, Xv = yx["y"].values, yx["x"].values
    X = np.column_stack([np.ones(len(Xv)), Xv])
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    covb = (resid @ resid) / (len(Y) - 2) * np.linalg.inv(X.T @ X)
    return float(beta[0]), float(beta[0] / np.sqrt(covb[0, 0])), float(beta[1])


def run_scenario(name, cm):
    """Re-run walk-forward for A and B under cost model cm; blend 0.5/0.5; same-cost static."""
    res_fam = {}
    for fam, sig_fn, grid in (("A", sig_A, GRID_A), ("B", sig_B, GRID_B)):
        oos = H.walk_forward(lambda g: backtest_c(sig_fn(g), **cm), grid, TRAIN, TEST)
        picks = oos.attrs["picks"]
        pnl_idx = backtest_c(sig_fn(grid[0]), **cm).index

        def posf(g, sig_fn=sig_fn):
            return sig_fn(g).reindex(ret.index).clip(-1, 1).shift(1).fillna(0.0)

        oos_cost = collect_blocks(lambda g: cost_components(posf(g), **cm)[0], picks, pnl_idx)
        oos_turn = collect_blocks(lambda g: posf(g).diff().abs().fillna(0.0), picks, pnl_idx)
        res_fam[fam] = dict(oos=oos, picks=picks, cost=oos_cost, turn=oos_turn)

    A, B = res_fam["A"], res_fam["B"]
    blend = (0.5 * A["oos"] + 0.5 * B["oos"]).dropna()
    blend_cost = (0.5 * A["cost"] + 0.5 * B["cost"]).reindex(blend.index)
    blend_gross = blend + blend_cost
    turn_leg = (0.5 * A["turn"] + 0.5 * B["turn"]).reindex(blend.index)

    static = backtest_c(pd.Series(1.0, index=ret.index), **cm).reindex(blend.index).dropna()
    a, ta, bcoef = alpha_t(blend, static)

    mA, mB, mbl, mst = (H.metrics(x) for x in (A["oos"], B["oos"], blend, static))
    yrs = len(blend) / 252.0
    row = dict(
        scenario=name,
        spread_mult=cm.get("mult", 1.0),
        sharpe_A=round(mA["sharpe"], 3), sharpe_B=round(mB["sharpe"], 3),
        sharpe_blend=round(mbl["sharpe"], 3), tstat_blend=round(mbl["tstat"], 2),
        skew_blend=round(mbl["skew"], 2),
        sharpe_static=round(mst["sharpe"], 3), skew_static=round(mst["skew"], 2),
        alpha_daily=a, alpha_t_vs_static=round(ta, 2), beta_vs_static=round(bcoef, 3),
        blend_gross_ann=float(blend_gross.mean() * 252),
        blend_cost_ann=float(blend_cost.mean() * 252),
        cost_pct_of_gross=round(100.0 * blend_cost.sum() / blend_gross.sum(), 1)
        if blend_gross.sum() != 0 else float("nan"),
        ann_turnover_units=round(float(turn_leg.sum() / yrs), 1),
        n_days=len(blend),
        oos_start=str(blend.index[0].date()), oos_end=str(blend.index[-1].date()),
    )
    print("\n--- %s ---" % name)
    print("  A %.3f | B %.3f | blend %.3f (skew %+.1f) | static %.3f (skew %+.1f) | "
          "alpha-t %.2f | cost %.1f%% of gross"
          % (row["sharpe_A"], row["sharpe_B"], row["sharpe_blend"], row["skew_blend"],
             row["sharpe_static"], row["skew_static"], row["alpha_t_vs_static"],
             row["cost_pct_of_gross"]))
    from collections import Counter
    for fam in ("A", "B"):
        top = Counter(res_fam[fam]["picks"]).most_common(2)
        print("  picks %s (top2 of %d blocks): %s" % (fam, len(res_fam[fam]["picks"]), top))
    return row


SCEN = [
    ("old_fixed_0.5pt", dict(spread_mode="fixed", slippage=False, funding=False)),
    ("dynamic_spread", dict(spread_mode="dyn", slippage=False, funding=False)),
    ("dynamic+slippage", dict(spread_mode="dyn", slippage=True, funding=False)),
    ("dynamic+slippage+funding", dict(spread_mode="dyn", slippage=True, funding=True)),
]
rows = [run_scenario(name, cm) for name, cm in SCEN]

# ---- spread-multiplier sweep on the FULL model to find where alpha-t < 2 ----
print("\n=== spread-multiplier sweep (full model: dynamic+slippage+funding) ===")
sweep_rows = []
for k in (1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
    cm = dict(spread_mode="dyn", slippage=True, funding=True, mult=k)
    sweep_rows.append(run_scenario("sweep_full_x%.1f" % k, cm))

ladder = pd.DataFrame(rows + sweep_rows)
ladder.to_csv(OUT, index=False)
print("\nsaved ladder -> %s" % OUT)

# ---- where does alpha-t drop below 2? ----
print("\n=== alpha-t breakdown point ===")
base_ok = all(r["alpha_t_vs_static"] >= 2.0 for r in rows)
print("base ladder alpha-t: %s"
      % ", ".join("%s=%.2f" % (r["scenario"], r["alpha_t_vs_static"]) for r in rows))
crossed = None
prev = ("x1.0 (full model)", rows[-1]["alpha_t_vs_static"])
for r in sweep_rows:
    if r["alpha_t_vs_static"] < 2.0:
        crossed = (r, prev)
        break
    prev = (r["scenario"], r["alpha_t_vs_static"])
if not base_ok:
    first = next(r for r in rows if r["alpha_t_vs_static"] < 2.0)
    print("alpha-t drops below 2 already at base scenario: %s" % first["scenario"])
elif crossed:
    r, prev = crossed
    k = r["spread_mult"]
    print("alpha-t < 2 first at spread multiplier x%.1f (alpha-t %.2f; previous rung %s had %.2f)"
          % (k, r["alpha_t_vs_static"], prev[0], prev[1]))
    print("  effective half-spread at that rung: %.2f vol-pts @ VXN20, %.2f @ VXN60 "
          "(plus 50%% more on stress days)"
          % (k * (0.3 + 0.04 * 20), k * (0.3 + 0.04 * 60)))
else:
    print("alpha-t stays >= 2 through x8.0 dynamic spread (%.2f at x8.0)"
          % sweep_rows[-1]["alpha_t_vs_static"])
