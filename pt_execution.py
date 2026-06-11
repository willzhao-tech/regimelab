# -*- coding: utf-8 -*-
"""
EXECUTION-REALISM ATTACK on Sleeve A (vol-targeted, 200d-trend-braked, FOMC-tilted long NQ).
Question: does the ~0.70 Sharpe and the maxDD-edge-over-NQ survive realistic frictions?
At what cost does the edge over vol-matched NQ vanish?

Attack axes:
  - cost in {2,5,10,20,30} bps (per unit of |position change|)
  - exec_lag in {1,2}  (position decided t-lag, applied to ret_t)
  - rebal in {D,W,M}
  - turnover/yr and capacity (NQ futures ADV * mult $20)

We compare Sleeve A vs a VOL-MATCHED long-NQ benchmark (the "is it just beta?" control),
scaling NQ so its realized vol equals Sleeve A's realized vol over the SAME sample, at the
SAME cost. The "edge" = Sleeve_Sharpe - VolMatchedNQ_Sharpe, and maxDD edge = NQ_mdd - Sleeve_mdd
(positive = Sleeve better).
"""
import sys, os
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
import sleeveA_harness as H

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

df, fomc = H.load()
SQ = np.sqrt(252.0)
MULT = 20.0  # NQ futures $ multiplier


def vol_match_nq(df, ref_r, cost, rebal, exec_lag):
    """
    Build a vol-matched long-NQ benchmark on the SAME sample as ref_r.
    Plain long NQ at constant position k, k chosen so realized vol == ref_r realized vol.
    Cost applied on the (tiny) turnover of a constant position -> essentially the initial ramp,
    so we instead just scale raw NQ returns (constant pos => zero ongoing turnover, zero cost).
    To be FAIR we also let NQ pay cost on any rebal it would do: a constant long has 0 turnover,
    so NQ benchmark cost ~ 0. That biases IN FAVOUR of NQ benchmark (harder bar for Sleeve), which
    is what we want for an honest 'edge' test.
    """
    ret = df["Close"].pct_change()
    # align to ref sample
    ret = ret.reindex(ref_r.index).dropna()
    ref = ref_r.reindex(ret.index).dropna()
    ret = ret.reindex(ref.index)
    base_vol = ret.std()
    target_vol = ref.std()
    k = target_vol / base_vol if base_vol > 0 else 0.0
    nq_r = (k * ret)
    return nq_r


def turnover_stats(pos):
    turn = pos.diff().abs().fillna(0.0)
    yrs = (pos.index[-1] - pos.index[0]).days / 365.25
    return float(turn.sum() / yrs), float(turn.sum()), yrs


def run_cell(cost_bps, exec_lag, rebal):
    cost = cost_bps / 1e4
    r, pos = H.sleeve_a(df, fomc, cost=cost, rebal=rebal, exec_lag=exec_lag)
    m = H.metrics(r)
    nq_r = vol_match_nq(df, r, cost, rebal, exec_lag)
    mnq = H.metrics(nq_r)
    tpy, ttot, yrs = turnover_stats(pos)
    edge_sharpe = m["sharpe"] - mnq["sharpe"]
    # maxDD edge: positive means Sleeve has SHALLOWER drawdown than vol-matched NQ
    edge_mdd = mnq["maxdd"] - m["maxdd"]   # both negative; (nq - sleeve): if sleeve shallower (closer to 0), this is +
    return dict(cost_bps=cost_bps, lag=exec_lag, rebal=rebal,
                sharpe=m["sharpe"], vol=m["vol"], cagr=m["cagr"], maxdd=m["maxdd"],
                nq_sharpe=mnq["sharpe"], nq_maxdd=mnq["maxdd"],
                edge_sharpe=edge_sharpe, edge_mdd=edge_mdd,
                turn_per_yr=tpy)


print("=" * 110)
print("PART 1: COST x EXEC_LAG x REBAL SWEEP  (Sleeve A vs vol-matched long NQ on same sample)")
print("=" * 110)
rows = []
for rebal in ["D", "W", "M"]:
    for lag in [1, 2]:
        for cb in [2, 5, 10, 20, 30]:
            rows.append(run_cell(cb, lag, rebal))
res = pd.DataFrame(rows)
for rebal in ["D", "W", "M"]:
    print(f"\n--- rebal = {rebal} ---")
    sub = res[res.rebal == rebal].copy()
    print(sub[["cost_bps", "lag", "sharpe", "vol", "maxdd", "nq_sharpe", "nq_maxdd",
               "edge_sharpe", "edge_mdd", "turn_per_yr"]].round(4).to_string(index=False))

print("\n" + "=" * 110)
print("PART 2: TURNOVER / YEAR by rebal (units of |dPos|, ~ round-trip notional fraction)")
print("=" * 110)
for rebal in ["D", "W", "M"]:
    _, pos = H.sleeve_a(df, fomc, rebal=rebal, exec_lag=1)
    tpy, ttot, yrs = turnover_stats(pos)
    print(f"rebal={rebal}: turnover/yr={tpy:7.2f}  total={ttot:8.2f} over {yrs:.1f}y  avg_pos={pos.mean():.3f}")

print("\n" + "=" * 110)
print("PART 3: AT WHAT COST DOES THE EDGE OVER VOL-MATCHED NQ VANISH? (fine cost grid, lag=1, rebal=M)")
print("=" * 110)
fine = []
for cb in [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
    fine.append(run_cell(cb, 1, "M"))
fdf = pd.DataFrame(fine)
print(fdf[["cost_bps", "sharpe", "edge_sharpe", "edge_mdd", "maxdd", "nq_maxdd"]].round(4).to_string(index=False))
# breakeven on sharpe edge
neg = fdf[fdf.edge_sharpe < 0]
if len(neg):
    print(f"\n>> Sharpe edge over vol-matched NQ first goes NEGATIVE at cost ~ {int(neg.cost_bps.iloc[0])} bps (rebal=M, lag=1).")
else:
    print("\n>> Sharpe edge over vol-matched NQ stays POSITIVE across entire cost grid (rebal=M, lag=1).")

print("\n" + "=" * 110)
print("PART 4: Same breakeven for DAILY rebal (worst turnover case), lag=1 and lag=2")
print("=" * 110)
for lag in [1, 2]:
    fine = []
    for cb in [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]:
        fine.append(run_cell(cb, lag, "D"))
    fdf = pd.DataFrame(fine)
    print(f"\n-- DAILY rebal, lag={lag} --")
    print(fdf[["cost_bps", "sharpe", "edge_sharpe", "edge_mdd", "turn_per_yr"]].round(4).to_string(index=False))
    neg = fdf[fdf.edge_sharpe < 0]
    if len(neg):
        print(f">> edge vanishes at ~{int(neg.cost_bps.iloc[0])} bps")
    else:
        print(">> edge positive across grid")

print("\n" + "=" * 110)
print("PART 5: CAPACITY from NQ ADV (Volume col, mult $20). How big can this sleeve be?")
print("=" * 110)
vol_contracts = df["Volume"].copy()
px = df["Close"]
# $ ADV = contracts/day * price * multiplier
dollar_adv = (vol_contracts * px * MULT)
# recent window (last 1y, 2y, 5y)
for label, n in [("last 1y", 252), ("last 2y", 504), ("last 5y", 1260), ("full", len(df))]:
    sub = dollar_adv.iloc[-n:]
    contr = vol_contracts.iloc[-n:]
    print(f"{label:8s}: median $ADV = ${sub.median()/1e9:8.2f}B  median contracts/day = {contr.median():12,.0f}  median px={px.iloc[-n:].median():.0f}")

# Turnover in $ terms: a fund of size $F running this sleeve trades, per rebalance event,
# |dPos| * F notional. Daily $ traded = turnover/yr * F / 252 (avg). Capacity rule of thumb:
# keep your trade < X% of ADV. Use M-rebal turnover.
_, pos = H.sleeve_a(df, fomc, rebal="M", exec_lag=1)
tpy, ttot, yrs = turnover_stats(pos)
adv_recent = dollar_adv.iloc[-504:].median()  # last ~2y
print(f"\nM-rebal turnover/yr = {tpy:.2f} (notional units).")
# On a rebal day the position can jump by up to ~max_lev. Look at the distribution of single-day |dPos|.
turn = pos.diff().abs().fillna(0.0)
big = turn[turn > 1e-9]
print(f"Per-trade |dPos| (nonzero days): mean={big.mean():.3f} median={big.median():.3f} p95={big.quantile(.95):.3f} max={big.max():.3f}")
print(f"Recent median $ADV (last 2y) = ${adv_recent/1e9:.2f}B")
for pct in [0.01, 0.05, 0.10]:
    # max single-trade notional = pct * ADV ; that must cover |dPos|_p95 * F
    cap_F = (pct * adv_recent) / big.quantile(.95)
    print(f"  If single trade kept <= {pct*100:.0f}% of $ADV and trade = p95 |dPos|*F  -> fund capacity F <= ${cap_F/1e9:6.2f}B")

print("\n" + "=" * 110)
print("PART 6: SANITY -- where does the Sharpe come from? decompose vs controls at realistic 10bps, lag=1, M")
print("=" * 110)
cost = 0.0010
def mm(use_trend, use_vt, use_fomc, lag=1, rebal="M"):
    r, pos = H.sleeve_a(df, fomc, cost=cost, rebal=rebal, exec_lag=lag,
                        use_trend=use_trend, use_voltarget=use_vt, use_fomc=use_fomc)
    m = H.metrics(r)
    tpy, _, _ = turnover_stats(pos)
    return m["sharpe"], m["vol"], m["maxdd"], tpy
configs = [
    ("full (trend+vt+fomc)", True, True, True),
    ("no fomc (trend+vt)",   True, True, False),
    ("no trend (vt+fomc)",   False, True, True),
    ("vt only",              False, True, False),
    ("trend only",           True, False, False),
    ("raw long (none)",      False, False, False),
]
print(f"{'config':28s} {'sharpe':>8s} {'vol':>7s} {'maxdd':>8s} {'turn/yr':>8s}")
for name, t, v, f in configs:
    s, vol, mdd, tpy = mm(t, v, f)
    print(f"{name:28s} {s:8.3f} {vol:7.3f} {mdd:8.3f} {tpy:8.2f}")

print("\nDONE.")
