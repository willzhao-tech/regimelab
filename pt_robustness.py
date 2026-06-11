# -*- coding: utf-8 -*-
"""
PARAM ROBUSTNESS attack on Sleeve A.

Chosen/published config: (parkinson, vol_win=21, trend_win=200, target=0.15, fomc_boost=0.5)
Question: is that a cherry-picked PEAK, or a robust PLATEAU?

Sweep:  vol_kind {parkinson, close}
      x vol_win {10, 21, 42, 63}
      x trend_win {100, 150, 200, 250, 300}
      x target {0.10, 0.15, 0.20}
= 2 * 4 * 5 * 3 = 120 configs.

For each: full Sleeve A (trend + voltarget + fomc), same cost/max_lev/rebal as published.
Report distribution of Sharpe, percentile rank of chosen config, worst config, a random config.
All signals causal (handled inside harness: .shift(1), monthly hold, trailing vol/trend).
"""
import sys, os
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
import sleeveA_harness as H

np.random.seed(12345)

df, fomc = H.load()
print("=== DATA ===")
print("rows:", len(df), " span:", df.index[0].date(), "->", df.index[-1].date())
print("FOMC scheduled dates kept:", len(fomc))
print()

# ---- published / chosen config ----
CHOSEN = dict(vol_kind="parkinson", vol_win=21, trend_win=200, target=0.15)

VOL_KINDS = ["parkinson", "close"]
VOL_WINS  = [10, 21, 42, 63]
TREND_WINS= [100, 150, 200, 250, 300]
TARGETS   = [0.10, 0.15, 0.20]

FIXED = dict(fomc_boost=0.5, cost=0.0005, max_lev=3.0,
             use_trend=True, use_voltarget=True, use_fomc=True)

rows = []
for vk in VOL_KINDS:
    for vw in VOL_WINS:
        for tw in TREND_WINS:
            for tg in TARGETS:
                r, pos = H.sleeve_a(df, fomc, vol_kind=vk, vol_win=vw,
                                    trend_win=tw, target=tg, **FIXED)
                m = H.metrics(r)
                rows.append(dict(vol_kind=vk, vol_win=vw, trend_win=tw, target=tg,
                                 sharpe=m["sharpe"], cagr=m["cagr"], maxdd=m["maxdd"],
                                 vol=m["vol"], calmar=m["calmar"], n=m["n"]))

res = pd.DataFrame(rows)
res_sorted = res.sort_values("sharpe", ascending=False).reset_index(drop=True)

sh = res["sharpe"].values
print("=== SHARPE DISTRIBUTION across", len(res), "configs (full Sleeve A) ===")
print(f"  mean  = {np.mean(sh):.4f}")
print(f"  std   = {np.std(sh, ddof=1):.4f}")
print(f"  min   = {np.min(sh):.4f}")
print(f"  p05   = {np.percentile(sh,5):.4f}")
print(f"  p25   = {np.percentile(sh,25):.4f}")
print(f"  median= {np.median(sh):.4f}")
print(f"  p75   = {np.percentile(sh,75):.4f}")
print(f"  p95   = {np.percentile(sh,95):.4f}")
print(f"  max   = {np.max(sh):.4f}")
print()

# ---- chosen config ----
mask = ((res.vol_kind==CHOSEN["vol_kind"]) & (res.vol_win==CHOSEN["vol_win"]) &
        (res.trend_win==CHOSEN["trend_win"]) & (res.target==CHOSEN["target"]))
chosen_row = res[mask].iloc[0]
chosen_sh = float(chosen_row["sharpe"])
# percentile rank: fraction of configs with sharpe <= chosen
pct_rank = float((sh <= chosen_sh).mean() * 100.0)
rank_from_top = int((res_sorted["sharpe"] > chosen_sh).sum()) + 1
print("=== CHOSEN config (parkinson,21,200,0.15) ===")
print(f"  sharpe        = {chosen_sh:.4f}")
print(f"  cagr          = {chosen_row['cagr']:.4f}")
print(f"  maxdd         = {chosen_row['maxdd']:.4f}")
print(f"  realized vol  = {chosen_row['vol']:.4f}")
print(f"  percentile rank = {pct_rank:.1f}%  (fraction of configs <= chosen)")
print(f"  rank          = {rank_from_top} of {len(res)} (1 = best)")
print()

# ---- worst config ----
worst = res_sorted.iloc[-1]
print("=== WORST config ===")
print(f"  {worst['vol_kind']}, vw={int(worst['vol_win'])}, tw={int(worst['trend_win'])}, tg={worst['target']}")
print(f"  sharpe = {worst['sharpe']:.4f}  cagr={worst['cagr']:.4f}  maxdd={worst['maxdd']:.4f}")
print()

# ---- best config ----
best = res_sorted.iloc[0]
print("=== BEST config ===")
print(f"  {best['vol_kind']}, vw={int(best['vol_win'])}, tw={int(best['trend_win'])}, tg={best['target']}")
print(f"  sharpe = {best['sharpe']:.4f}  cagr={best['cagr']:.4f}  maxdd={best['maxdd']:.4f}")
print()

# ---- a random config ----
ridx = np.random.randint(0, len(res))
rnd = res.iloc[ridx]
print("=== RANDOM config (seed 12345) ===")
print(f"  {rnd['vol_kind']}, vw={int(rnd['vol_win'])}, tw={int(rnd['trend_win'])}, tg={rnd['target']}")
print(f"  sharpe = {rnd['sharpe']:.4f}  cagr={rnd['cagr']:.4f}  maxdd={rnd['maxdd']:.4f}")
print()

# ---- how many configs clear common thresholds ----
print("=== FRACTION of configs clearing Sharpe thresholds ===")
for thr in [0.4, 0.5, 0.6, 0.65, 0.70]:
    frac = float((sh >= thr).mean()*100)
    print(f"  Sharpe >= {thr:.2f} : {frac:5.1f}%  ({int((sh>=thr).sum())}/{len(res)})")
print()

# ---- marginal sensitivity: hold all-but-one at chosen, vary one axis ----
print("=== MARGINAL one-axis sweeps (others fixed at chosen) ===")
def one_axis(axis, values):
    out=[]
    for v in values:
        cfg = dict(CHOSEN)
        cfg[axis]=v
        r,_=H.sleeve_a(df,fomc, vol_kind=cfg["vol_kind"], vol_win=cfg["vol_win"],
                       trend_win=cfg["trend_win"], target=cfg["target"], **FIXED)
        out.append((v, H.metrics(r)["sharpe"]))
    return out
for axis,vals in [("vol_kind",VOL_KINDS),("vol_win",VOL_WINS),
                  ("trend_win",TREND_WINS),("target",TARGETS)]:
    s = one_axis(axis, vals)
    print(f"  {axis:9s}: " + "  ".join(f"{v}={sh_:.3f}" for v,sh_ in s))
print()

# ---- top 10 and bottom 10 ----
print("=== TOP 10 configs ===")
print(res_sorted.head(10).to_string(index=False))
print()
print("=== BOTTOM 10 configs ===")
print(res_sorted.tail(10).to_string(index=False))
print()

# ---- save full table ----
outcsv = os.path.join(os.path.dirname(__file__), "pt_robustness_results.csv")
res_sorted.to_csv(outcsv, index=False)
print("Saved full table:", outcsv)
