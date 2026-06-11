# -*- coding: utf-8 -*-
"""
Strategy B (short-vol) tail-protection optimizer: OPTIMAL WING STRIKE via WALK-FORWARD.

Idea: short variance, but cap the daily loss at a wing strike `cap` (you own OTM wings
that you PAY for via a trailing, smile-marked-up wing cost). Tighter cap = more tail
protection but higher wing cost. We do NOT pick the cap on the full sample: we grid over
cap x cost_win and let H.walk_forward() pick the OOS-best combo on a trailing-train Sharpe,
applied to the next out-of-sample test block, rolling forward.

All signals are causal (the harness's hedged_pnl uses vxn.shift(1), trailing rolling wing
cost .shift(1), and causal_scale uses trailing std .shift(1)). See the self-audit at bottom.
"""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import shortvol_harness as H
import numpy as np

# ---- load causal inputs -----------------------------------------------------
df, ret, vxn = H.load()

# ---- grid: cap (wing strike) x cost_win, smile fixed = 1.5 ------------------
SMILE = 1.5
caps = [0.03, 0.04, 0.05, 0.07, 0.10]
cost_wins = [126, 252, 504]
grid = [dict(cap=c, cost_win=w) for c in caps for w in cost_wins]

# ---- build_fn: causal hedged short-vol pnl for a given param dict -----------
# hedged_pnl is fully causal (iv from vxn.shift(1); wing = smile * trailing-mean(tail).shift(1)).
def build_fn(p):
    return H.hedged_pnl(ret, vxn, cap=p["cap"], smile=SMILE, cost_win=p["cost_win"])

# ---- walk-forward: params chosen OUT-OF-SAMPLE on trailing-train Sharpe -----
oos = H.walk_forward(build_fn, grid, train=1260, test=252)

# ---- scale OOS pnl to 10% vol with CAUSAL trailing-std scaler ----------------
oos_scaled = H.causal_scale(oos, target=0.10)

# ---- metrics on the OOS series ----------------------------------------------
m = H.metrics(oos_scaled)
worst_day = float(oos_scaled.min())
picks = oos.attrs.get("picks", [])

# pick frequency summary
from collections import Counter
pick_counts = Counter((p["cap"], p["cost_win"]) for p in picks)
pick_summary = sorted(pick_counts.items(), key=lambda kv: -kv[1])

BASELINE_SHARPE = 1.394

print("=" * 70)
print("Strategy B - OPTIMAL WING STRIKE (walk-forward, OOS)")
print("=" * 70)
print(f"grid: cap in {caps} x cost_win in {cost_wins}, smile fixed = {SMILE}")
print(f"walk_forward train=1260, test=252  | n test blocks = {len(picks)}")
print(f"OOS series length (scaled) = {len(oos_scaled)}")
print("-" * 70)
print(f"OOS Sharpe      : {m['sharpe']:.4f}")
print(f"OOS CAGR        : {m['cagr']:.4f}")
print(f"OOS maxDD       : {m['maxdd']:.4f}")
print(f"OOS skew        : {m['skew']:.4f}")
print(f"worst 1-day ret : {worst_day:.4f}  ({worst_day*100:.2f}%)")
print(f"n               : {m['n']}")
print("-" * 70)
print(f"Realistic baseline Sharpe (smile=1.5 static hedged): {BASELINE_SHARPE}")
print(f"Beats baseline  : {m['sharpe'] > BASELINE_SHARPE}")
print("-" * 70)
print("Walk-forward picks (cap, cost_win) -> count:")
for (cap, cw), n in pick_summary:
    print(f"   cap={cap:.2f}, cost_win={cw:>3d}  ->  {n} block(s)")
print("=" * 70)

# emit a compact machine-readable line for the orchestrator
print("RESULT_JSON", {
    "oos_sharpe": m["sharpe"],
    "oos_maxdd": m["maxdd"],
    "worst_day": worst_day,
    "skew": m["skew"],
    "beats_baseline": bool(m["sharpe"] > BASELINE_SHARPE),
    "picks": pick_summary,
})
