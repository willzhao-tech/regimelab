"""
Example 24 — independent audit of the optimize-shortvol workflow's "2.52 Sharpe / tail-safe" claim.

Two suspicions:
  (1) the harness baseline returned fair Sharpe 2.09, but the honest de-bugged baseline is 0.45
      -> the jump must be the causal vol-targeting (causal_scale), not a real edge.
  (2) the -1.72% "worst day" is a fiction of the perfect variance cap + the 8x leverage cap;
      the synthesis itself admitted uncapped it loses -16.95%.
We trace the Sharpe to its source and stress what the worst day really is.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")

import numpy as np, pandas as pd
import shortvol_harness as H

df, ret, vxn = H.load()
SQ = H.SQ

print("=" * 80)
print("(1) WHERE DOES THE SHARPE COME FROM?  raw hedged P&L vs causal-vol-targeted")
print("=" * 80)
print(f"  {'smile':>6}{'RAW Sharpe':>12}{'SCALED Sharpe':>15}{'raw vol':>9}{'%days @8x cap':>15}")
for smile in (1.0, 1.5, 2.5, 3.0):
    pnl = H.hedged_pnl(ret, vxn, cap=0.05, smile=smile)
    scaled = H.causal_scale(pnl)
    raw_sh = pnl.mean() / pnl.std() * SQ
    sc_sh = scaled.mean() / scaled.std() * SQ
    s = pnl.rolling(63).std().shift(1) * SQ
    lev = (0.10 / s).clip(upper=8.0)
    at_cap = (lev >= 7.99).mean() * 100
    print(f"  {smile:>6.1f}{raw_sh:>12.2f}{sc_sh:>15.2f}{pnl.std()*SQ*100:>8.1f}%{at_cap:>14.0f}%")
print("  -> the SCALED Sharpe >> RAW Sharpe, and leverage sits pinned at the 8x cap most days:")
print("     causal vol-targeting just LEVERS the capped carry to the max in calm periods.")

print("\n" + "=" * 80)
print("(2) IS THE TAIL ACTUALLY SAFE?  worst day WITH the perfect variance cap vs WITHOUT it")
print("=" * 80)
smile = 2.5
pnl_capped = H.hedged_pnl(ret, vxn, cap=0.05, smile=smile)              # min(rvar,capv): perfect wings
# same strategy but the wings DON'T perfectly cap daily variance (you pay full realized):
iv = (vxn.shift(1) / 100) ** 2 / 252
rvar = ret ** 2
wing = smile * np.maximum(rvar - 0.05**2, 0).rolling(252, min_periods=60).mean().shift(1)
pnl_uncapped = (iv - rvar - wing).dropna()
# apply the SAME causal leverage (from the capped stream) to both, to compare the real worst day
s = pnl_capped.rolling(63).std().shift(1) * SQ
lev = (0.10 / s).clip(upper=8.0).fillna(0.0)
cap_scaled = (lev * pnl_capped).dropna()
unc_scaled = (lev.reindex(pnl_uncapped.index).fillna(0.0) * pnl_uncapped).dropna()
print(f"  perfect-cap (model): worst day {cap_scaled.min()*100:+.1f}%   Sharpe {cap_scaled.mean()/cap_scaled.std()*SQ:.2f}")
print(f"  imperfect wings    : worst day {unc_scaled.min()*100:+.1f}%   Sharpe {unc_scaled.mean()/unc_scaled.std()*SQ:.2f}")
wd = unc_scaled.nsmallest(3)
print("  3 worst days if the cap isn't perfect (gap/basis):")
for d, v in wd.items():
    print(f"    {d.date()}  {v*100:+.1f}%   (NQ {ret.get(d,0)*100:+.1f}%, VXN {vxn.get(d,float('nan')):.0f})")

print("\n" + "=" * 80)
print("(3) HONEST BASELINE: raw hedged Sharpe with realistic wings, NO causal-vol-target leverage")
print("=" * 80)
for smile in (1.5, 2.0, 2.5, 3.0):
    pnl = H.hedged_pnl(ret, vxn, cap=0.05, smile=smile)
    print(f"  smile {smile:.1f}: raw Sharpe {pnl.mean()/pnl.std()*SQ:+.2f}, "
          f"ann premium {pnl.mean()*252/ (0.05**2) :+.2f} (rough), worst capped day {pnl.min()*100:+.2f}%")

print("\nVERDICT: the 2.0-2.5 Sharpe = causal vol-targeting LEVERING a tail-capped carry to 8x in")
print("calm periods. The -1.7% worst day is an artifact of (a) a PERFECT variance cap and (b) the")
print("8x leverage cap; if the wings don't perfectly cap a gap day, the 8x book loses ~15-17% in a")
print("day. Honest, unlevered, realistically-priced short-vol is ~0.3-0.5 Sharpe at best. Same")
print("illusion as the original bug, one layer deeper: lever the carry, assume the tail away.")
