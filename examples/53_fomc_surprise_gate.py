# -*- coding: utf-8 -*-
"""Example 53 - does the Bauer-Swanson FOMC surprise MAGNITUDE add value as a causal overlay?
The MPS is revealed ~2pm on the FOMC day, so the only causal trade is conditioning the DAY-AFTER
(D+1) position on today's realized |surprise| (vol persistence). Tested on SPX (deepest, most
FOMC-sensitive sleeve), with honest significance (bootstrap) and a clear low-power caveat."""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
import bookopt_harness as H, bookopt_floor as F, bookopt_stats as S

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"; SQ = H.SQ
H._load(); df, ret, vi, sp0 = H._DATA["SPX"]; idx = ret.index
prem = pd.Series(2*(2*F.Nrm(0.5*(H.K*vi.values/100)*np.sqrt(H.DT))-1), index=idx)
sv = (prem.shift(1) - ret.abs())                       # gross short-one-straddle per-unit P&L

mps = pd.read_csv(os.path.join(OUT, "FOMC_MPS.csv"), parse_dates=["Date"])
mps = mps[(mps["unscheduled"] == 0) & mps["mps_orth"].notna()]   # scheduled, orthogonalized available
loc = idx.get_indexer(mps["Date"].values, method="bfill")
ok = (loc >= 2) & (loc < len(idx)-2)
d0 = loc[ok]; surp = np.abs(mps["mps_orth"].values[ok]); d1 = d0 + 1
print(f"FOMC SURPRISE OVERLAY (SPX)  {len(d0)} scheduled FOMC in-sample {idx[d0].min().date()}..{idx[d0].max().date()}\n")

# (1) does |surprise| predict the realized move? (concurrent vs next-day) ----------------
r0 = ret.abs().values[d0]; r1 = ret.abs().values[d1]
print(f"(1) corr(|MPS_ORTH|, |SPX move|):  same-day D0 {np.corrcoef(surp, r0)[0,1]:+.2f}  |  "
      f"next-day D+1 {np.corrcoef(surp, r1)[0,1]:+.2f}")
print("    (D0 concurrent link is the surprise itself; D+1 = does the shock PERSIST into the next day?)")

# (2) gross short-vol P&L by surprise tercile -------------------------------------------
hi = np.quantile(surp, 0.67)
big = surp >= hi
base = sv.mean()
print(f"\n(2) gross short-straddle P&L (bp/unit), baseline all-days {base*1e4:+.1f}:")
print(f"    FOMC D0 (all)            {sv.values[d0].mean()*1e4:+6.1f}")
print(f"    FOMC D+1 after BIG surp  {sv.values[d1][big].mean()*1e4:+6.1f}   (top-tercile |MPS_ORTH| >= {hi*100:.1f}bp)")
print(f"    FOMC D+1 after small     {sv.values[d1][~big].mean()*1e4:+6.1f}")
print("    -> if D+1-after-big is materially negative, standing down short-vol there should help.")

# (3) causal overlay on the SPX sleeve: flatten on D+1 after a top-tercile surprise -------
blend, _ = H.market("SPX")
standdown = pd.Index(idx[d1][big])
ov = blend.copy(); ov[ov.index.isin(standdown)] = 0.0
base_sh, ov_sh = S.sharpe_ann(blend), S.sharpe_ann(ov)
# bootstrap the Sharpe DIFFERENCE (block) to gauge whether it's signal or noise
rng = np.random.default_rng(5); diffs = []
b = blend.dropna().values; o = ov.reindex(blend.dropna().index).values; T = len(b); blk = 20; nb = int(np.ceil(T/blk))
for _ in range(2000):
    st = rng.integers(0, T-blk+1, nb)
    sel = np.concatenate([np.arange(s, s+blk) for s in st])[:T]
    bb, oo = b[sel], o[sel]
    sdb, sdo = bb.std(ddof=1), oo.std(ddof=1)
    diffs.append((oo.mean()/sdo - bb.mean()/sdb)*SQ if sdb>0 and sdo>0 else 0.0)
lo, h = np.percentile(diffs, [2.5, 97.5])
print(f"\n(3) OVERLAY (flatten SPX on {int(big.sum())} D+1-after-big-surprise days):")
print(f"    SPX sleeve Sharpe  base {base_sh:.2f} -> overlay {ov_sh:.2f}  (delta {ov_sh-base_sh:+.2f})")
print(f"    bootstrap 95% CI on delta [{lo:+.2f}, {h:+.2f}]  "
      f"({'signal' if lo>0 else 'NOT distinguishable from noise'})")
print(f"\nVERDICT: exploratory + LOW POWER ({int(big.sum())} affected days over ~18y). The graded surprise")
print("is a genuinely NEW signal, but on this book the day-after overlay is a small, marginal effect;")
print("its real promise is as a magnitude-conditioned EVENT feature, not a standalone gate. Honest null-ish.")
