# -*- coding: utf-8 -*-
"""
p1_sens.py -- P1-6 PARAMETER SENSITIVITY for the NQ/VXN long-short vol-arb study.

PANELS 1-3 (heatmaps): FIXED params, FULL SAMPLE 2001-2026, proxy instrument,
NO walk-forward.  *** THESE ARE IN-SAMPLE NUMBERS *** -- their only legitimate
use is the SHAPE of the response surface (plateau vs peak), never as a
performance claim.  Levels are inflated by both in-sampleness and the
untradable 1-day-variance-at-VXN-strike proxy.

  P1: family A Sharpe over r_hi in 1..8 (step 1) x d in 0..4 (step 0.5), r_lo=-2
  P2: family A Sharpe over r_lo in -6..0 (step 1) x r_hi in 1..8, d=0
  P3: family B Sharpe over b1 in 0.6..1.4 (step 0.1) x b2 in 1.2..2.4 (step 0.2)

PANEL 4 (tornado): the honest number -- OOS walk-forward 50/50 blend Sharpe --
re-run under perturbed harness assumptions:
  train window 1000/1260/1500, test window 126/252, cost coeff 0.25/0.5/1.0/2.0 vol-pt.
Signal grids inside the walk-forward stay the ORIGINAL small grids (18 + 9 combos),
i.e. the tornado does NOT add selection pressure; it only varies harness settings.

Output: C:\\Users\\ASUS\\Desktop\\claude doc\\1\\p1_sensitivity.png  (2x2 panel)
"""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from collections import Counter
import volarb_harness as H

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

# ---------- data + causal blocks (identical to production scripts) ----------
df, ret, vxn = H.load()
fc21 = H.fcast_vol(df, ret, "park21")   # value at t uses data through t-1
fc10 = H.fcast_vol(df, ret, "park10")
fc42 = H.fcast_vol(df, ret, "park42")
richness = vxn - fc21
trend = fc10 - fc42
rng = np.log(df["High"] / df["Low"]) * 100.0
be = vxn / SQ


def sig_A(r_hi, r_lo, d):
    s = pd.Series(0.0, index=ret.index)
    s[((richness >= r_hi) & (trend <= -d)).fillna(False)] = 1.0
    s[((richness <= r_lo) & (trend >= d)).fillna(False)] = -1.0
    return s


def sig_B(b1, b2):
    s = pd.Series(0.0, index=ret.index)
    ok = rng.notna() & be.notna()
    s[ok & (rng < b1 * be)] = 1.0
    s[ok & (rng > b2 * be)] = -1.0
    return s


def sharpe_full(s, cost=0.5):
    """FULL-SAMPLE (in-sample) Sharpe of one fixed-param signal."""
    return H.metrics(H.backtest(s, ret, vxn, cost_volpt=cost))["sharpe"]


# ---------- panels 1-3: in-sample heatmaps ----------
R_HI = np.arange(1.0, 8.01, 1.0)                 # 8
D    = np.round(np.arange(0.0, 4.01, 0.5), 1)    # 9
R_LO = np.arange(-6.0, 0.01, 1.0)                # 7
B1   = np.round(np.arange(0.6, 1.401, 0.1), 1)   # 9
B2   = np.round(np.arange(1.2, 2.401, 0.2), 1)   # 7

print("building in-sample heatmaps (FULL-SAMPLE, FIXED PARAMS -- disclosure: in-sample) ...")
M1 = np.array([[sharpe_full(sig_A(rh, -2.0, d)) for rh in R_HI] for d in D])    # rows=d, cols=r_hi
M2 = np.array([[sharpe_full(sig_A(rh, rl, 0.0)) for rh in R_HI] for rl in R_LO])  # rows=r_lo
M3 = np.array([[sharpe_full(sig_B(b1, b2)) for b1 in B1] for b2 in B2])          # rows=b2, cols=b1
n_cells = M1.size + M2.size + M3.size
print("heatmap cells evaluated: %d (for deflation accounting; none used for selection)" % n_cells)

# ---------- panel 4: tornado on the OOS walk-forward blend ----------
GRID_A = [(a, b, c) for a in (2.0, 4.0, 6.0) for b in (0.0, -2.0) for c in (0.0, 1.0, 2.0)]
GRID_B = [(b1, b2) for b1 in (0.8, 1.0, 1.2) for b2 in (1.3, 1.6, 2.0)]


def blend_oos(train, test, cost):
    pa = H.walk_forward(lambda g: H.backtest(sig_A(*g), ret, vxn, cost_volpt=cost),
                        GRID_A, train=train, test=test)
    pb = H.walk_forward(lambda g: H.backtest(sig_B(*g), ret, vxn, cost_volpt=cost),
                        GRID_B, train=train, test=test)
    common = pa.index.intersection(pb.index)
    bl = 0.5 * pa.loc[common] + 0.5 * pb.loc[common]
    return H.metrics(bl)["sharpe"], pa.attrs["picks"], pb.attrs["picks"], common


print("\nrunning walk-forward tornado (OOS, honest numbers) ...")
sh_base, picks_A, picks_B, common = blend_oos(1260, 252, 0.5)
print("base blend OOS Sharpe (train=1260,test=252,cost=0.5): %.3f  span %s..%s (%d d)"
      % (sh_base, common.min().date(), common.max().date(), len(common)))

modal_A = Counter(picks_A).most_common(1)[0]
modal_B = Counter(picks_B).most_common(1)[0]
print("modal walk-forward pick A (r_hi,r_lo,d): %s in %d/%d blocks" % (modal_A[0], modal_A[1], len(picks_A)))
print("modal walk-forward pick B (b1,b2):       %s in %d/%d blocks" % (modal_B[0], modal_B[1], len(picks_B)))

tornado = {}
tornado["train window (d)"] = {}
for tr in (1000, 1260, 1500):
    tornado["train window (d)"][tr] = sh_base if tr == 1260 else blend_oos(tr, 252, 0.5)[0]
tornado["test window (d)"] = {252: sh_base, 126: blend_oos(1260, 126, 0.5)[0]}
tornado["cost (vol-pt)"] = {}
for c in (0.25, 0.5, 1.0, 2.0):
    tornado["cost (vol-pt)"][c] = sh_base if c == 0.5 else blend_oos(1260, 252, c)[0]

print("\nOOS blend Sharpe under perturbed harness settings:")
for fac, dd in tornado.items():
    for k, v in sorted(dd.items()):
        tag = "  <- base" if v == sh_base and ((fac.startswith("train") and k == 1260)
                                               or (fac.startswith("test") and k == 252)
                                               or (fac.startswith("cost") and k == 0.5)) else ""
    # printed in detail below
for fac, dd in tornado.items():
    parts = ", ".join("%s -> %.2f" % (k, v) for k, v in sorted(dd.items()))
    print("  %-16s %s" % (fac, parts))

# ---------- plateau-vs-peak diagnostics ----------
def plateau_stats(M, iy, ix, name):
    v = M[iy, ix]
    y0, y1 = max(0, iy - 1), iy + 2
    x0, x1 = max(0, ix - 1), ix + 2
    nb = M[y0:y1, x0:x1]
    st = dict(chosen=float(v), grid_max=float(np.nanmax(M)), grid_med=float(np.nanmedian(M)),
              pct_within_025=float(np.mean(M >= v - 0.25) * 100),
              pct_above_half=float(np.mean(M >= 0.5 * np.nanmax(M)) * 100),
              nb_min=float(np.nanmin(nb)), nb_mean=float(np.nanmean(nb)))
    print("  %-28s chosen %.2f | grid max %.2f med %.2f | %4.0f%% cells within 0.25 of chosen | "
          "neighbors min %.2f mean %.2f" % (name, st["chosen"], st["grid_max"], st["grid_med"],
                                            st["pct_within_025"], st["nb_min"], st["nb_mean"]))
    return st

# chosen cells = modal walk-forward picks (nearest grid index)
gA = modal_A[0]; gB = modal_B[0]
ixA1 = int(np.argmin(np.abs(R_HI - gA[0]))); iyA1 = int(np.argmin(np.abs(D - gA[2])))
ixA2 = int(np.argmin(np.abs(R_HI - gA[0]))); iyA2 = int(np.argmin(np.abs(R_LO - gA[1])))
ixB  = int(np.argmin(np.abs(B1 - gB[0])));   iyB  = int(np.argmin(np.abs(B2 - gB[1])))

print("\nplateau-vs-peak diagnostics (IN-SAMPLE surfaces, chosen = modal WF pick):")
stA1 = plateau_stats(M1, iyA1, ixA1, "A: r_hi x d (r_lo=-2)")
stA2 = plateau_stats(M2, iyA2, ixA2, "A: r_lo x r_hi (d=0)")
stB  = plateau_stats(M3, iyB, ixB,  "B: b1 x b2")

# ---------- figure ----------
fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))
fig.suptitle("P1-6 parameter sensitivity -- NQ/VXN long-short variance proxy\n"
             "Heatmaps: FIXED params, FULL-SAMPLE 2001-2026, IN-SAMPLE (no walk-forward) -- "
             "shape only, NOT performance claims.  Tornado: OOS walk-forward blend.",
             fontsize=11)

def heat(ax, M, xvals, yvals, xlab, ylab, title, ix, iy):
    vmin, vmax = float(np.nanmin(M)), float(np.nanmax(M))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=min(vmin, -1e-6), vmax=max(vmax, 1e-6))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="RdYlGn", norm=norm)
    ax.set_xticks(range(len(xvals))); ax.set_xticklabels(["%g" % v for v in xvals], fontsize=8)
    ax.set_yticks(range(len(yvals))); ax.set_yticklabels(["%g" % v for v in yvals], fontsize=8)
    ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=6.5,
                    color="black")
    ax.add_patch(plt.Rectangle((ix - 0.5, iy - 0.5), 1, 1, fill=False, edgecolor="blue", lw=2.2))
    fig.colorbar(im, ax=ax, shrink=0.85, label="in-sample Sharpe")

heat(axes[0, 0], M1, R_HI, D, "r_hi (vol pts)", "d (vol pts)",
     "Family A: Sharpe vs (r_hi, d) at r_lo=-2   [blue = modal WF pick]", ixA1, iyA1)
heat(axes[0, 1], M2, R_HI, R_LO, "r_hi (vol pts)", "r_lo (vol pts)",
     "Family A: Sharpe vs (r_hi, r_lo) at d=0   [blue = modal WF pick]", ixA2, iyA2)
heat(axes[1, 0], M3, B1, B2, "b1 (short-vol gate)", "b2 (long-vol gate)",
     "Family B: Sharpe vs (b1, b2)   [blue = modal WF pick]", ixB, iyB)

# tornado
axT = axes[1, 1]
items = []
for fac, dd in tornado.items():
    vals = list(dd.values())
    items.append((fac, min(vals), max(vals), dd))
items.sort(key=lambda r: (r[2] - r[1]), reverse=True)
ypos = np.arange(len(items))[::-1]
for y, (fac, lo, hi, dd) in zip(ypos, items):
    axT.barh(y, hi - lo, left=lo, height=0.5, color="steelblue", alpha=0.65)
    for k, v in dd.items():
        axT.plot(v, y, "o", color="black", ms=4)
        axT.annotate("%g" % k, (v, y), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=7.5)
axT.axvline(sh_base, color="firebrick", ls="--", lw=1.4,
            label="base %.2f (train 1260, test 252, cost 0.5)" % sh_base)
axT.set_yticks(ypos); axT.set_yticklabels([r[0] for r in items], fontsize=9)
axT.set_xlabel("OOS walk-forward blend Sharpe", fontsize=9)
axT.set_title("Tornado: OOS blend Sharpe vs harness assumptions (honest, walk-forward)",
              fontsize=10)
axT.legend(fontsize=8, loc="lower left")
axT.grid(axis="x", alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.94])
png = os.path.join(OUT, "p1_sensitivity.png")
fig.savefig(png, dpi=110)
plt.close(fig)
print("\nsaved -> %s" % png)

# ---------- verdict ----------
def verdict(st):
    # plateau if chosen is within 0.25 of grid max AND worst 8-neighbor keeps >=70% of chosen
    near_max = st["chosen"] >= st["grid_max"] - 0.25
    sturdy_nb = st["nb_min"] >= 0.7 * st["chosen"] if st["chosen"] > 0 else False
    return near_max, sturdy_nb

print("\nVERDICT (in-sample surface shape):")
for name, st in [("A (r_hi x d)", stA1), ("A (r_lo x r_hi)", stA2), ("B (b1 x b2)", stB)]:
    nm, sn = verdict(st)
    kind = "PLATEAU" if (nm and sn) else ("RIDGE/EDGE" if sn or nm else "PEAK")
    print("  %-16s -> %s  (near-max=%s, sturdy-neighborhood=%s)" % (name, kind, nm, sn))
tor_lo = min(min(d.values()) for d in tornado.values())
tor_hi = max(max(d.values()) for d in tornado.values())
print("  tornado: OOS blend Sharpe stays in [%.2f, %.2f] across all harness perturbations." % (tor_lo, tor_hi))
print("\nDISCLOSURES: heatmap Sharpes are FULL-SAMPLE IN-SAMPLE on the proxy instrument;")
print("they exist only to judge surface shape. The only out-of-sample numbers in this")
print("figure are in the tornado panel. Proxy P&L (1-day variance at VXN strike) inflates")
print("levels for every cell equally; a tradable implementation would scale all panels down.")
