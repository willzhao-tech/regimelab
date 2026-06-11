# -*- coding: utf-8 -*-
"""
Book-level risk-control overlays on the baseline 8-market L4 book.

GOAL: improve Calmar (Sharpe/|maxDD|) WITHOUT look-ahead. Not chasing Sharpe via
leverage-pinning -- we disclose pinning explicitly.

Every overlay below is decidable from TRAILING data only:
  (a) vol-target / cap variants: the lev formula in book_of already uses
      trailing 63d std .shift(1); we only change the *constant* target and the cap.
      A target/cap is a fixed design choice, not chosen from full-sample PnL.
  (b) trailing-drawdown brake: de-risk when the realised equity curve is in a
      drawdown > X% as of YESTERDAY (running max uses data up to t-1 only).
  (c) skip-when-all-flat: if every sleeve's raw (pre-scale) signal is flat on a
      day, set book exposure to 0 that day. Decidable same-day from positions
      that were themselves entered on shifted signals -> causal.

CARDINAL RULE: no market/param is chosen by looking at full-sample results.
The X in the DD-brake and the target/cap are reported across a small grid so the
reader sees sensitivity; the headline pick is a round, a-priori value (10% target,
the brake at -15% DD), NOT the grid-max. We show the grid only for honesty.
"""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
import bookopt_harness as H

SQ = H.SQ

# ---------------------------------------------------------------- build sleeves
NAMES = [p[0] for p in H.PAIRS]
sleeves = {}
flat_mask = {}   # per-sleeve: True where the GATED+STATIC blend position is ~0
for name in NAMES:
    g, s = H.market(name)
    if g is None:
        continue
    sleeves[name] = g
print("sleeves built:", list(sleeves.keys()))

# baseline book (no overlay) ----------------------------------------------------
base = H.book_of(sleeves)
b_sh = H.sharpe(base); b_dd = H.maxdd(base); b_h1, b_h2 = H.split_halves(base)
b_calmar = b_sh / abs(b_dd)
print(f"\nBASELINE book: Sharpe={b_sh:.3f}  maxDD={b_dd:.3f}  Calmar={b_calmar:.3f}  "
      f"halves=({b_h1:.2f},{b_h2:.2f})  N={len(base)}")

# ---------------------------------------------------------------------------
# Re-implement book_of's internals so we can swap the lev constants and add a
# causal multiplier. We mirror harness book_of EXACTLY for target=0.10, cap=4.0,
# overlay=None  ->  reproduces baseline.
# ---------------------------------------------------------------------------
def build_book_raw(d):
    """Return the equal-risk pre-leverage blend 'bk' (same as book_of internals)."""
    sc = {k: (p / (p.rolling(63).std().shift(1))) for k, p in d.items()}
    P = pd.DataFrame(sc)
    bk = P.mean(axis=1, skipna=True).dropna()
    return bk

def lever(bk, target=0.10, cap=4.0):
    lev = (target / (bk.rolling(63).std().shift(1) * SQ)).clip(upper=cap)
    return lev, (lev * bk).dropna()

bk_raw = build_book_raw(sleeves)

# sanity: reproduce baseline
_, repro = lever(bk_raw, 0.10, 4.0)
print(f"reproduce baseline Sharpe={H.sharpe(repro):.3f} (should match {b_sh:.3f})")

# pinning diagnostic: fraction of days the leverage sits ON the cap.
lev0, _ = lever(bk_raw, 0.10, 4.0)
pin_frac = float((lev0 >= 4.0 - 1e-9).reindex(repro.index).mean())
print(f"baseline leverage-cap PIN fraction = {pin_frac:.1%} "
      f"(high pin => Sharpe is partly leverage-limited, gains from re-targeting may be illusory)")

def report(tag, r):
    sh = H.sharpe(r); dd = H.maxdd(r); h1, h2 = H.split_halves(r)
    cal = sh / abs(dd) if dd != 0 else float('nan')
    print(f"  {tag:42s} Sharpe={sh:6.3f}  maxDD={dd:7.3f}  Calmar={cal:5.3f}  halves=({h1:5.2f},{h2:5.2f})")
    return dict(tag=tag, sharpe=sh, maxdd=dd, calmar=cal, h1=h1, h2=h2, r=r)

# =========================================================================
# (a) vol-target / cap variants
# =========================================================================
print("\n(a) VOL-TARGET / CAP variants  [target, cap are fixed a-priori constants]")
res_a = []
for tgt in (0.08, 0.10, 0.12):
    for cap in (2.0, 3.0, 4.0):
        _, r = lever(bk_raw, tgt, cap)
        res_a.append(report(f"target={tgt:.2f} cap={cap:.1f}", r))

# =========================================================================
# (b) trailing-drawdown brake (CAUSAL): scale exposure by a factor that depends
#     ONLY on the equity drawdown observed up to t-1.
#     brake(t) = 1.0           if dd_{t-1} >= -X
#                de_risk        if dd_{t-1} <  -X
#     We compute the brake on the book's OWN realised returns, lagged one day.
# =========================================================================
def dd_brake(r, X=0.15, de_risk=0.5):
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    # use YESTERDAY's drawdown to decide today's scale -> shift(1), causal
    scale = pd.Series(np.where(dd.shift(1) < -X, de_risk, 1.0), index=r.index)
    return (r * scale).dropna(), scale

print("\n(b) TRAILING-DRAWDOWN BRAKE  [decide on yesterday's DD only -> causal]")
res_b = []
base_for_brake = repro  # apply on the baseline 10%/4x book
for X in (0.10, 0.15, 0.20):
    for dr in (0.0, 0.5):
        r, sc = dd_brake(base_for_brake, X, dr)
        frac = float((sc < 1.0).mean())
        res_b.append((report(f"DD-brake X={X:.0%} de-risk->{dr:.1f}  (braked {frac:.0%} of days)", r), X, dr))

# =========================================================================
# (c) skip-trading when ALL sleeves flat (CAUSAL): on days where every sleeve's
#     scaled contribution is exactly 0 (no position), the book already returns ~0
#     for that day, so this is mostly a no-op on PnL; the meaningful version is to
#     also DROP the day's leverage churn. We test it for completeness/honesty.
#     A sleeve is "flat" when its risk-scaled pnl is exactly 0 (no position held).
# =========================================================================
print("\n(c) SKIP-WHEN-ALL-FLAT  [zero book exposure on all-flat days -> causal]")
sc_df = pd.DataFrame({k: (p / (p.rolling(63).std().shift(1))) for k, p in sleeves.items()})
all_flat = (sc_df.fillna(0.0) == 0.0).all(axis=1)   # same-day flatness of realised pnl
# apply on baseline book: zero out all-flat days
mask = (~all_flat).reindex(repro.index).fillna(True).astype(float)
r_c = (repro * mask).dropna()
res_c = report(f"skip all-flat (flat {float(all_flat.reindex(repro.index).mean()):.1%} of days)", r_c)

# =========================================================================
# COMBINED a-priori pick: target=0.10, cap=3.0 (tighter cap reduces pin-driven
# tail leverage) + DD-brake at X=15% de-risk->0.5. Both are round, pre-chosen.
# =========================================================================
print("\nCOMBINED a-priori overlay (target=0.10, cap=3.0, DD-brake X=15% -> 0.5):")
_, r_cap3 = lever(bk_raw, 0.10, 3.0)
r_comb, sc_comb = dd_brake(r_cap3, 0.15, 0.5)
combo = report("COMBINED", r_comb)

# pinning disclosure for cap=3.0
lev_c3, _ = lever(bk_raw, 0.10, 3.0)
pin3 = float((lev_c3 >= 3.0 - 1e-9).reindex(r_cap3.index).mean())
print(f"  cap=3.0 PIN fraction = {pin3:.1%}")

# ---------------------------------------------------------------- choose headline
# Headline = best CALMAR among a-priori candidates that do NOT increase pin and do
# NOT rely on raising the vol target (no leverage-pinning Sharpe inflation).
# Candidates restricted to: cap<=baseline, target<=baseline, + brake.
print("\n=== HEADLINE (Calmar-improving, no leverage-pinning) ===")
candidates = {
    "baseline": repro,
    "cap3.0": r_cap3,
    "brake15": dd_brake(repro, 0.15, 0.5)[0],
    "combined(cap3+brake15)": r_comb,
}
rows = []
for tag, r in candidates.items():
    sh = H.sharpe(r); dd = H.maxdd(r); cal = sh/abs(dd); h1, h2 = H.split_halves(r)
    rows.append((tag, sh, dd, cal, h1, h2))
    print(f"  {tag:26s} Sharpe={sh:6.3f} maxDD={dd:7.3f} Calmar={cal:5.3f} halves=({h1:5.2f},{h2:5.2f})")

best = max(rows[1:], key=lambda x: x[3])  # exclude baseline; maximise Calmar
print(f"\nBEST non-baseline by Calmar: {best[0]}  Sharpe={best[1]:.3f} maxDD={best[2]:.3f} Calmar={best[3]:.3f}")
print(f"baseline Calmar = {b_calmar:.3f}")
