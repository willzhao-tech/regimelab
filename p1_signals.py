# -*- coding: utf-8 -*-
"""
p1_signals.py -- TASK P1-7: ORTHOGONAL SIGNALS as AND-gates on family A (regime_combo).

Window: 2013-10-16+ (first SKEW print) so ALL candidates and the un-gated baseline
run on the IDENTICAL calendar and identical walk-forward blocks.

Candidates (each gates ONLY the short-vol leg of family A; the long-vol leg is
unchanged; an 'off' threshold = +inf is included in every grid so the gated
family is a strict superset of family A and walk-forward can decline the gate):

  1) skew_z   : CBOE SKEW 252d trailing z-score. short-vol additionally requires
                skew_z <= thr (don't sell vol when tail protection is heavily bid).
                thr in {inf, 1.0, 0.5, 0.0}
  2) volofvol : trailing 21d std of VXN daily changes (uses changes through close t;
                position is taken at t+1 by the harness shift -> causal).
                short-vol requires vov <= thr. thr in {inf, 2.0, 1.5, 1.0}
  3) vxn_term : VXN_t minus its own 63d trailing mean (through t). Positive =
                vol elevated vs own recent history (stress / backwardation proxy).
                short-vol requires term <= thr. thr in {inf, 4.0, 2.0, 0.0}

Causality: identical conventions to va_regime_combo.py. s_t may use close-t info;
H.backtest() shifts s by one day. fcast_vol is itself shifted (through t-1).
Params picked ONLY by H.walk_forward (1260d train -> 252d test). No caps, no
clipping, no tail edits.

Honest trial count for deflation: see DEFLATION NOTE printed at the end.
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import volarb_harness as H

DATA = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

# ---------------------------------------------------------------- data
df_full, ret_full, vxn_full = H.load()
skew_raw = (pd.read_csv(os.path.join(DATA, "SKEW_all_history.csv"), parse_dates=["Date"])
            .set_index("Date").sort_index()["Close"].dropna())

start = skew_raw.index[0]
base_idx = ret_full.index[ret_full.index >= start]
df = df_full.loc[base_idx]
ret = ret_full.loc[base_idx]          # returns computed on FULL history, then selected
vxn = vxn_full.loc[base_idx]
skew = skew_raw.reindex(base_idx).ffill()   # NQ/VXN calendar; ffill SKEW holidays
n_ffill = int(skew_raw.reindex(base_idx).isna().sum())
print(f"window: {base_idx[0].date()} .. {base_idx[-1].date()}  n={len(base_idx)}  "
      f"(SKEW days ffilled: {n_ffill})")

# ---------------------------------------------------------------- family-A blocks
fc21 = H.fcast_vol(df, ret, "park21")   # value at t uses data through t-1
fc10 = H.fcast_vol(df, ret, "park10")
fc42 = H.fcast_vol(df, ret, "park42")
richness = vxn - fc21
trend = fc10 - fc42

# ---------------------------------------------------------------- gate variables
skew_z = (skew - skew.rolling(252).mean()) / skew.rolling(252).std()  # through t
vov = vxn.diff().rolling(21).std()                                    # through t
term = vxn - vxn.rolling(63).mean()                                   # through t

for nm, g in [("skew_z", skew_z), ("volofvol", vov), ("vxn_term", term)]:
    q = g.dropna().quantile([0.10, 0.50, 0.90])
    print(f"gate {nm}: q10={q.iloc[0]:.2f} q50={q.iloc[1]:.2f} q90={q.iloc[2]:.2f}")

# ---------------------------------------------------------------- signal builder
R_HI = [2.0, 4.0, 6.0]
R_LO = [0.0, -2.0]
D = [0.0, 1.0, 2.0]
BASE_GRID = [(a, b, c) for a in R_HI for b in R_LO for c in D]   # 18 combos


def make_signal(g, gate=None, thr=np.inf):
    r_hi, r_lo, d = g
    short_mask = (richness >= r_hi) & (trend <= -d)
    long_mask = (richness <= r_lo) & (trend >= d)
    if gate is not None and np.isfinite(thr):
        short_mask = short_mask & (gate <= thr)   # NaN gate -> False -> flat
    s = pd.Series(0.0, index=ret.index)
    s[short_mask.fillna(False)] = 1.0
    s[long_mask.fillna(False)] = -1.0
    return s


def ols_alpha(y, x):
    """y = a + b*x; returns alpha, t(alpha), beta."""
    yx = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    X = np.column_stack([np.ones(len(yx)), yx["x"].values])
    yv = yx["y"].values
    beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    sigma2 = float(resid @ resid) / (len(yv) - 2)
    covb = sigma2 * np.linalg.inv(X.T @ X)
    return float(beta[0]), float(beta[0] / np.sqrt(covb[0, 0])), float(beta[1])


def oos_positions(picks, pnl_idx, gate=None, train=1260, test=252):
    """Reconstruct harness positions on OOS dates given per-block picks."""
    parts = []
    start_i, b = train, 0
    while start_i + test <= len(pnl_idx):
        te = pnl_idx[start_i:start_i + test]
        p = picks[b]
        if gate is None:
            s = make_signal(p)
        else:
            s = make_signal(p[:3], gate=gate, thr=p[3])
        pos = s.reindex(ret.index).clip(-1, 1).shift(1).fillna(0.0)
        parts.append(pos.reindex(te))
        start_i += test
        b += 1
    return pd.concat(parts).dropna()


# ---------------------------------------------------------------- baseline A (un-gated, same window)
def build_base(g):
    return H.backtest(make_signal(g), ret, vxn)


oos_A = H.walk_forward(build_base, BASE_GRID)
picks_A = oos_A.attrs["picks"]
pnl_idx = build_base(BASE_GRID[0]).index
mA = H.metrics(oos_A)

static_pnl = H.backtest(pd.Series(1.0, index=ret.index), ret, vxn)
static_on = static_pnl.reindex(oos_A.index).dropna()
mS = H.metrics(static_on)
aA, tA, bA = ols_alpha(oos_A, static_on)

print(f"\nOOS dates: {oos_A.index[0].date()} .. {oos_A.index[-1].date()}  "
      f"n={len(oos_A)}  blocks={len(picks_A)}")
print("\n=== BASELINE: un-gated family A on 2013+ window ===")
print(f"  sharpe={mA['sharpe']:.3f} t={mA['tstat']:.2f} skew={mA['skew']:.1f} n={mA['n']}")
print(f"  alpha-t vs static short-vol (same dates): {tA:.2f} (beta={bA:.3f})")
print(f"  static short-vol same dates: sharpe={mS['sharpe']:.3f} skew={mS['skew']:.1f}")
posA = oos_positions(picks_A, pnl_idx)
print(f"  breadth: short {float((posA>0).mean())*100:.1f}%  long {float((posA<0).mean())*100:.1f}%  "
      f"flat {float((posA==0).mean())*100:.1f}%")
print(f"  picks: {dict(Counter(picks_A))}")

# ---------------------------------------------------------------- candidates
CANDS = {
    "skew_z":   dict(gate=skew_z, thrs=[np.inf, 1.0, 0.5, 0.0],
                     desc="short-vol requires SKEW 252d z <= thr"),
    "volofvol": dict(gate=vov,    thrs=[np.inf, 2.0, 1.5, 1.0],
                     desc="short-vol requires 21d std(dVXN) <= thr"),
    "vxn_term": dict(gate=term,   thrs=[np.inf, 4.0, 2.0, 0.0],
                     desc="short-vol requires VXN - 63d mean <= thr"),
}

rows = []
rows.append(dict(run="A_ungated", sharpe=mA["sharpe"], tstat=mA["tstat"], skew=mA["skew"],
                 alpha_t_vs_static=tA, alpha_t_vs_A=np.nan, t_paired_diff=np.nan,
                 pct_short=float((posA > 0).mean()) * 100, pct_long=float((posA < 0).mean()) * 100,
                 corr_with_A=1.0, blocks_gate_on=0, n=mA["n"]))

for name, c in CANDS.items():
    gate, thrs = c["gate"], c["thrs"]
    grid = [(a, b, d, t) for (a, b, d) in BASE_GRID for t in thrs]   # 72 combos

    def build_gated(g, _gate=gate):
        return H.backtest(make_signal(g[:3], gate=_gate, thr=g[3]), ret, vxn)

    oos_g = H.walk_forward(build_gated, grid)
    picks_g = oos_g.attrs["picks"]
    mg = H.metrics(oos_g)

    # comparisons on identical OOS dates
    a_s, t_s, b_s = ols_alpha(oos_g, static_on)         # vs static short-vol
    diff = (oos_g - oos_A).dropna()                     # paired improvement test
    if diff.abs().max() == 0:                           # gate never picked -> identical series
        t_a, b_a, t_diff = np.nan, 1.0, 0.0             # OLS alpha-t is 0/0 garbage; mark NaN
    else:
        a_a, t_a, b_a = ols_alpha(oos_g, oos_A)         # vs un-gated A (does it ADD alpha?)
        t_diff = float(diff.mean() / (diff.std() / np.sqrt(len(diff)))) if diff.std() > 0 else 0.0
    corr = float(pd.concat([oos_g, oos_A], axis=1).dropna().corr().iloc[0, 1])

    pos_g = oos_positions(picks_g, pnl_idx, gate=gate)
    gate_on_blocks = sum(1 for p in picks_g if np.isfinite(p[3]))
    # days where the gate actually changed the position vs same params un-gated
    pos_ung = oos_positions([p[:3] for p in picks_g], pnl_idx)
    changed = float((pos_g != pos_ung).mean()) * 100

    print(f"\n=== CANDIDATE {name}: {c['desc']} ===")
    print(f"  grid: {len(grid)} combos (18 base x {len(thrs)} thr incl. off)")
    print(f"  OOS  sharpe={mg['sharpe']:.3f} t={mg['tstat']:.2f} skew={mg['skew']:.1f} n={mg['n']}")
    print(f"  vs static short-vol: alpha-t={t_s:.2f}")
    print(f"  vs un-gated A:       alpha-t={t_a:.2f} (beta={b_a:.3f})  "
          f"paired-diff t={t_diff:.2f}  corr={corr:.3f}")
    print(f"  breadth: short {float((pos_g>0).mean())*100:.1f}%  long {float((pos_g<0).mean())*100:.1f}%  "
          f"flat {float((pos_g==0).mean())*100:.1f}%   gate changed position on {changed:.1f}% of OOS days")
    print(f"  gate active in {gate_on_blocks}/{len(picks_g)} blocks; picks: {dict(Counter(picks_g))}")

    rows.append(dict(run=name, sharpe=mg["sharpe"], tstat=mg["tstat"], skew=mg["skew"],
                     alpha_t_vs_static=t_s, alpha_t_vs_A=t_a, t_paired_diff=t_diff,
                     pct_short=float((pos_g > 0).mean()) * 100, pct_long=float((pos_g < 0).mean()) * 100,
                     corr_with_A=corr, blocks_gate_on=gate_on_blocks, n=mg["n"]))

res = pd.DataFrame(rows)
out_csv = os.path.join(DATA, "p1_signals_results.csv")
res.to_csv(out_csv, index=False)
print(f"\nresults -> {out_csv}")

# ---------------------------------------------------------------- look-ahead self-audit
print("\n=== look-ahead self-audit (truncation test, skew_z gate) ===")
ok = True
for k in [1500, 2500, len(ret) - 10]:
    df2, ret2, vxn2, skew2 = df.iloc[:k], ret.iloc[:k], vxn.iloc[:k], skew.iloc[:k]
    f21 = H.fcast_vol(df2, ret2, "park21"); f10 = H.fcast_vol(df2, ret2, "park10")
    f42 = H.fcast_vol(df2, ret2, "park42")
    rich2, tr2 = vxn2 - f21, f10 - f42
    z2 = (skew2 - skew2.rolling(252).mean()) / skew2.rolling(252).std()
    for g in [(2.0, -2.0, 0.0, 0.5), (4.0, 0.0, 1.0, 1.0)]:
        s2 = pd.Series(0.0, index=ret2.index)
        sm = (rich2 >= g[0]) & (tr2 <= -g[2]) & (z2 <= g[3])
        lm = (rich2 <= g[1]) & (tr2 >= g[2])
        s2[sm.fillna(False)] = 1.0
        s2[lm.fillna(False)] = -1.0
        full = make_signal(g[:3], gate=skew_z, thr=g[3])
        ok &= bool((s2.iloc[-5:] == full.reindex(s2.index).iloc[-5:]).all())
print(f"truncation test: {'PASS' if ok else 'FAIL'}")

# ---------------------------------------------------------------- deflation note
print("\n=== DEFLATION NOTE (honest trial accounting) ===")
print("  new combos searched inside walk-forward: 3 candidates x 54 gated variants = 162")
print("  (the 18 thr=inf variants per candidate duplicate the existing family A).")
print("  Within-grid selection is internalized by walk-forward (picked on TRAIN only),")
print("  so OOS numbers are not directly inflated by the 162; BUT the researcher-level")
print("  choice 'which of 3 candidates (with hand-chosen gate direction and threshold")
print("  ranges) to adopt' is a selection over >=3 trials. A candidate should clear a")
print("  Bonferroni bar of p<0.05/3 (paired-diff |t|>~2.4) before being believed, and")
print("  this sits on top of the session-wide search (families A, B, 3 discarded fakes).")
print("  All P&L is the variance-swap PROXY: levels are inflated; only the RELATIVE")
print("  gated-vs-ungated comparison on identical dates is the claim being tested.")
