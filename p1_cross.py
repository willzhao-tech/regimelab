# -*- coding: utf-8 -*-
"""
p1_cross.py — TASK P1-9: CROSS-MARKET VALIDATION of the long-short vol-arb families.

Replicates Family A (regime_combo), Family B (range_forecast) and the 50/50 blend
with IDENTICAL grids, costs and walk-forward mechanics (train=1260d -> test=252d,
pick best TRAIN Sharpe, concatenate TEST blocks only) on two new markets, using
the same proxy instrument (1-day variance at the vol-index strike):

  1) SPX / VIX      trimmed to 2001+ to match the NQ/VXN study span (the big test,
                    ~25y of data). A 1999+ extended run is added as robustness
                    (clearly labelled; same grids, no new search).
  2) SX5E / VSTOXX  2013+. CAVEAT: the VSTOXX series here is a FUTURES front-month
                    (not the spot index): the strike embeds the futures basis /
                    rolldown, richness = futures - park21 is regime-biased relative
                    to a spot-index version, and contract rolls put jumps in the
                    series. Short sample (~7y OOS) => supporting evidence only.

NO new parameters and NO new grid search anywhere in this file: grids are copied
verbatim from the NQ/VXN study (A: 18 combos, B: 9 combos), so the multiplicity
is unchanged from the original study — this is pure out-of-family replication.

Universality criterion (fixed before running): alpha-t >= 2 vs that market's own
static short-vol on the SAME OOS dates AND positive OOS skew, judged on SPX/VIX.

Causality: identical to the audited NQ/VXN scripts — park forecasts are
harness-shifted (value at t uses data through t-1), the vol index enters at
close t, and backtest() shifts the signal one more day before P&L accrues.
A programmatic truncation audit is re-run per market below. No caps, no
clipping, no tail edits; uncapped variance P&L including every crash day.

Output: C:\\Users\\ASUS\\Desktop\\claude doc\\1\\p1_crossmarket.csv
Run:    .venv python p1_cross.py
"""
import os
import sys
from collections import Counter

sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd

import volarb_harness as H

DATA = r"C:\Users\ASUS\Desktop\claude doc\1"
OUT_CSV = os.path.join(DATA, "p1_crossmarket.csv")
SQ = np.sqrt(252.0)
TRAIN, TEST = 1260, 252

# grids copied VERBATIM from va_regime_combo.py / va_range_forecast.py
GRID_A = [(a, b, c) for a in (2.0, 4.0, 6.0) for b in (0.0, -2.0) for c in (0.0, 1.0, 2.0)]
GRID_B = [(b1, b2) for b1 in (0.8, 1.0, 1.2) for b2 in (1.3, 1.6, 2.0)]
N_COMBOS = len(GRID_A) + len(GRID_B)   # 27, unchanged from the original study


def load_market(und_file, vol_file, start=None):
    """Same pipeline as H.load(), parameterized by file: align on common dates."""
    df = (pd.read_csv(os.path.join(DATA, und_file), parse_dates=["Date"])
            .set_index("Date").sort_index())
    vol = (pd.read_csv(os.path.join(DATA, vol_file), parse_dates=["Date"])
             .set_index("Date")["Close"].dropna().sort_index())
    if start is not None:
        df = df.loc[df.index >= start]
        vol = vol.loc[vol.index >= start]
    ret = df["Close"].pct_change()
    idx = ret.index.intersection(vol.index)
    return df.loc[idx], ret.loc[idx], vol.loc[idx]


def ols_alpha_t(p, base):
    """OLS  p = a + b*base  on common dates; returns (t_alpha, beta, n)."""
    yx = pd.concat([p, base], axis=1).dropna().values
    y, x = yx[:, 0], yx[:, 1]
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    ta = b[0] / np.sqrt((r @ r / (len(y) - 2)) * np.linalg.inv(X.T @ X)[0, 0])
    return float(ta), float(b[1]), len(y)


def rebuild_oos_positions(sig_fn, picks, pnl_idx):
    """Replicate walk_forward's block structure; return the HELD position
    (signal shifted by 1, exactly as H.backtest trades it) on OOS dates."""
    parts, start, b = [], TRAIN, 0
    while start + TEST <= len(pnl_idx):
        te = pnl_idx[start:start + TEST]
        pos = sig_fn(picks[b]).clip(-1, 1).shift(1).fillna(0.0)
        parts.append(pos.reindex(te))
        start += TEST
        b += 1
    return pd.concat(parts)


def run_market(market, und_file, vol_file, start):
    print("\n" + "=" * 78)
    print(f"MARKET {market}: {und_file} + {vol_file}  (start filter: {start})")
    print("=" * 78)
    df, ret, vol = load_market(und_file, vol_file, start)
    print(f"aligned days: {len(ret)}  span {ret.index[0].date()}..{ret.index[-1].date()}")

    # ---- causal building blocks (identical formulas to NQ/VXN study) ----
    fc21 = H.fcast_vol(df, ret, "park21")   # value at t uses data through t-1
    fc10 = H.fcast_vol(df, ret, "park10")
    fc42 = H.fcast_vol(df, ret, "park42")
    richness = vol - fc21                   # vol-index close t minus trailing fcast
    trend = fc10 - fc42                     # >0: range vol rising
    rng = np.log(df["High"] / df["Low"]) * 100.0
    be = vol / SQ                           # implied daily breakeven, %

    def sig_A(g):
        r_hi, r_lo, d = g
        s = pd.Series(0.0, index=ret.index)
        s[((richness >= r_hi) & (trend <= -d)).fillna(False)] = 1.0
        s[((richness <= r_lo) & (trend >= d)).fillna(False)] = -1.0
        return s

    def sig_B(g):
        b1, b2 = g
        s = pd.Series(0.0, index=ret.index)
        ok = rng.notna() & be.notna()
        s[ok & (rng < b1 * be)] = 1.0
        s[ok & (rng > b2 * be)] = -1.0
        return s

    # ---- walk-forward, identical harness mechanics ----
    oos_A = H.walk_forward(lambda g: H.backtest(sig_A(g), ret, vol), GRID_A, TRAIN, TEST)
    oos_B = H.walk_forward(lambda g: H.backtest(sig_B(g), ret, vol), GRID_B, TRAIN, TEST)
    picks_A, picks_B = oos_A.attrs["picks"], oos_B.attrs["picks"]
    common = oos_A.index.intersection(oos_B.index)
    pnl_A, pnl_B = oos_A.reindex(common).dropna(), oos_B.reindex(common).dropna()
    blend = (0.5 * pnl_A + 0.5 * pnl_B).dropna()
    static = H.backtest(pd.Series(1.0, index=ret.index), ret, vol).reindex(common).dropna()

    # ---- positions held OOS (diagnostic: does the gate even fire here?) ----
    pnl_idx = H.backtest(sig_A(GRID_A[0]), ret, vol).index
    pos_A = rebuild_oos_positions(sig_A, picks_A, pnl_idx).reindex(common)
    pos_B = rebuild_oos_positions(sig_B, picks_B, pnl_idx).reindex(common)
    pos_blend = 0.5 * pos_A.fillna(0.0) + 0.5 * pos_B.fillna(0.0)
    pos_static = pd.Series(1.0, index=common)

    rows = []
    for name, p, pos in [("A_regime_combo", pnl_A, pos_A),
                         ("B_range_forecast", pnl_B, pos_B),
                         ("blend_50_50", blend, pos_blend),
                         ("static_short_vol", static, pos_static)]:
        m = H.metrics(p)
        if name != "static_short_vol":
            ta, beta, _ = ols_alpha_t(p, static)
        else:
            ta, beta = float("nan"), float("nan")
        n = len(pos.dropna())
        row = dict(market=market, strategy=name,
                   oos_start=str(p.index[0].date()), oos_end=str(p.index[-1].date()),
                   n_days=m["n"], sharpe=round(m["sharpe"], 3), tstat=round(m["tstat"], 2),
                   skew=round(m["skew"], 2), worst_ratio=round(m["worst_ratio"], 1),
                   maxdd_units=round(m["maxdd_units"], 2),
                   alpha_t_vs_static=round(ta, 2) if ta == ta else "",
                   beta_vs_static=round(beta, 3) if beta == beta else "",
                   pct_short_vol=round(float((pos > 0).sum()) / n * 100, 1),
                   pct_long_vol=round(float((pos < 0).sum()) / n * 100, 1),
                   pct_flat=round(float((pos == 0).sum()) / n * 100, 1),
                   n_blocks=len(picks_A))
        rows.append(row)
        print(f"  {name:<18} Sharpe {m['sharpe']:+6.2f}  t {m['tstat']:+5.1f}  "
              f"skew {m['skew']:+6.1f}  alpha-t {ta:+5.2f}  beta {beta:+6.3f}  "
              f"short/long/flat {row['pct_short_vol']}/{row['pct_long_vol']}/{row['pct_flat']}%")

    print(f"  OOS span {common[0].date()}..{common[-1].date()}  ({len(common)} days, "
          f"{len(picks_A)} blocks)")
    for fam, picks in [("A", picks_A), ("B", picks_B)]:
        cnt = Counter(picks).most_common()
        print(f"  picks {fam}: " + "; ".join(f"{g}x{c}" for g, c in cnt))

    # ---- look-ahead truncation audit (signal at t unchanged if future removed) ----
    cut = len(ret) // 2
    df2, ret2, vol2 = df.iloc[:cut], ret.iloc[:cut], vol.iloc[:cut]
    f21 = H.fcast_vol(df2, ret2, "park21")
    f10 = H.fcast_vol(df2, ret2, "park10")
    f42 = H.fcast_vol(df2, ret2, "park42")
    rich2, tr2 = vol2 - f21, f10 - f42
    rng2, be2 = np.log(df2["High"] / df2["Low"]) * 100.0, vol2 / SQ
    ok = True
    for g in (GRID_A[0], GRID_A[-1]):
        r_hi, r_lo, d = g
        s2 = pd.Series(0.0, index=ret2.index)
        s2[((rich2 >= r_hi) & (tr2 <= -d)).fillna(False)] = 1.0
        s2[((rich2 <= r_lo) & (tr2 >= d)).fillna(False)] = -1.0
        ok &= bool((s2 == sig_A(g).iloc[:cut]).all())
    for g in (GRID_B[0], GRID_B[-1]):
        b1, b2 = g
        s2 = pd.Series(0.0, index=ret2.index)
        ok2 = rng2.notna() & be2.notna()
        s2[ok2 & (rng2 < b1 * be2)] = 1.0
        s2[ok2 & (rng2 > b2 * be2)] = -1.0
        ok &= bool((s2 == sig_B(g).iloc[:cut]).all())
    print(f"  look-ahead truncation audit: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError(f"look-ahead audit FAILED on {market}")
    return rows, dict((r["strategy"], r) for r in rows)


# ============================ run all markets ============================
all_rows = []
rows_spx, res_spx = run_market("SPX_VIX", "SPX_all_history.csv", "VIX_all_history.csv",
                               start="2001-01-01")
all_rows += rows_spx
rows_ext, _ = run_market("SPX_VIX_ext1999", "SPX_all_history.csv", "VIX_all_history.csv",
                         start=None)
all_rows += rows_ext
rows_eu, res_eu = run_market("SX5E_VSTOXX", "SX5E_all_history.csv", "VSTOXX_all_history.csv",
                             start=None)
all_rows += rows_eu

out = pd.DataFrame(all_rows)
out.to_csv(OUT_CSV, index=False)
print(f"\nsaved -> {OUT_CSV}  ({len(out)} rows)")

# ============================ verdict ============================
print("\n" + "=" * 78)
print("UNIVERSALITY VERDICT (criterion fixed ex-ante: SPX/VIX alpha-t>=2 AND skew>0)")
print("=" * 78)
for name in ("A_regime_combo", "B_range_forecast", "blend_50_50"):
    r = res_spx[name]
    ta = float(r["alpha_t_vs_static"])
    ok = (ta >= 2.0) and (float(r["skew"]) > 0)
    print(f"  SPX/VIX {name:<18} alpha-t {ta:+5.2f}  skew {r['skew']:+6.2f}  "
          f"Sharpe {r['sharpe']:+5.2f}  -> {'PASS' if ok else 'FAIL'}")
print("  SX5E/VSTOXX (supporting only — short sample, futures-based vol index):")
for name in ("A_regime_combo", "B_range_forecast", "blend_50_50"):
    r = res_eu[name]
    print(f"    {name:<18} alpha-t {r['alpha_t_vs_static']:+5.2f}  skew {r['skew']:+6.2f}  "
          f"Sharpe {r['sharpe']:+5.2f}")

print("""
NOTES (where the numbers come from / what breaks):
 - Proxy instrument: 1-day variance struck at the vol-index prior close, cost
   0.5 vol-pt per unit turnover. NOT directly tradable; inflates Sharpe LEVELS
   for strategy and static baseline alike. The robust claims are RELATIVE:
   alpha-t vs static short-vol and the skew sign, on identical OOS dates.
 - VIX/VSTOXX indexes are 30d implied vol; the proxy treats them as a 1-day
   variance strike, so a persistent term-structure premium flatters BOTH the
   strategy and the static baseline equally — another reason only the relative
   numbers travel.
 - SX5E/VSTOXX: vol series is a FUTURES front-month (basis + roll jumps in
   richness and in the strike); sample is ~7y OOS. Supporting evidence only.
 - Grids/walk-forward copied verbatim (27 combos total, unchanged multiplicity).
   No new search was run for this task; nothing here was tuned to these markets.
""")
