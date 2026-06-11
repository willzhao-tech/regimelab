# -*- coding: utf-8 -*-
"""
P1-8 DYNAMIC RISK MANAGEMENT on the blend OOS variance P&L (volarb_ledger_blend.csv).

All overlays are CAUSAL: every scaling applied to day-t pnl uses information through
the close of day t-1 only. No parameter is optimized here -- all constants (63d, 15%,
cap 3, 10%/5% hysteresis, 252d) are fixed a priori by the task spec.

(a) VOL TARGETING to 15% ann, 63d trailing vol of the pnl, shifted, leverage cap 3.
    Units problem: the ledger pnl is in fractional variance units (ann vol ~0.1% of
    notional), so the LITERAL leverage sigma_tgt/sigma63 is O(100) and a literal cap
    of 3 pins 100% of days -> constant 3x raw -> Sharpe identical to raw (degenerate).
    We report that literal diagnostic explicitly, then implement the economically
    meaningful version:
        base_t = sigma_tgt_daily / sigma_exp(t-1)     causal unit conversion, where
                 sigma_exp = expanding std of raw pnl (min 252 obs)
        m_t    = min(CAP, sigma_exp(t-1)/sigma63(t-1)) relative leverage, cap 3
        a_pnl_t = base_t * m_t * p_t
    "Pinned" = m_t at the cap. We also report the cap-free and base-only (constant
    relative leverage) Sharpes to isolate where any Sharpe change comes from.

(b) DRAWDOWN BRAKE (precise causal definition):
    cum_t   = cumsum of RAW pnl through close t
    dd_t    = cum_t - max(cum_{s<=t})  (<= 0)
    annv_t  = std(p, trailing 252d through t) * sqrt(252)   ("trailing 1y pnl-vol")
    depth_t = -dd_t / annv_t            drawdown in trailing-1y-ann-vol units
    State machine evaluated at close t, applied to day t+1's pnl:
        OFF -> ON  (halve)  when depth_t > 0.10
        ON  -> OFF (restore) when depth_t < 0.05   (recovery = depth back below
                                                    half the trigger; hysteresis
                                                    avoids flip-flopping)
    b_pnl_t = (0.5 if state(t-1)==ON else 1.0) * p_t
    The state machine runs on RAW equity (the signal), not on braked equity;
    live implementation on actual braked equity would recover more slowly.

(c) BOTH: c_pnl_t = base_t * m_t * brake_scale_t * p_t.

All variants are evaluated on the common window where every overlay is defined
(after the 252d+1 burn-in); raw is reported on both the full and common window.

Caveat (stated up front): risk overlays create no alpha. Any Sharpe change comes
from time-varying leverage interacting with vol clustering / skew of the same OOS
pnl stream. Incremental transaction costs of daily leverage changes are NOT
modeled (raw pnl already nets costs of signal-driven position changes only).
"""
import os
import numpy as np
import pandas as pd

DATA = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

TGT_ANN = 0.15
TGT_DAY = TGT_ANN / SQ
CAP = 3.0
VT_WIN = 63
REF_MIN = 252          # burn-in for expanding reference vol
DD_WIN = 252
DD_TRIG = 0.10         # enter brake
DD_EXIT = 0.05         # exit brake (hysteresis = trigger/2)
BRAKE = 0.5


def metrics(x, label):
    p = pd.Series(x).dropna()
    mu, sd = p.mean(), p.std()
    if len(p) < 60 or sd == 0:
        return dict(label=label, n=len(p))
    cum = p.cumsum()
    dd = cum - cum.cummax()
    return dict(label=label, n=int(len(p)),
                sharpe=float(mu / sd * SQ),
                tstat=float(mu / (sd / np.sqrt(len(p)))),
                skew=float(p.skew()),
                ann_vol=float(sd * SQ),
                maxdd_native=float(dd.min()),
                maxdd_over_annvol=float(-dd.min() / (sd * SQ)),
                worst_day_native=float(p.min()),
                worst_day_sigma=float(-p.min() / sd))


def main():
    led = pd.read_csv(os.path.join(DATA, "volarb_ledger_blend.csv"),
                      parse_dates=["Date"]).set_index("Date").sort_index()
    p = led["daily_pnl"].astype(float)

    # ---------- (a) vol targeting ----------
    vol63 = p.rolling(VT_WIN).std().shift(1)                  # through t-1
    ref = p.expanding(REF_MIN).std().shift(1)                 # causal long-run vol
    # literal-spec diagnostic: leverage in raw units
    lev_lit = (TGT_DAY / vol63).replace([np.inf, -np.inf], np.nan)
    # meaningful version: relative leverage, cap applies to the multiplier
    m = (ref / vol63).replace([np.inf, -np.inf], CAP).clip(upper=CAP)
    base = TGT_DAY / ref                                      # causal unit conversion
    a_pnl = base * m * p
    a_pnl_nocap = base * (ref / vol63).replace([np.inf, -np.inf], np.nan) * p
    norm_only = base * p                                      # base conversion only

    # ---------- (b) drawdown brake ----------
    cum = p.cumsum()
    dd = cum - cum.cummax()
    annv = p.rolling(DD_WIN).std() * SQ
    depth = (-dd / annv)                                      # NaN during burn-in
    dep = depth.values
    scale = np.ones(len(p))
    state = 0                                                 # 0=full, 1=braked
    for i in range(len(p)):
        scale[i] = BRAKE if state == 1 else 1.0               # state from close t-1
        d = dep[i]
        if np.isfinite(d):
            if state == 0 and d > DD_TRIG:
                state = 1
            elif state == 1 and d < DD_EXIT:
                state = 0
    b_scale = pd.Series(scale, index=p.index)
    b_pnl = b_scale * p

    # ---------- (c) combined ----------
    c_pnl = base * m * b_scale * p

    # ---------- common evaluation window ----------
    idx = a_pnl.dropna().index  # (a) has the longest burn-in (252d expanding + shift);
    # the brake cannot be ON before depth exists, so (b)/(c) are defined everywhere
    raw_c, a_c = p.reindex(idx), a_pnl.reindex(idx)
    b_c, c_c = b_pnl.reindex(idx), c_pnl.reindex(idx)
    norm_c, nocap_c = norm_only.reindex(idx), a_pnl_nocap.reindex(idx)

    rows = [metrics(p, "raw (full sample)"),
            metrics(raw_c, "raw (common window)"),
            metrics(a_c, "(a) vol-target 15%, cap 3"),
            metrics(b_c, "(b) drawdown brake"),
            metrics(c_c, "(c) both")]
    diag = [metrics(norm_c, "diag: base-conversion only (no 63d targeting)"),
            metrics(nocap_c, "diag: vol-target, NO cap")]
    tab = pd.DataFrame(rows).set_index("label")
    dtab = pd.DataFrame(diag).set_index("label")

    pinned = float((m.reindex(idx) >= CAP - 1e-12).mean())
    lev_med = float(lev_lit.reindex(idx).median())
    lev_min = float(lev_lit.reindex(idx).min())
    pct_braked = float((b_scale.reindex(idx) < 1.0).mean())
    n_brakes = int(((b_scale < 1.0) & (b_scale.shift(1) >= 1.0)).sum())

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.4g}")
    print("=== P1-8 dynamic risk management, blend OOS pnl",
          f"({idx[0].date()} .. {idx[-1].date()}, n={len(idx)}) ===\n")
    print(tab[["n", "sharpe", "tstat", "skew", "ann_vol", "maxdd_native",
               "maxdd_over_annvol", "worst_day_native", "worst_day_sigma"]])
    print("\n--- diagnostics ---")
    print(dtab[["sharpe", "ann_vol", "maxdd_over_annvol", "worst_day_sigma"]])

    print("\n--- (a) pinning check ---")
    print(f"LITERAL spec leverage tgt/vol63 (raw units): median {lev_med:,.0f}x, "
          f"min {lev_min:,.0f}x -> a literal cap of 3 binds on 100% of days, i.e. "
          f"constant 3x raw, Sharpe identical to raw (degenerate). Implemented cap "
          f"on the RELATIVE multiplier m = sigma_exp/sigma63 instead.")
    print(f"% days at cap (m == 3):           {pinned:.1%}")
    print(f"mean / median m:                  {float(m.reindex(idx).mean()):.2f} / "
          f"{float(m.reindex(idx).median()):.2f}")
    print(f"realized ann vol of (a):          {float(a_c.std()*SQ):.2%} (target 15%)")
    sh_raw = tab.loc['raw (common window)', 'sharpe']
    sh_a = tab.loc['(a) vol-target 15%, cap 3', 'sharpe']
    sh_nc = dtab.loc['diag: vol-target, NO cap', 'sharpe']
    print(f"Sharpe raw -> (a) -> no-cap:      {sh_raw:.2f} -> {sh_a:.2f} -> {sh_nc:.2f}")
    if sh_a > sh_raw * 1.15:
        print("FLAG: scaled Sharpe inflates >15% vs raw -- the gain comes from "
              "leveraging quiet (often flat-position) stretches and de-levering "
              "after vol spikes; it relies on pnl-vol clustering persisting and "
              "on costless daily leverage changes.")
    else:
        print("No material Sharpe inflation from vol targeting (within 15% of raw).")

    print("\n--- (b) brake stats ---")
    print(f"% days braked (half size):        {pct_braked:.1%}  "
          f"({n_brakes} brake episodes over full sample)")
    print("NOTE: the 10% (of 1y ann pnl-vol) trigger is shallow for a Sharpe ~1.8 "
          "stream, so the brake is on half the time; it acts as persistent "
          "deleveraging in choppy periods rather than a rare crash brake.")

    # --- sensitivity: (a) worst days cluster in 2007 when the expanding ref vol
    # had only ~1y of quiet data (burn-in artifact of the unit conversion).
    ref2 = p.expanding(2 * REF_MIN).std().shift(1)
    m2 = (ref2 / vol63).replace([np.inf, -np.inf], CAP).clip(upper=CAP)
    a2 = ((TGT_DAY / ref2) * m2 * p).dropna()
    r2 = metrics(a2, "x")
    rr2 = metrics(p.reindex(a2.index), "x")
    print("\n--- (a) burn-in sensitivity (504d expanding ref, window from "
          f"{a2.index[0].date()}) ---")
    print(f"(a) 504d: sharpe {r2['sharpe']:.2f}, maxDD/annvol {r2['maxdd_over_annvol']:.2f}, "
          f"worst-day {r2['worst_day_sigma']:.1f} sigma  |  raw same window: "
          f"sharpe {rr2['sharpe']:.2f}, maxDD/annvol {rr2['maxdd_over_annvol']:.2f}, "
          f"worst-day {rr2['worst_day_sigma']:.1f} sigma")

    out = pd.DataFrame({
        "raw_pnl": p, "vol63_shift": vol63, "ref_expand_shift": ref,
        "lev_literal": lev_lit, "m_rel_lev": m, "a_pnl": a_pnl,
        "dd_raw": dd, "depth_volunits": depth, "brake_scale": b_scale,
        "b_pnl": b_pnl, "c_pnl": c_pnl}).reindex(p.index)
    out.to_csv(os.path.join(DATA, "p1_risk_ledger.csv"))
    tab.to_csv(os.path.join(DATA, "p1_risk_summary.csv"))
    print(f"\nartifacts: {os.path.join(DATA, 'p1_risk_ledger.csv')}, "
          f"{os.path.join(DATA, 'p1_risk_summary.csv')}")


if __name__ == "__main__":
    main()
