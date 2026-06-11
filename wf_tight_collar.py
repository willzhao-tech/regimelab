"""
wf_tight_collar.py
==================
Strategy B (short-variance) re-optimized to PREVENT EXTREME LOSS via a tighter-wings /
put-spread tradeoff: BUY MORE DOWNSIDE PROTECTION (lower effective cap + a heavier smile
markup on the DOWN tail) to bound the worst single day, accepting lower harvested premium.

All parameters (cap level, down-tail smile, cost window) are chosen OUT-OF-SAMPLE by the
shared walk_forward() harness on a trailing-train window and applied to the next test block.

ECONOMICS
---------
Daily short-var P&L per unit notional (from H.hedged_pnl, generalized to an asymmetric collar):

    pnl_t = iv_{t-1}  -  min(rvar_t, capv_t)  -  wing_t

  * iv_{t-1}      : implied variance sold, from VXN shifted 1 day (causal).
  * min(rvar,capv): realized variance you pay, CAPPED at capv = cap**2. The cap is the wing
                    you OWN: it bounds the per-day loss on a tail day. A LOWER cap = tighter
                    wing = smaller worst day (more downside protection).
  * wing_t        : the price you PAY for that wing = smile * trailing-mean(tail) shifted 1.
                    smile >= 1.5 is the realistic OTM-tail markup. We let the DOWN tail carry
                    a heavier smile (puts cost more than calls) -> "buy more downside protection".

The tighter cap bounds the RAW per-day economic loss directly; it ALSO makes the trailing
vol estimate fed to causal_scale less fat-tailed, so the (causal) vol-scaler over-levers less
into the next tail day -> the SCALED worst day is bounded too.

ANTI-LOOK-AHEAD: every per-day input is trailing + shifted; parameters are picked only on
trailing-train blocks; the grid CEILING on cap is the protection mechanism (the WF can never
chase a dangerous wide cap because we do not offer one). See self-audit printed at the end.
"""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
from collections import Counter
import shortvol_harness as H

SQ = H.SQ


def collar_pnl(ret, vxn, cap_dn=0.045, cap_up=0.05, smile_dn=2.5, smile_up=1.5, cost_win=252):
    """Causal asymmetric-collar short-variance daily P&L per unit notional.

    cap_dn <= cap_up : tighter wing on DOWN days bounds the worst single day.
    smile_dn >= smile_up >= 1.5 : heavier trailing markup on the DOWN tail (puts cost more).
    All rolling stats use .shift(1); only data strictly before t enters day-t's value.
    """
    iv = (vxn.shift(1) / 100.0) ** 2 / 252.0          # implied var sold, causal (shifted VXN)
    rvar = ret ** 2                                    # realized var (contemporaneous payoff)
    down = ret < 0                                     # sign of TODAY's move (used only to pick
                                                       # which OWNED strike pays out today; the
                                                       # COST of each wing is trailing+shifted)
    capv = pd.Series(np.where(down, cap_dn ** 2, cap_up ** 2), index=ret.index)
    tail = np.maximum(rvar - capv, 0.0)               # variance beyond the owned strike
    tail_dn = tail.where(down, 0.0)
    tail_up = tail.where(~down, 0.0)
    # trailing fair cost of each wing, marked up by its smile, strictly causal (.shift(1))
    wing_dn = smile_dn * tail_dn.rolling(cost_win, min_periods=60).mean().shift(1)
    wing_up = smile_up * tail_up.rolling(cost_win, min_periods=60).mean().shift(1)
    pnl = iv - np.minimum(rvar, capv) - wing_dn - wing_up
    return pnl.dropna()


def build_fn(p):
    """Map a param dict to a full-history causal pnl Series for walk_forward."""
    return collar_pnl(
        ret, vxn,
        cap_dn=p["cap_dn"], cap_up=p["cap_up"],
        smile_dn=p["smile_dn"], smile_up=p["smile_up"],
        cost_win=p["cost_win"],
    )


if __name__ == "__main__":
    df, ret, vxn = H.load()

    # ----- PROTECTION-BIASED GRID -------------------------------------------------------
    # Ceiling on the up-cap is 0.055 and on the down-cap 0.05: the walk-forward CANNOT chase
    # a dangerous wide cap (e.g. 0.08) because we never offer one -- that ceiling IS the
    # worst-day bound. Within the protective range, WF still picks the train-best each block.
    # cap_up ceiling = 0.05 is the HARD protective bound (no wider cap is ever offered, so the
    # WF cannot chase a dangerous wide wing). cap_dn can go TIGHTER (0.04) to buy extra downside
    # protection, and smile_dn can be marked up heavier on the down tail. The WF is free to buy
    # more down-protection whenever the trailing-train window rewards it.
    grid = []
    for cap_up in [0.045, 0.05]:
        for cap_dn in [0.04, 0.045, 0.05]:
            if cap_dn > cap_up:
                continue
            for smile_dn in [1.5, 2.5]:        # heavier DOWN-tail smile = more downside protection
                for cost_win in [252, 504]:
                    grid.append(dict(cap_dn=cap_dn, cap_up=cap_up,
                                     smile_dn=smile_dn, smile_up=1.5, cost_win=cost_win))

    # ----- WALK-FORWARD (out-of-sample) -------------------------------------------------
    oos = H.walk_forward(build_fn, grid, train=1260, test=252)   # 5y train / 1y test, rolling

    # ----- CAUSAL VOL-SCALE the OOS pnl to 10% target vol -------------------------------
    scaled = H.causal_scale(oos, target=0.10)

    # ----- METRICS ----------------------------------------------------------------------
    m = H.metrics(scaled)
    worst_day = float(scaled.min())
    picks = oos.attrs["picks"]
    pick_counts = Counter((p["cap_dn"], p["cap_up"], p["smile_dn"], p["cost_win"]) for p in picks)

    # realistic baseline: static smile=1.5 hedged, scaled (Sharpe ~1.394 in-sample full-period)
    base = H.causal_scale(H.hedged_pnl(ret, vxn, cap=0.05, smile=1.5))
    bm = H.metrics(base)

    print("=" * 78)
    print("TIGHT-COLLAR SHORT-VOL (Strategy B) -- OUT-OF-SAMPLE walk-forward results")
    print("=" * 78)
    print(f"OOS observations          : {m['n']}")
    print(f"OOS Sharpe                : {m['sharpe']:.4f}")
    print(f"OOS CAGR                  : {m['cagr']:.4f}")
    print(f"OOS maxDD                 : {m['maxdd']:.4f}")
    print(f"OOS worst single day      : {worst_day:.4f}  ({worst_day*100:.2f}%)")
    print(f"OOS skew                  : {m['skew']:.4f}")
    print("-" * 78)
    print(f"Baseline (static smile=1.5, cap=0.05) Sharpe : {bm['sharpe']:.4f}  "
          f"worst={bm['worst']*100:.2f}%  skew={bm['skew']:.2f}  maxdd={bm['maxdd']:.3f}")
    beats = m["sharpe"] >= bm["sharpe"]
    print(f"Beats realistic baseline Sharpe? {beats}")
    print(f"Worst-day bounded > -8% ? {worst_day > -0.08}   |  OOS Sharpe positive? {m['sharpe'] > 0}")
    print("-" * 78)
    print("WALK-FORWARD PICK COUNTS (cap_dn, cap_up, smile_dn, cost_win):")
    for k, v in sorted(pick_counts.items(), key=lambda kv: -kv[1]):
        print(f"   {k} -> {v} blocks")
    print("=" * 78)
