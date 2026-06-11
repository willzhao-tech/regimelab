"""
LONG-SHORT NQ strategy -- family: time_of_day.

Thesis (structural, NOT data-mined per-hour):
  Equity-index futures exhibit a well-documented "overnight drift": a large share
  of the long-run return accrues OUTSIDE the U.S. regular trading hours (RTH),
  while the RTH session is comparatively flat / mean-reverting and bears most of
  the realized volatility. We exploit this by going LONG during the overnight
  Globex session and FLAT-or-SHORT during RTH.

Causality:
  - Signal for bar t is a pure function of bar t's CLOCK HOUR (known at the open
    of bar t, i.e. fully causal -- the hour label is the bar's start time).
  - H.backtest shifts the signal by 1 (pos = signal.shift(1)) and applies it to
    ret of the next bar. So even though the hour itself is known, the harness
    still only lets the position earn the *following* bar's return -- no leakage.
  - The first-hour-momentum gate (optional) uses the SIGN of the most recent
    completed RTH-open bar return, again strictly from the past.

Time zones:
  Raw index carries mixed -05:00/-04:00 offsets (US/Eastern w/ DST). We convert
  to America/New_York so hour-of-day is a stable session clock.
  NQ RTH = 09:30-16:00 ET. With hour-start bar labels: RTH bars ~ hours 9..15;
  overnight/Globex ~ hours 18..23 and 0..8; 16,17 = post-close handoff (treated
  as flat unless RTH side spills, see params).

Optimization protocol:
  - 50/50 time split: first ~3 months TRAIN, last ~3 months TEST.
  - Grid-search a SMALL param set on TRAIN; pick best TRAIN Sharpe.
  - Report TRAIN and held-out TEST at 1x, plus leverage table on TEST.
  - Count combos for deflation. TEST Sharpe is the only number that counts.
"""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import ls_harness as H
import pandas as pd, numpy as np
import itertools

PPY = 252 * 23  # ~23 tradable hours/day

# ---------------------------------------------------------------- load + tz
df = H.load_1h()
df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
df = df.sort_index()
ret = df["Close"].pct_change().fillna(0.0)
hour = pd.Series(df.index.hour, index=df.index)

# Session masks (Eastern clock). RTH bars = 9..15 (covers 09:30-16:00 handoff).
RTH_HOURS = set(range(9, 16))          # 9,10,11,12,13,14,15
OVN_HOURS = set(range(18, 24)) | set(range(0, 9))  # 18..23 + 0..8 (Globex)
# hours 16,17 = post-close / pre-Globex gap -> flat by default.

is_rth = hour.isin(RTH_HOURS)
is_ovn = hour.isin(OVN_HOURS)

# First-hour RTH momentum: sign of the bar labelled hour==9 (09:00-10:00 ET),
# forward-filled so it is available for the *rest* of that RTH session and is
# strictly causal (uses only that already-closed first-hour bar).
first_hr_bar = (hour == 9)
fh_ret = ret.where(first_hr_bar)                 # ret only on the 9:00 bar
fh_sign = np.sign(fh_ret).ffill().fillna(0.0)    # last known first-hour sign


def build_signal(ovn_side, rth_side, use_fh_gate):
    """
    Causal time-of-day signal in [-1,1].
      ovn_side : position during overnight Globex (+1 long is the thesis)
      rth_side : base position during RTH (0 flat, or -1 short the chop)
      use_fh_gate : if True, RTH position = rth_strength * first-hour momentum sign
                    (momentum: ride the direction set by the RTH open). If the
                    first-hour was up, lean long into RTH; if down, lean short.
    """
    s = pd.Series(0.0, index=ret.index)
    s[is_ovn] = ovn_side
    if use_fh_gate:
        # magnitude from rth_side (use its abs as strength), direction from fh_sign
        strength = abs(rth_side)
        s[is_rth] = strength * fh_sign[is_rth]
    else:
        s[is_rth] = rth_side
    return s.clip(-1, 1)


# ---------------------------------------------------------------- split
ret_tr, ret_te = H.split(ret)

# ---------------------------------------------------------------- grid
# Keep it SMALL & structural. ovn_side in {0,+1}: flat-overnight is the null;
# +1 is the drift thesis. (-1 short-overnight is anti-thesis, included to be
# honest about the search.) rth_side in {0,-0.5,-1}: flat or fade the RTH chop.
# fh_gate in {False,True}: optional first-hour momentum overlay on RTH.
OVN_SIDES = [0.0, 1.0, -1.0]
RTH_SIDES = [0.0, -0.5, -1.0]
FH_GATES  = [False, True]

results = []
combos = 0
for ovn_side, rth_side, fh in itertools.product(OVN_SIDES, RTH_SIDES, FH_GATES):
    # skip degenerate all-flat
    if ovn_side == 0.0 and rth_side == 0.0:
        continue
    # if fh_gate on but rth_side==0 -> no RTH exposure, redundant with fh off
    if fh and rth_side == 0.0:
        continue
    combos += 1
    sig = build_signal(ovn_side, rth_side, fh)
    sig_tr = sig.reindex(ret_tr.index)
    bt = H.backtest(sig_tr, ret_tr, leverage=1.0, cost_bps=1.0, ppy=PPY)
    results.append(dict(ovn_side=ovn_side, rth_side=rth_side, fh_gate=fh,
                        train_sharpe=bt["sharpe"], train_ret=bt["ret"]))

res = pd.DataFrame(results).sort_values("train_sharpe", ascending=False)
print("=== TRAIN grid (sorted by Sharpe) ===")
print(res.to_string(index=False))
print(f"\n#param-combos tried (TRAIN): {combos}")

best = res.iloc[0]
ovn_b, rth_b, fh_b = best["ovn_side"], best["rth_side"], bool(best["fh_gate"])
print(f"\nBEST PARAMS: ovn_side={ovn_b}, rth_side={rth_b}, fh_gate={fh_b}")

# ---------------------------------------------------------------- report best on TRAIN + TEST
sig = build_signal(ovn_b, rth_b, fh_b)
sig_tr = sig.reindex(ret_tr.index)
sig_te = sig.reindex(ret_te.index)

bt_tr = H.backtest(sig_tr, ret_tr, leverage=1.0, cost_bps=1.0, ppy=PPY)
bt_te = H.backtest(sig_te, ret_te, leverage=1.0, cost_bps=1.0, ppy=PPY)
print("\n=== BEST @1x ===")
print("TRAIN:", {k: round(v, 4) if isinstance(v, float) else v for k, v in bt_tr.items()})
print("TEST :", {k: round(v, 4) if isinstance(v, float) else v for k, v in bt_te.items()})

# ---------------------------------------------------------------- leverage table on TEST
print("\n=== LEVERAGE TABLE on TEST (5/10/15/20x) ===")
lt = H.lev_table(sig_te, ret_te, cost_bps=1.0, ppy=PPY)
for L, d in lt.items():
    print(f"{L:>2}x: ret={d['ret']:+.4f}  sharpe={d['sharpe']:.3f}  maxdd={d['maxdd']:.4f}  "
          f"ruined={'Y' if d['ruined'] else 'N'}  ruin_dt={d['ruin_dt'] if d['ruined'] else '-'}")

# ---------------------------------------------------------------- diagnostics
print("\n=== diagnostics ===")
print("TEST bars:", len(ret_te), " TRAIN bars:", len(ret_tr))
print("avg |signal| TEST:", float(sig_te.abs().mean()).__round__(4))
print("frac long TEST:", float((sig_te > 0).mean()).__round__(4),
      " frac short TEST:", float((sig_te < 0).mean()).__round__(4),
      " frac flat TEST:", float((sig_te == 0).mean()).__round__(4))
