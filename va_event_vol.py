"""
va_event_vol.py -- LONG-SHORT VOL-ARB on NQ/VXN, family: event_vol (scheduled-event calendar).

Idea: short vol into scheduled macro events (NFP day via the deterministic BLS
3rd-Friday-after-refweek rule from examples/14; FOMC day-before+day-of from
fomc_dates.csv), flat or baseline-short otherwise.

Causality:
  - The signal uses NO market data at all -- it is a pure deterministic calendar.
    NFP dates are computed from the BLS rule (knowable years in advance);
    FOMC dates come from the published schedule (announced ~1y in advance).
  - The harness backtest() shifts the signal: pos_t = s_{t-1}. To hold a short-vol
    position ON event day d we set s at trading day d-1 (and d-2 for the day-before
    leg). All placements are at index positions strictly before the position day.
  - Caveat (flagged, quantified below): fomc_dates.csv contains ~7 UNSCHEDULED /
    emergency dates (2019-10-11; 2020-03-03/15/23/31; 2020-08-27; 2025-08-22) that
    were not knowable ex ante. They are KEPT (no hindsight pruning). These are
    vol-explosion days where short-vol loses, so the inclusion biases AGAINST the
    strategy, not for it; their realized P&L contribution is printed.
  - No caps / clips / winsorization anywhere: catastrophe days hit the book in full.
  - Parameters picked exclusively via H.walk_forward (trailing 5y train -> next 1y OOS).

Run:
  "C:\\Users\\ASUS\\Desktop\\claude doc\\market study\\regimelab\\regimelab\\regimelab\\.venv\\Scripts\\python.exe" va_event_vol.py
"""
import os, sys, datetime as dt
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
import volarb_harness as H

DATA = r"C:\Users\ASUS\Desktop\claude doc\1"
UNSCHEDULED = pd.DatetimeIndex(["2019-10-11", "2020-03-03", "2020-03-15",
                                "2020-03-23", "2020-03-31", "2020-08-27",
                                "2025-08-22"])


# ---------------------------------------------------------------- NFP calendar
def nfp_dates(start, end):
    """BLS rule (examples/14): Employment Situation released the 3rd Friday after the
    reference week (Sun-Sat week containing the 12th), shifted off observed July-4."""
    out = []
    y, m = start.year - 1, start.month
    for _ in range((end.year - start.year + 2) * 12):
        ref12 = dt.date(y, m, 12)
        sat = ref12 + dt.timedelta(days=(5 - ref12.weekday()) % 7)
        first_fri = sat + dt.timedelta(days=((4 - sat.weekday()) % 7) or 7)
        rel = first_fri + dt.timedelta(days=14)
        jul4 = dt.date(rel.year, 7, 4)
        if rel.month == 7 and jul4.weekday() == 5 and rel == jul4 - dt.timedelta(days=1):
            rel -= dt.timedelta(days=1)
        out.append(rel)
        m += 1
        if m > 12:
            m = 1; y += 1
    return pd.DatetimeIndex([d for d in out if start.date() <= d <= end.date()])


# ------------------------------------------------------------------- load data
df, ret, vxn = H.load()
idx = ret.index

NFP = nfp_dates(idx.min(), idx.max())
fomc_raw = pd.read_csv(os.path.join(DATA, "fomc_dates.csv"), parse_dates=["fomc_date"])["fomc_date"]
FOMC = pd.DatetimeIndex(sorted(d for d in fomc_raw if idx.min() <= d <= idx.max()))


def event_day_positions(dates):
    """Map event dates -> integer positions of the trading day on/after each date."""
    loc = idx.get_indexer(pd.DatetimeIndex(dates), method="bfill")
    return sorted({int(p) for p in loc if 0 <= p < len(idx)})


NFP_POS = event_day_positions(NFP)
FOMC_POS = event_day_positions(FOMC)
UNSCHED_POS = set(event_day_positions([d for d in UNSCHEDULED if idx.min() <= d <= idx.max()]))

print(f"sample {idx.min().date()}..{idx.max().date()}  n={len(idx)}")
print(f"events in sample: NFP {len(NFP_POS)}  FOMC {len(FOMC_POS)} "
      f"(of which unscheduled/emergency kept: {len(UNSCHED_POS)})")


# ------------------------------------------------------------------ signal gen
def make_signal(g):
    """s_t in [0,1]; harness shifts -> pos_{t+1}=s_t. Calendar-only, fully causal."""
    if g["events"] == "none":                       # explicit static-short control
        return pd.Series(1.0, index=idx)
    s = np.full(len(idx), float(g["baseline"]))
    days = set()
    plist = []
    if g["events"] in ("nfp", "both"):
        plist += NFP_POS
    if g["events"] in ("fomc", "both"):
        plist += FOMC_POS
    for p in plist:
        days.add(p)                                  # hold short ON event day p
        if g["window"] == "dm1d0":
            days.add(p - 1)                          # ... and on the day before
    for d in days:
        if d - 1 >= 0:
            s[d - 1] = 1.0                           # signal one day ahead of position
    return pd.Series(s, index=idx)


def build_fn(g):
    return H.backtest(make_signal(g), ret, vxn)


# ------------------------------------------------------------------------ grid
grid = [dict(events=e, window=w, baseline=b)
        for e in ("nfp", "fomc", "both")
        for w in ("d0", "dm1d0")
        for b in (0.0, 0.5)]
grid.append(dict(events="none", window="-", baseline=1.0))   # static short control
print(f"combos tried: {len(grid)}")

# ---------------------------------------------------------------- walk-forward
oos = H.walk_forward(build_fn, grid)
picks = oos.attrs["picks"]

# static always-short on the SAME OOS dates
static_pnl = H.backtest(pd.Series(1.0, index=idx), ret, vxn)
static_oos = static_pnl.reindex(oos.index).dropna()

# ------------------------------------------- reconstruct OOS positions per block
pnl0 = build_fn(grid[0])
pidx = pnl0.index
train, test = 1260, 252
pos_parts, start, k = [], train, 0
while start + test <= len(pidx) and k < len(picks):
    te = pidx[start:start + test]
    s = make_signal(picks[k])
    pos = s.reindex(ret.index).clip(-1, 1).shift(1).fillna(0.0).reindex(te)
    pos_parts.append(pos)
    start += test; k += 1
posall = pd.concat(pos_parts).reindex(oos.index)

pct_long = float((posall < 0).mean() * 100)          # long vol (negative s) -- none by design
pct_short = float((posall > 0).mean() * 100)
pct_flat = float((posall == 0).mean() * 100)

# ----------------------------------------------------------------------- report
m = H.metrics(oos)
mb = H.metrics(static_oos)
print("\n=== OOS (walk-forward, trailing 1260d train / 252d test) ===")
print({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})
print("=== static s=+1 on SAME OOS dates ===")
print({k: round(v, 3) if isinstance(v, float) else v for k, v in mb.items()})
print(f"\nOOS span: {oos.index.min().date()}..{oos.index.max().date()}  n={len(oos)}")
print(f"%days long-vol {pct_long:.1f}  short-vol {pct_short:.1f}  flat {pct_flat:.1f}")

print("\npicks per block:")
cnt = {}
for g in picks:
    key = f"{g['events']}/{g['window']}/b{g['baseline']}"
    cnt[key] = cnt.get(key, 0) + 1
for kk, v in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"  {kk}: {v}")

# timing-alpha regression of OOS pnl on the static stream (same dates)
y = oos.values; x = static_oos.reindex(oos.index).fillna(0.0).values
b = np.cov(y, x)[0, 1] / np.var(x)
a = y.mean() - b * x.mean()
resid = y - (a + b * x)
t_a = a / (resid.std(ddof=2) / np.sqrt(len(y)))
print(f"\ntiming regression  oos = a + b*static:  beta={b:.3f}  alpha={a:.3e}/day  t(alpha)={t_a:.2f}")

# ------------------------------------------------------------------ self-audit
print("\n=== LOOK-AHEAD / TAIL SELF-AUDIT ===")
print("1. signal is calendar-only (deterministic NFP rule + published FOMC schedule);")
print("   zero market data enters the signal -> no forecast/threshold look-ahead possible.")
print("2. harness shifts s; signal additionally placed at d-1 for day-d positions,")
print("   so every position day uses only ex-ante calendar information.")
print("3. no caps/clips/winsorization; worst 5 OOS days (uncapped):")
worst = oos.nsmallest(5)
for d, v in worst.items():
    print(f"     {d.date()}  {v:+.6f}  (pos={posall.loc[d]:+.2f})")
print("4. unscheduled FOMC dates kept (no hindsight pruning); their OOS event-day P&L:")
unsched_days = [idx[p] for p in sorted(UNSCHED_POS)]
hit = [d for d in unsched_days if d in oos.index and posall.get(d, 0) != 0]
contrib = float(oos.reindex(hit).sum()) if hit else 0.0
print(f"     exposed on {len(hit)} of them; total contribution {contrib:+.6f} "
      f"({'LOSS -> conservative bias' if contrib < 0 else 'gain -> would need scrutiny'})")
print("5. params picked only via H.walk_forward trailing-train Sharpe; OOS series reported.")
print("6. annualization: harness metrics() uses sqrt(252) on full calendar-time daily series")
print("   (flat days kept as zeros) -- no sparse-day annualization mirage.")
