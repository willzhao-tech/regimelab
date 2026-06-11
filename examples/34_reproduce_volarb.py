"""
Example 34 — REPRODUCIBILITY PACKAGE for the two Sharpe>0.8 positive-alpha survivors.

Strategies (the only ones in the whole study passing Sharpe>0.8 with timing alpha):
  A  regime_combo   : long-short variance gated on (richness, vol-trend)
  B  range_forecast : long-short variance gated on (today's range vs implied breakeven)
  A+B 50/50 blend.
Baseline: static short-variance (s=+1) on identical dates.

Outputs (all to C:\\Users\\ASUS\\Desktop\\claude doc\\1):
  volarb_equity.png        equity curves (cumulative P&L per unit variance notional) + DD + rolling Sharpe
  volarb_ledger_A.csv / _B.csv / _blend.csv   daily ledgers (every input, signal, position, pnl)
  volarb_trades_A.csv / _B.csv                position-change logs
  VOLARB_SPEC.md           complete formulas/data spec for independent recreation
"""
import os, sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)
COST_VOLPT = 0.5
TRAIN, TEST = 1260, 252

# ---------- data ----------
df = pd.read_csv(os.path.join(OUT, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
vxn = pd.read_csv(os.path.join(OUT, "VXN_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
ret = df["Close"].pct_change()
idx = ret.index.intersection(vxn.index)
df, ret, vxn = df.loc[idx], ret.loc[idx], vxn.loc[idx]

# ---------- shared causal building blocks ----------
def park(win):  # trailing Parkinson vol forecast, %; value at t uses data through t-1
    return (np.sqrt((np.log(df["High"]/df["Low"])**2).rolling(win).mean()/(4*np.log(2)))*SQ*100).shift(1)

fc21, fc10, fc42 = park(21), park(10), park(42)
richness = vxn - fc21                       # vol pts
trend = fc10 - fc42                         # vol pts (>0 vol rising)
rng = np.log(df["High"]/df["Low"]) * 100.0  # today's range, %
be = vxn / SQ                               # implied daily breakeven, %
iv = (vxn.shift(1)/100.0)**2/252.0          # variance strike accruing day t (set at t-1)
rvar = ret**2

def sig_A(g):
    r_hi, r_lo, d = g
    s = pd.Series(0.0, index=idx)
    s[((richness >= r_hi) & (trend <= -d)).fillna(False)] = 1.0
    s[((richness <= r_lo) & (trend >= d)).fillna(False)] = -1.0
    return s

def sig_B(g):
    b1, b2 = g
    s = pd.Series(0.0, index=idx)
    ok = rng.notna() & be.notna()
    s[ok & (rng < b1*be)] = 1.0
    s[ok & (rng > b2*be)] = -1.0
    return s

GRID_A = [(a, b, c) for a in (2.0, 4.0, 6.0) for b in (0.0, -2.0) for c in (0.0, 1.0, 2.0)]
GRID_B = [(b1, b2) for b1 in (0.8, 1.0, 1.2) for b2 in (1.3, 1.6, 2.0)]

def pnl_of(s):
    pos = s.clip(-1, 1).shift(1).fillna(0.0)
    cost = (2*vxn.shift(1)*COST_VOLPT/1e4/252).fillna(0.0)*pos.diff().abs().fillna(0.0)
    return (pos*(iv - rvar) - cost).dropna(), pos

def walk_forward(sig_fn, grid):
    series = {g: pnl_of(sig_fn(g))[0] for g in grid}
    base_idx = list(series.values())[0].index
    parts, pos_parts, picks = [], [], []
    start = TRAIN
    while start + TEST <= len(base_idx):
        tr = base_idx[start-TRAIN:start]; te = base_idx[start:start+TEST]
        def sh(p):
            x = p.reindex(tr).dropna()
            return x.mean()/x.std()*SQ if len(x) > 60 and x.std() > 0 else -9.0
        best = max(grid, key=lambda g: sh(series[g]))
        picks.append((str(te[0].date()), str(te[-1].date()), best))
        parts.append(series[best].reindex(te))
        pos_parts.append(pnl_of(sig_fn(best))[1].reindex(te))
        start += TEST
    return pd.concat(parts).dropna(), pd.concat(pos_parts), picks

def metrics(p):
    p = p.dropna()
    sh = p.mean()/p.std()*SQ; t = p.mean()/(p.std()/np.sqrt(len(p)))
    cum = p.cumsum(); mdd = float((cum-cum.cummax()).min())
    return dict(sharpe=float(sh), tstat=float(t), skew=float(p.skew()),
                worst_x=float(abs(p.min())/p.mean()), mdd_units=float(mdd/(p.mean()*252)), n=len(p))

# ---------- run ----------
pnl_A, pos_A, picks_A = walk_forward(sig_A, GRID_A)
pnl_B, pos_B, picks_B = walk_forward(sig_B, GRID_B)
common = pnl_A.index.intersection(pnl_B.index)
pnl_A, pnl_B = pnl_A.loc[common], pnl_B.loc[common]
blend = 0.5*pnl_A + 0.5*pnl_B
static, _ = pnl_of(pd.Series(1.0, index=idx))
static = static.reindex(common).dropna()

print(f"OOS period {common.min().date()}..{common.max().date()}  ({len(common)} days)")
print(f"{'strategy':<18}{'Sharpe':>8}{'t':>7}{'skew':>7}{'worst/mean':>11}{'maxDD(mean-yrs)':>16}")
res = {}
for name, p in [("A regime_combo", pnl_A), ("B range_forecast", pnl_B), ("A+B blend", blend), ("static short-vol", static)]:
    m = metrics(p); res[name] = m
    print(f"{name:<18}{m['sharpe']:>8.2f}{m['tstat']:>7.1f}{m['skew']:>7.1f}{m['worst_x']:>11.0f}{m['mdd_units']:>16.1f}")

# alpha vs static
for name, p in [("A", pnl_A), ("B", pnl_B), ("blend", blend)]:
    yx = pd.concat([p, static], axis=1).dropna().values
    y, x = yx[:, 0], yx[:, 1]
    X = np.column_stack([np.ones(len(x)), x])
    b_, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X@b_; ta = b_[0]/np.sqrt((r@r/(len(y)-2))*np.linalg.inv(X.T@X)[0, 0])
    print(f"  {name}: beta {b_[1]:+.2f}  alpha t (OLS) {ta:+.1f}")

# ---------- dynamic benchmark: yearly ----------
print("\nYEARLY Sharpe (dynamic benchmark)  [strategy | static on same year]")
yr = pd.DataFrame({"A": pnl_A, "B": pnl_B, "blend": blend, "static": static})
for y, g in yr.groupby(yr.index.year):
    line = f"  {y}: "
    for cname in ["A", "B", "blend", "static"]:
        s = g[cname].dropna()
        line += f"{cname} {s.mean()/s.std()*SQ:+5.1f}  " if len(s) > 60 and s.std() > 0 else f"{cname}   -   "
    print(line)

# ---------- plot ----------
fig, ax = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)
for name, p, lw in [("A regime_combo", pnl_A, 1.2), ("B range_forecast", pnl_B, 1.2),
                    ("A+B blend", blend, 1.8), ("static short-vol", static, 1.0)]:
    ax[0].plot(p.index, p.cumsum().values*1e4, label=name, lw=lw)
ax[0].set_ylabel("cum P&L (1e-4 per unit var notional)"); ax[0].legend(loc="upper left")
ax[0].set_title("Long-short vol-arb (walk-forward OOS) vs static short-vol — cumulative P&L")
cb = blend.cumsum(); ax[1].fill_between(cb.index, (cb-cb.cummax()).values*1e4, 0, color="firebrick", alpha=.5)
ax[1].set_ylabel("blend DD")
rs = blend.rolling(252).mean()/blend.rolling(252).std()*SQ
ax[2].plot(rs.index, rs.values, color="navy"); ax[2].axhline(0, color="grey", lw=.5)
ax[2].set_ylabel("blend rolling 1y Sharpe")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "volarb_equity.png"), dpi=110); plt.close(fig)

# ---------- ledgers & trade logs ----------
def ledger(pnl, pos, fname):
    led = pd.DataFrame({
        "NQ_close": df["Close"].reindex(pnl.index), "VXN": vxn.reindex(pnl.index),
        "park21_fcast": fc21.reindex(pnl.index), "richness": richness.reindex(pnl.index),
        "vol_trend": trend.reindex(pnl.index), "range_pct": rng.reindex(pnl.index),
        "breakeven_pct": be.reindex(pnl.index), "iv_strike_daily_var": iv.reindex(pnl.index),
        "realized_var": rvar.reindex(pnl.index), "position": pos.reindex(pnl.index),
        "daily_pnl": pnl, "cum_pnl": pnl.cumsum()}).round(8)
    led.to_csv(os.path.join(OUT, fname), index_label="Date")
    return led

led_A = ledger(pnl_A, pos_A.reindex(pnl_A.index), "volarb_ledger_A.csv")
led_B = ledger(pnl_B, pos_B.reindex(pnl_B.index), "volarb_ledger_B.csv")
ledger(blend, (0.5*pos_A.reindex(common).fillna(0)+0.5*pos_B.reindex(common).fillna(0)), "volarb_ledger_blend.csv")

def tradelog(pos, fname):
    ch = pos[pos.diff().fillna(pos).abs() > 1e-9]
    tl = pd.DataFrame({"new_position": ch,
                       "VXN": vxn.reindex(ch.index), "richness": richness.reindex(ch.index),
                       "vol_trend": trend.reindex(ch.index), "range_pct": rng.reindex(ch.index),
                       "breakeven_pct": be.reindex(ch.index)}).round(4)
    tl.to_csv(os.path.join(OUT, fname), index_label="Date")
    return tl

tl_A = tradelog(pos_A.reindex(pnl_A.index), "volarb_trades_A.csv")
tl_B = tradelog(pos_B.reindex(pnl_B.index), "volarb_trades_B.csv")
print(f"\ntrade log A: {len(tl_A)} position changes;  B: {len(tl_B)}")
print("sample trades (A):"); print(tl_A.head(5).to_string())

# ---------- walk-forward picks (needed for recreation) ----------
print("\nwalk-forward picks A (block, params r_hi/r_lo/d):")
for a, b, g in picks_A: print(f"  {a}..{b}: {g}")
print("walk-forward picks B (block, params b1/b2):")
for a, b, g in picks_B: print(f"  {a}..{b}: {g}")

with open(os.path.join(OUT, "VOLARB_SPEC.md"), "w", encoding="utf-8") as f:
    f.write(f"""# Long-short vol-arb on NQ/VXN — independent recreation spec

## Data
- NQ daily OHLCV: Investing.com financialdata API pairId 8874 (or any NQ continuous future source). 1999-06-22..present.
- VXN daily close: Investing.com pairId 44369 (CBOE Nasdaq-100 vol index; also Yahoo ^VXN). 2001-02-05..present.
- Align on common dates. ret_t = Close_t/Close_(t-1) - 1.

## Instrument (PROXY — the key caveat)
Daily variance-swap proxy: P&L_t per unit notional = pos_t * (iv_t - rvar_t) - cost_t
  iv_t = (VXN_(t-1)/100)^2 / 252   (strike set at prior close)
  rvar_t = ret_t^2
  cost_t = 2*VXN_(t-1)*{COST_VOLPT}/10^4/252 * |pos_t - pos_(t-1)|   ({COST_VOLPT} vol-pt spread)
This 1-day-variance-at-VXN-strike is NOT directly tradable; it inflates Sharpe LEVELS for
strategy and baseline alike. Expect a real NDX var-swap/option implementation ~0.8-1.2 Sharpe.
The relative results (alpha vs static, skew inversion) are the robust claims.

## Causal signals (s_t uses data through close t; position pos_(t+1) = s_t)
Forecast: park(w)_t = sqrt( mean_(i=t-w..t-1) ln(High_i/Low_i)^2 / (4 ln 2) ) * sqrt(252) * 100

A) regime_combo: richness_t = VXN_t - park(21)_t ; volTrend_t = park(10)_t - park(42)_t
   s=+1 (short vol) if richness >= r_hi AND volTrend <= -d
   s=-1 (long vol)  if richness <= r_lo AND volTrend >= +d ; else 0
   grid: r_hi in (2,4,6), r_lo in (0,-2), d in (0,1,2)

B) range_forecast: range_t = ln(High_t/Low_t)*100 ; breakeven_t = VXN_t/sqrt(252)
   s=+1 if range < b1*breakeven ; s=-1 if range > b2*breakeven ; else 0
   grid: b1 in (0.8,1.0,1.2), b2 in (1.3,1.6,2.0)

## Walk-forward (NO in-sample param choice)
Train=1260 trading days, Test=252, rolling. In each block pick the grid params with the
highest TRAIN Sharpe; apply to the next 252-day TEST block; concatenate TEST blocks only.
Blend = 0.5*A + 0.5*B daily P&L.

## Expected results (verify against volarb_ledger_*.csv)
A: Sharpe ~{res['A regime_combo']['sharpe']:.2f}, skew ~{res['A regime_combo']['skew']:.0f} | B: ~{res['B range_forecast']['sharpe']:.2f} | blend ~{res['A+B blend']['sharpe']:.2f}
static short-vol same dates: ~{res['static short-vol']['sharpe']:.2f}, skew ~{res['static short-vol']['skew']:.0f}
OOS {common.min().date()}..{common.max().date()}. Daily ledgers + trade logs in volarb_*.csv.
""")
print(f"\nfiles -> volarb_equity.png, volarb_ledger_[A|B|blend].csv, volarb_trades_[A|B].csv, VOLARB_SPEC.md")
