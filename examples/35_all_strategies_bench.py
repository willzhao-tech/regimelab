"""
Example 35 — UNIFIED BENCHMARK: every strategy of the study, one window, one risk footing.

All strategies evaluated on the common OOS window (vol-arb walk-forward period 2006-2026).
Each is constant-levered to realize exactly 10% annualized vol over the window (display
normalization: Sharpe, t-stat, skew and curve SHAPE are unaffected; only the y-scale is).
Vol-arb daily P&L is read from the saved walk-forward ledgers (volarb_ledger_*.csv).

Outputs: all_strategies_equity.png + printed benchmark table.
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
COST = 0.0002

df = pd.read_csv(os.path.join(OUT, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
vxn = pd.read_csv(os.path.join(OUT, "VXN_all_history.csv"), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
ret = c.pct_change()

# ---------- directional strategies (ex 27/28 conventions, causal) ----------
def rsi(s, n):
    d = s.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    return 100 - 100/(1 + up.ewm(alpha=1/n, adjust=False).mean()/dn.ewm(alpha=1/n, adjust=False).mean())
def state(entry, exit_):
    pos = np.zeros(len(c)); s = 0; e1 = entry.values; e2 = exit_.values
    for i in range(len(c)):
        if s == 0 and e1[i]: s = 1
        elif s == 1 and e2[i]: s = 0
        pos[i] = s
    return pd.Series(pos, index=c.index)
sma = lambda n: c.rolling(n).mean()
park21v = np.sqrt((np.log(h/l)**2).rolling(21).mean()/(4*np.log(2)))*SQ
r2 = rsi(c, 2)

def ev(sig):
    pos = sig.shift(1).fillna(0.0)
    return (pos*ret - COST*pos.diff().abs().fillna(0.0)).dropna()

def sleeve_a():
    fomc = pd.read_csv(os.path.join(OUT, "FOMC_dates.csv"), parse_dates=["fomc_date"])["fomc_date"].sort_values()
    kept, last = [], None
    for d_ in fomc:
        if last is None or (d_-last).days >= 20: kept.append(d_); last = d_
    fomc = pd.DatetimeIndex(kept)
    sig = (c > sma(200)).astype(float)
    desired = ((0.15/park21v).clip(upper=3.0)*sig).shift(1).fillna(0.0)
    base, cur, pm = [], 0.0, None
    for d_, v in desired.items():
        if pm is None or d_.month != pm.month: cur = float(v)
        base.append(cur); pm = d_
    base = pd.Series(base, index=desired.index)
    win = pd.Series(False, index=ret.index)
    for p in ret.index.get_indexer(fomc, method="bfill"):
        if 1 <= p < len(ret.index): win.iloc[p-1] = True; win.iloc[p] = True
    pos = base.where(~win, np.minimum(base+0.5, 4.0))
    return (pos*ret - 0.0005*pos.diff().abs().fillna(0.0)).dropna()

S = {
  "buy&hold NQ":        ev(pd.Series(1.0, index=c.index)),
  "vol_target 15%":     ev((0.15/park21v).clip(upper=3.0)),
  "Sleeve A (vt+tr+FOMC)": sleeve_a(),
  "sma200 filter":      ev((c > sma(200)).astype(float)),
  "golden cross 50/200":ev((sma(50) > sma(200)).astype(float)),
  "tsmom 12m":          ev((c > c.shift(252)).astype(float)),
  "RSI2 Connors":       ev(state((r2 < 10) & (c > sma(200)), c > sma(5))),
  "buy-the-dip 3d":     ev(state((ret < 0) & (ret.shift(1) < 0) & (ret.shift(2) < 0), ret > 0)),
}
# ---------- vol space ----------
iv = (vxn.shift(1)/100.0)**2/252.0
idx2 = ret.index.intersection(vxn.index)
S["static short-vol"] = ((iv - ret**2)).reindex(idx2).dropna()
for tag, f in [("vol-arb A (regime)", "volarb_ledger_A.csv"),
               ("vol-arb B (range)", "volarb_ledger_B.csv"),
               ("vol-arb blend", "volarb_ledger_blend.csv")]:
    S[tag] = pd.read_csv(os.path.join(OUT, f), parse_dates=["Date"]).set_index("Date")["daily_pnl"].dropna()

# ---------- common window + normalization ----------
window = S["vol-arb blend"].index
print(f"common OOS window {window.min().date()}..{window.max().date()}  ({len(window)} days)")
rows = []; curves = {}
for name, p in S.items():
    p = p.reindex(window).dropna()
    if len(p) < 1000: continue
    k = 0.10/(p.std()*SQ)                      # constant display leverage to 10% vol
    r10 = (k*p)
    eq = (1+r10).cumprod()
    yrs = (p.index[-1]-p.index[0]).days/365.25
    rows.append(dict(strategy=name, sharpe=p.mean()/p.std()*SQ, skew=p.skew(),
                     cagr10=eq.iloc[-1]**(1/yrs)-1, maxdd10=float((eq/eq.cummax()-1).min()),
                     worst10=float(r10.min()), term10=float(eq.iloc[-1])))
    curves[name] = eq

t = pd.DataFrame(rows).set_index("strategy").sort_values("sharpe", ascending=False)
print(f"\nALL STRATEGIES @ equal 10% vol footing  (Sharpe/skew native; CAGR/DD at 10% vol)")
print(f"{'strategy':<24}{'Sharpe':>8}{'skew':>7}{'CAGR@10%':>10}{'maxDD@10%':>11}{'worstday':>10}{'$1->':>8}")
for n_, r in t.iterrows():
    print(f"{n_:<24}{r['sharpe']:>8.2f}{r['skew']:>7.1f}{r['cagr10']*100:>9.1f}%{r['maxdd10']*100:>10.0f}%"
          f"{r['worst10']*100:>9.1f}%{r['term10']:>7.1f}x")

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(13, 8))
emph = {"vol-arb blend": ("navy", 2.2), "vol-arb A (regime)": ("royalblue", 1.2),
        "vol-arb B (range)": ("deepskyblue", 1.0), "static short-vol": ("firebrick", 1.4),
        "buy&hold NQ": ("black", 1.6), "vol_target 15%": ("darkorange", 1.4),
        "Sleeve A (vt+tr+FOMC)": ("green", 1.4)}
for name, eq in curves.items():
    col, lw = emph.get(name, ("grey", 0.7))
    al = 1.0 if name in emph else 0.55
    ax.plot(eq.index, eq.values, label=f"{name}  (Sh {t.loc[name,'sharpe']:.2f})", color=col, lw=lw, alpha=al)
ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)  —  every strategy at identical 10% vol")
ax.set_title("ALL strategies of the study — common window 2006-2026, equal-risk footing\n"
             "(constant per-strategy leverage to 10% realized vol: display only, Sharpe/shape unaffected; "
             "vol-arb & short-vol are variance-swap PROXY P&L)")
ax.legend(loc="upper left", fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "all_strategies_equity.png"), dpi=110); plt.close(fig)
print(f"\nchart -> {os.path.join(OUT, 'all_strategies_equity.png')}")
