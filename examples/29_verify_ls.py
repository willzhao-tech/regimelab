"""Independent verification of the long-short workflow's headline claims."""
import os, numpy as np, pandas as pd
from math import erf, sqrt
D = r"C:\Users\ASUS\Desktop\claude doc\1"

# --- 1) verify breakout_ls 1h n=48 'winner' is noise ---
h = pd.read_csv(os.path.join(D,"NQ_1h_6m.csv"), parse_dates=["Datetime"]).set_index("Datetime").sort_index()
c = h["Close"]; ret = c.pct_change(); n = 48
hh = c.rolling(n).max().shift(1); ll = c.rolling(n).min().shift(1)
sig = pd.Series(np.nan, index=c.index); sig[c > hh] = 1; sig[c < ll] = -1; sig = sig.ffill().fillna(0)
lr = (sig.shift(1).fillna(0)*ret - 0.0001*sig.shift(1).fillna(0).diff().abs().fillna(0)).dropna()
half = len(lr)//2; tr, te = lr.iloc[:half], lr.iloc[half:]
ppy = 252*23
shp = lambda x: x.mean()/x.std()*np.sqrt(ppy)
t = te.mean()/(te.std()/np.sqrt(len(te)))
p = 2*(1 - 0.5*(1+erf(abs(t)/sqrt(2))))
print("BREAKOUT n=48 (the workflow's 'best OOS'):")
print(f"  TRAIN Sharpe {shp(tr):+.2f}   TEST Sharpe {shp(te):+.2f}")
print(f"  TEST: {len(te)} bars, per-bar t-stat {t:+.2f}  (two-sided p ~ {p:.2f})  <- the real significance")
print(f"  the '{shp(te):.1f} Sharpe' = per-bar t {t:.2f} x sqrt(ppy/bars); annualization mirage, NOT significant")
print(f"  SIGN-FLIP: train mean/bar {tr.mean()*1e4:+.2f}bps vs test {te.mean()*1e4:+.2f}bps "
      f"({'opposite signs -> regime noise' if tr.mean()*te.mean()<0 else 'same sign'})")

# --- 2) realistic DAILY long-short ruin point on the actual 6m ---
d = pd.read_csv(os.path.join(D,"NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
d = d.loc[d.index >= d.index.max()-pd.Timedelta(days=190)]; dr = d["Close"].pct_change().dropna()
w = dr.min()
print(f"\nDAILY ruin: worst day {w*100:.2f}% on {dr.idxmin().date()} -> a wrong-way position at >= {1/abs(w):.1f}x is WIPED that day")
for L in (10, 15, 20, 21):
    eq = 1.0
    for x in dr:
        eq *= (1 + L*x)
        if eq <= 0: eq = 0.0; break
    print(f"  long {L}x over 6m: {'RUINED' if eq<=0 else f'{(eq-1)*100:+.0f}%'}")
print("NOTE: a -5/-7/-10% NQ day (routine in history) instantly wipes 20x/15x/10x. This window's min was only -4.8%.")
