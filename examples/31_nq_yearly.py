"""NQ statistical profile, year by year (1999-2026)."""
import os, numpy as np, pandas as pd
D = r"C:\Users\ASUS\Desktop\claude doc\1"
SQ = np.sqrt(252.0)

df = pd.read_csv(os.path.join(D, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
ret = c.pct_change().dropna()
rv = ((h - l) / o).reindex(ret.index)

rows = []
for y, r in ret.groupby(ret.index.year):
    e = (1 + r).cumprod()
    mdd = float((e / e.cummax() - 1).min())
    rng = rv.loc[r.index]
    rows.append(dict(year=y, n=len(r), ret=float(e.iloc[-1] - 1), vol=float(r.std() * SQ),
                     sharpe=float(r.mean() / r.std() * SQ) if r.std() > 0 else np.nan,
                     maxdd=mdd, skew=float(r.skew()), kurt=float(r.kurt()),
                     worst=float(r.min()), best=float(r.max()),
                     pct_up=float((r > 0).mean()), rng_med=float(rng.median()),
                     big3=int((r.abs() > 0.03).sum())))
t = pd.DataFrame(rows).set_index("year")

print(f"NQ year-by-year profile  ({ret.index.min().date()} .. {ret.index.max().date()})")
print(f"{'year':<6}{'ret':>8}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}{'skew':>7}{'kurt':>6}"
      f"{'worst':>8}{'best':>8}{'%up':>6}{'medRng':>8}{'|>3%|':>6}")
for y, r in t.iterrows():
    print(f"{y:<6}{r['ret']*100:>+7.0f}%{r['vol']*100:>6.0f}%{r['sharpe']:>8.2f}{r['maxdd']*100:>7.0f}%"
          f"{r['skew']:>7.1f}{r['kurt']:>6.1f}{r['worst']*100:>+7.1f}%{r['best']*100:>+7.1f}%"
          f"{r['pct_up']*100:>5.0f}%{r['rng_med']*100:>7.2f}%{r['big3']:>6}")

print("\nsummary across the 28 calendar years:")
full = t.iloc[:-1] if t.iloc[-1]["n"] < 200 else t   # exclude partial current year from averages
print(f"  positive years: {(t['ret']>0).sum()}/{len(t)}   median yearly ret {t['ret'].median()*100:+.0f}%   "
      f"mean {t['ret'].mean()*100:+.0f}%")
print(f"  vol regime range: calmest {t['vol'].min()*100:.0f}% ({t['vol'].idxmin()})  "
      f"wildest {t['vol'].max()*100:.0f}% ({t['vol'].idxmax()})")
print(f"  Sharpe by year: min {t['sharpe'].min():.2f} ({t['sharpe'].idxmin()})  "
      f"max {t['sharpe'].max():.2f} ({t['sharpe'].idxmax()})  |  years Sharpe>1: {(t['sharpe']>1).sum()}, <0: {(t['sharpe']<0).sum()}")
print(f"  vol persistence year-to-year: corr(vol_y, vol_y+1) = {t['vol'].autocorr(1):.2f}   "
      f"ret persistence: corr(ret_y, ret_y+1) = {t['ret'].autocorr(1):.2f}")
