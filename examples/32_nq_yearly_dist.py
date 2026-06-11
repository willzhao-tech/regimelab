"""NQ per-year DISTRIBUTION profiles: daily price change + range volatility (H-L)/O."""
import os, numpy as np, pandas as pd
D = r"C:\Users\ASUS\Desktop\claude doc\1"

df = pd.read_csv(os.path.join(D, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
ret = (c.pct_change() * 100).dropna()          # daily price change, %
rv = ((h - l) / o * 100).reindex(ret.index)    # daily range vol, %

print("=" * 112)
print("DAILY PRICE CHANGE (%) — distribution by year")
print("=" * 112)
print(f"{'year':<6}{'mean':>7}{'std':>6}{'skew':>6}{'kurt':>6}{'min':>7}{'p5':>7}{'p25':>7}"
      f"{'p50':>7}{'p75':>7}{'p95':>7}{'max':>7}")
for y, r in ret.groupby(ret.index.year):
    q = r.quantile([.05, .25, .50, .75, .95])
    print(f"{y:<6}{r.mean():>+7.2f}{r.std():>6.2f}{r.skew():>6.1f}{r.kurt():>6.1f}{r.min():>+7.1f}"
          f"{q[.05]:>+7.2f}{q[.25]:>+7.2f}{q[.50]:>+7.2f}{q[.75]:>+7.2f}{q[.95]:>+7.2f}{r.max():>+7.1f}")

print("\n" + "=" * 112)
print("RANGE VOLATILITY (High-Low)/Open (%) — distribution by year")
print("=" * 112)
print(f"{'year':<6}{'mean':>7}{'med':>7}{'std':>6}{'skew':>6}{'p25':>7}{'p75':>7}{'p90':>7}"
      f"{'p95':>7}{'max':>7}{'AC1':>6}{'annVol':>8}")
for y, v in rv.groupby(rv.index.year):
    q = v.quantile([.25, .75, .90, .95])
    # Parkinson annualized vol for the year from ranges
    hy = np.log(h / l).loc[v.index]
    park = np.sqrt((hy ** 2).mean() / (4 * np.log(2))) * np.sqrt(252) * 100
    print(f"{y:<6}{v.mean():>7.2f}{v.median():>7.2f}{v.std():>6.2f}{v.skew():>6.1f}{q[.25]:>7.2f}"
          f"{q[.75]:>7.2f}{q[.90]:>7.2f}{q[.95]:>7.2f}{v.max():>7.1f}{v.autocorr(1):>6.2f}{park:>7.0f}%")
