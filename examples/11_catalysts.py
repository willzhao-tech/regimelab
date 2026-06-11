"""
Example 11 — catalyst analysis of NQ (first cut: what's computable from OHLCV alone).

No prior catalyst analysis existed in the project (the 'calendar' html is a chart and the
'nq' folders are raw-price screenshots), so this builds real findings from the NQ OHLCV:

  1. OVERNIGHT vs INTRADAY decomposition — where does NQ's return actually accrue?
     overnight = close[t-1]->open[t];  intraday = open[t]->close[t].
  2. DAY-OF-WEEK effect.
  3. TURN-OF-MONTH effect (last trading day + first 3).
  4. MONTH-OF-YEAR seasonality.
  5. OPEX / triple-witching week (week of the 3rd Friday).

Each with an effect size and a t-stat. These are SCHEDULED/structural catalysts; macro-event
catalysts (FOMC, CPI, NFP) need event dates and are the flagged next step. Caveat throughout:
many calendar patterns are in-sample, small, and partly arbitraged — t-stats are NOT deflated
for the number of patterns tested, so treat them as descriptive, not tradable claims.

Run:  python examples/11_catalysts.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"


def tstat(x):
    x = x.dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else float("nan")


def ann(mean_daily):
    return mean_daily * 252 * 100


def sharpe(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else float("nan")


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"),
                     parse_dates=["Date"]).set_index("Date").sort_index()
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    overnight = (o / c.shift(1) - 1).dropna()
    intraday = (c / o - 1).dropna()
    total = (c.pct_change()).dropna()
    idx = total.index
    print(f"NQ {idx.min().date()}..{idx.max().date()}  ({len(total)} days)\n")

    # 1) overnight vs intraday
    print("=" * 78)
    print("1) OVERNIGHT vs INTRADAY — where the return accrues")
    print("=" * 78)
    print(f"{'segment':<14}{'ann return':>12}{'Sharpe':>9}{'t-stat':>9}{'$1 ->':>10}")
    for name, s in [("overnight", overnight), ("intraday", intraday), ("total(24h)", total)]:
        eq = float((1 + s).prod())
        print(f"{name:<14}{ann(s.mean()):>11.1f}%{sharpe(s):>9.2f}{tstat(s):>9.1f}{eq:>9.1f}x")
    print("  (classic 'overnight drift': index gains accrue close->open; intraday adds little/negative)")

    # 2) day-of-week (on total daily return)
    print("\n" + "=" * 78)
    print("2) DAY-OF-WEEK (total daily return)")
    print("=" * 78)
    print(f"{'day':<6}{'ann return':>12}{'mean bps':>10}{'t-stat':>9}{'n':>7}")
    for i, day in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
        seg = total[total.index.dayofweek == i]
        print(f"{day:<6}{ann(seg.mean()):>11.1f}%{seg.mean()*1e4:>9.1f}{tstat(seg):>9.1f}{len(seg):>7}")

    # 3) turn-of-month: last trading day of month + first 3 of next
    tom = pd.Series(False, index=idx)
    by_month = pd.Series(idx, index=idx).groupby([idx.year, idx.month])
    first3 = pd.Index([])
    for _, days in by_month:
        first3 = first3.union(days.iloc[:3])
    last1 = pd.Index([])
    for _, days in by_month:
        last1 = last1.union(days.iloc[-1:])
    tom_days = first3.union(last1)
    tom_mask = idx.isin(tom_days)
    print("\n" + "=" * 78)
    print("3) TURN-OF-MONTH (last trading day + first 3) vs rest of month")
    print("=" * 78)
    print(f"{'window':<16}{'ann return':>12}{'mean bps':>10}{'t-stat':>9}{'n':>7}")
    for name, m in [("turn-of-month", tom_mask), ("rest", ~tom_mask)]:
        seg = total[m]
        print(f"{name:<16}{ann(seg.mean()):>11.1f}%{seg.mean()*1e4:>9.1f}{tstat(seg):>9.1f}{len(seg):>7}")

    # 4) month-of-year seasonality
    print("\n" + "=" * 78)
    print("4) MONTH-OF-YEAR (mean daily return within each calendar month)")
    print("=" * 78)
    mser = total.groupby(total.index.month).mean() * 1e4
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    print("  " + "  ".join(f"{months[i-1]}:{mser.get(i,np.nan):+.1f}" for i in range(1, 13)) + "  (mean bps/day)")

    # 5) OPEX week (week containing the 3rd Friday) and triple-witching (Mar/Jun/Sep/Dec)
    def third_friday(y, m):
        d = pd.Timestamp(y, m, 1)
        fri = d + pd.Timedelta(days=(4 - d.dayofweek) % 7)
        return fri + pd.Timedelta(days=14)
    opex_weeks = set()
    tw_weeks = set()
    for y in range(idx.min().year, idx.max().year + 1):
        for m in range(1, 13):
            tf = third_friday(y, m)
            wk = (tf.isocalendar().year, tf.isocalendar().week)
            opex_weeks.add(wk)
            if m in (3, 6, 9, 12):
                tw_weeks.add(wk)
    isoweek = pd.Series([(d.isocalendar().year, d.isocalendar().week) for d in idx], index=idx)
    opex_mask = isoweek.isin(opex_weeks).values
    tw_mask = isoweek.isin(tw_weeks).values
    print("\n" + "=" * 78)
    print("5) OPEX / TRIPLE-WITCHING WEEK (week of 3rd Friday)")
    print("=" * 78)
    print(f"{'window':<18}{'ann return':>12}{'mean bps':>10}{'t-stat':>9}{'n':>7}")
    for name, m in [("opex week", opex_mask), ("triple-witch wk", tw_mask), ("other weeks", ~opex_mask)]:
        seg = total[m]
        print(f"{name:<18}{ann(seg.mean()):>11.1f}%{seg.mean()*1e4:>9.1f}{tstat(seg):>9.1f}{len(seg):>7}")

    print("\nNOTE: descriptive, in-sample, NOT deflated for multiple patterns tested. Macro-event")
    print("catalysts (FOMC pre-drift, CPI/NFP day) need event dates — the flagged next step.")


if __name__ == "__main__":
    main()
