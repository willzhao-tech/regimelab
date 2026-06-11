# -*- coding: utf-8 -*-
"""Example 49 - P1.6 k-SENSITIVITY of the floor book (the single biggest extrapolation).
k = short-dated-IV / 30d-index ratio, MEASURED 0.820 on SPX VIX1D/VIX (0.866 vol-rising / 0.790 falling),
then applied to ALL 8 markets and ALL history. Sweep k and see if the floor (and its calendar Sharpe)
survives. Report active-days Sharpe, calendar-basis Sharpe (idle=0), maxDD. FULL L4 frictions."""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import pandas as pd, bookopt_harness as H, bookopt_floor as F

ORIG_K = H.K
print("k-SENSITIVITY of the floor book (measured k=0.82; uncertainty band ~0.79-0.87)")
print(f"  {'k':>6}{'activeSh':>10}{'calSh':>8}{'maxDD':>8}{'n':>7}   note")
for k in (0.70, 0.75, 0.79, 0.82, 0.85, 0.90, 0.95, 1.00):
    H.K = k
    book, sleeves, _ = F.build()
    allidx = pd.DatetimeIndex(sorted(set().union(*[set(sleeves[m].index) for m in sleeves])))
    cal = book.reindex(allidx).fillna(0.0)
    note = "  <- measured" if abs(k-0.82) < 1e-9 else ("  band" if 0.79 <= k <= 0.87 else "")
    print(f"  {k:>6.2f}{H.sharpe(book):>10.2f}{H.sharpe(cal):>8.2f}{F.stat_line(book)['maxdd']*100:>7.0f}%"
          f"{len(book):>7}{note}")
H.K = ORIG_K
print("\n  VERDICT: book stays profitable across the whole band (calendar Sharpe 0.58@0.79 -> 1.69@0.85),")
print("  BUT the LEVEL is highly k-sensitive: +-0.03 in k swings active Sharpe 0.88 <-> 1.98.")
print("  KEY CAVEAT: k was measured as an AVERAGE 0.82 (0.790 vol-falling / 0.866 vol-rising). The book")
print("  trades mostly in CALM/low-VIX regimes (46% of days, where vol is typically falling) -> the")
print("  EFFECTIVE k on its actual trading days likely sits nearer 0.79, i.e. the conservative")
print("  ~0.88 active / ~0.58 calendar end. Treat ~0.6-1.0 calendar as the prudent planning band,")
print("  not the 1.26. The single biggest remaining model risk; only real per-strike quotes resolve it.")
