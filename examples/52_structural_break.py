# -*- coding: utf-8 -*-
"""Example 52 - is the floor book's edge a STRUCTURAL property or 'one good crisis'?
Quandt-Andrews sup-Wald break test on the book P&L (bootstrap p-value), plus per-year-bucket
Sharpe to show whether the edge concentrates. Ported from the semiconductor study's break battery."""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
import bookopt_harness as H, bookopt_floor as F, bookopt_stats as S

book, sleeves, W = F.build()
allidx = pd.DatetimeIndex(sorted(set().union(*[set(sleeves[k].index) for k in sleeves])))
cal = book.reindex(allidx).fillna(0.0)

print(f"STRUCTURAL-BREAK TEST  floor book {book.index.min().date()}..{book.index.max().date()}\n")
for tag, r in [("active", book), ("calendar", cal)]:
    qa = S.quandt_andrews(r.values, index=r.index, n_boot=400, seed=3)
    print(f"({tag}) sup-Wald F={qa['sup_F']:.1f} at {qa['break_date']}  bootstrap p={qa['p_value']:.3f} "
          f"({'BREAK' if qa['p_value']<0.05 else 'no significant break'})")
    print(f"        pre-break Sharpe {qa['pre_sharpe']:+.2f} (n={qa['n_pre']})  |  "
          f"post-break {qa['post_sharpe']:+.2f} (n={qa['n_post']})")

# concentration check: is any single regime/crisis carrying the edge? --------------------
print("\nSharpe by 3-year bucket (is the edge persistent or one-window?):")
SQ = S.SQ
for lo in range(2005, 2027, 3):
    g = book[(book.index.year >= lo) & (book.index.year < lo+3)]
    if len(g) < 60: continue
    sh = g.mean()/g.std()*SQ if g.std() > 0 else float("nan")
    bar = "#" * max(0, int(sh*5))
    print(f"  {lo}-{lo+2}: Sharpe {sh:+.2f}  {bar}")
print("\nREAD: a significant break with one side ~0 = 'one regime' fragility (the vol analogue of")
print("'one AI boom'). Persistent positive buckets = structural. Both halves significant = robust.")
