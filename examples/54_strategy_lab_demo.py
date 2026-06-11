# -*- coding: utf-8 -*-
"""Example 54 - STRATEGY LAB DEMO: testing a brand-new signal end-to-end in ~20 lines.
Demo idea ('vol_snapback'): vol-index mean reversion - SHORT vol after a spike above its
trailing mean (sell the panic premium as it decays), LONG vol when the index compresses far
below its mean (own cheap convexity before expansion). A different mechanism from both
production families (A: richness-vs-forecast, B: range-vs-breakeven). DEMO of the intake
path - the equity curve is an audition, not a result; the gauntlet remains the bar."""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import pandas as pd
from strategy_lab import run_lab


def vol_snapback(ctx, p):
    """p = (window, spike_frac, crush_frac): vi vs its trailing-window mean (shifted, causal)."""
    w, spike, crush = p
    vi = ctx["vi"]
    m = vi.rolling(w).mean().shift(1)
    s = pd.Series(0., index=vi.index)
    s[(vi > m * (1 + spike)).fillna(False)] = 1.    # spike -> short the decaying panic premium
    s[(vi < m * (1 - crush)).fillna(False)] = -1.   # compression -> own cheap convexity
    return s


GRID = [(w, sp, cr) for w in (10, 21, 63) for sp in (.15, .30) for cr in (.10, .20)]

if __name__ == "__main__":
    run_lab("vol_snapback", [(vol_snapback, GRID)])
