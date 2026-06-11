# -*- coding: utf-8 -*-
"""Discover + EMPIRICALLY VERIFY Investing.com pairIds for the book inputs not yet in DATASETS.
For each instrument: search candidates, fetch ~90 days for each, and accept ONLY a candidate whose
overlap Closes match the existing CSV (median |rel diff| < 0.2%). Falls back to trying Yahoo for
ETFs. Prints a ready-to-paste verdict per instrument. One-off discovery tool; run via proxy."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from market_data import find_investing_pair, investing, yahoo, DATA_DIR, DEFAULT_PROXY

TARGETS = [
    ("SPX",      "S&P 500",                 None),
    ("EEM",      "EEM",                     "EEM"),     # ETF -> also try Yahoo
    ("DAX",      "DAX",                     None),
    ("SX5E",     "Euro Stoxx 50",           None),
    ("N225",     "Nikkei 225",              None),
    ("HSI",      "Hang Seng",               None),
    ("NSEI",     "Nifty 50",                None),
    ("INDIAVIX", "India VIX",               None),
    ("VDAX",     "VDAX",                    None),
    ("VSTOXX",   "VSTOXX",                  None),
    ("JNIV",     "Nikkei Volatility",       None),
    ("VHSI",     "Hang Seng Volatility",    "^VHSI"),
    ("VXEEM",    "Emerging Markets ETF Volatility", "^VXEEM"),
]
START = "2026-01-01"


def match_score(fetched, existing):
    """median |rel diff| of Close on overlapping dates (NaN if <10 overlaps)."""
    common = fetched.index.intersection(existing.index)
    if len(common) < 10:
        return float("nan"), len(common)
    f = fetched.loc[common, "Close"].astype(float)
    e = existing.loc[common, "Close"].astype(float)
    return float(((f - e).abs() / e.abs()).median()), len(common)


def main():
    for stem, query, ytick in TARGETS:
        path = os.path.join(DATA_DIR, stem + "_all_history.csv")
        if not os.path.exists(path):
            print(f"[{stem}] no existing CSV — skip"); continue
        existing = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        best = None
        try:
            cands = find_investing_pair(query)[:8]
        except Exception as e:
            print(f"[{stem}] search failed: {type(e).__name__}: {e}"); cands = []
        for c in cands:
            pid = c.get("pairId")
            if not pid:
                continue
            try:
                f = investing(int(pid))(START, DEFAULT_PROXY)
            except Exception:
                continue
            if f.empty:
                continue
            score, n = match_score(f, existing)
            if score == score and (best is None or score < best[1]):
                best = (int(pid), score, n, c.get("symbol"), c.get("name"))
            time.sleep(0.8)
        if best and best[1] < 0.002:
            print(f"[{stem}] VERIFIED investing({best[0]})  median|reldiff|={best[1]*100:.3f}% "
                  f"on {best[2]} overlaps  ({best[3]} / {best[4]})")
            continue
        if ytick:                                  # ETF / Yahoo fallback
            try:
                f = yahoo(ytick)(START, DEFAULT_PROXY)
                score, n = match_score(f, existing)
                if score == score and score < 0.002:
                    print(f"[{stem}] VERIFIED yahoo('{ytick}')  median|reldiff|={score*100:.3f}% "
                          f"on {n} overlaps")
                    continue
                print(f"[{stem}] yahoo('{ytick}') mismatch: median|reldiff|="
                      f"{(score*100 if score==score else float('nan')):.3f}% on {n} overlaps")
            except Exception as e:
                print(f"[{stem}] yahoo fallback failed: {type(e).__name__}")
        if best:
            print(f"[{stem}] BEST CANDIDATE investing({best[0]}) but median|reldiff|="
                  f"{best[1]*100:.3f}% on {best[2]} overlaps — NOT verified, do not add blindly")
        else:
            print(f"[{stem}] NO candidate matched")


if __name__ == "__main__":
    main()
