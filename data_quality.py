# -*- coding: utf-8 -*-
"""Data-quality guards for OHLCV CSVs. A single corrupt tick once flipped the sign of a headline
alpha in the semiconductor study (a Saturday-dated row ~40x its neighbours); we hit our own bad day
(NVDA -98% un-split-adjusted). This fails LOUD on structural corruption and warns on suspicious moves.

Usage:  from data_quality import audit_ohlcv; issues = audit_ohlcv(df, "VIX")
        hard = [i for i in issues if i.startswith("HARD")]   # block these"""
import numpy as np
import pandas as pd


def audit_ohlcv(df, name, move_cap=0.60):
    """Return a list of issue strings. 'HARD ' = corruption that breaks the BOOK's math (the book
    uses only High/Low (Parkinson range) and Close — never Open); 'WARN ' = suspicious but either
    legitimate (real vol spike) or harmless to this strategy (Open quirks, O/C bracketing)."""
    issues = []
    cols = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
    if "Close" not in df.columns:
        return [f"HARD {name}: no Close column"]

    # Close drives returns + the vol level -> NaN/non-positive Close is fatal
    n_nan_c = int(df["Close"].isna().sum())
    if n_nan_c:
        issues.append(f"WARN {name}.Close: {n_nan_c} NaN rows (dropped, but check)")
    if int((df["Close"] <= 0).sum()):
        issues.append(f"HARD {name}.Close: {int((df['Close'] <= 0).sum())} non-positive values")

    # High<Low breaks the Parkinson range ln(H/L); High/Low must be positive
    if {"High", "Low"} <= set(cols):
        if int((df["High"] < df["Low"]).sum()):
            issues.append(f"HARD {name}: {int((df['High'] < df['Low']).sum())} rows with High < Low")
        if int(((df["High"] <= 0) | (df["Low"] <= 0)).sum()):
            issues.append(f"HARD {name}: non-positive High/Low (breaks Parkinson ln(H/L))")

    # Open issues and O/C bracketing are WARN — the book never reads Open, and uses vol-index Close only
    if "Open" in cols and int((df["Open"] <= 0).sum()):
        issues.append(f"WARN {name}.Open: {int((df['Open'] <= 0).sum())} non-positive (unused by book)")
    if {"High", "Low", "Open", "Close"} <= set(cols):
        nb = int(((df["High"] < df[["Open", "Close"]].max(axis=1)) |
                  (df["Low"] > df[["Open", "Close"]].min(axis=1))).sum())
        if nb:
            issues.append(f"WARN {name}: {nb} rows where High/Low don't bracket O/C (vendor quirk)")

    if isinstance(df.index, pd.DatetimeIndex):
        if int(df.index.duplicated().sum()):
            issues.append(f"HARD {name}: {int(df.index.duplicated().sum())} duplicate dates")
        if not df.index.is_monotonic_increasing:
            issues.append(f"WARN {name}: index not sorted ascending")
        wknd = int((df.index.dayofweek >= 5).sum())
        if wknd:
            issues.append(f"WARN {name}: {wknd} weekend-dated rows")

    ret = df["Close"].pct_change(fill_method=None)
    ext = ret[ret.abs() > move_cap]
    if len(ext):
        issues.append(f"WARN {name}: {len(ext)} |move|>{move_cap*100:.0f}% "
                      f"(max {ext.abs().max()*100:.0f}% on {pd.Timestamp(ext.abs().idxmax()).date()})")
    return issues
