# -*- coding: utf-8 -*-
"""MARKET CALENDAR — unified scheduled-event table for the platform.
The one thing about future realized variance we genuinely know IN ADVANCE: the schedule.

Events:
  FOMC — from FOMC_dates.csv (Fed-published schedule; extends into 2027 -> forward-looking),
         emergency intermeeting cuts filtered out (min 20-day gap, as in examples/12);
  NFP  — first Friday of each month (rule-based; past and future);
  CPI  — from CPI_dates.csv (announced); future months PROJECTED from the median release
         day-of-month of the trailing 24 releases (marked source='projected').

API:
  calendar_df(start, end)      -> DataFrame(date, event, source) sorted
  event_flags(trading_index)   -> aligned DataFrame: fomc/nfp/cpi/any (event lands on the first
                                  trading day >= event date) + any_next (the FOLLOWING trading
                                  day — where overnight US announcements land for Asia/Europe)
                                  + pre_any (day BEFORE an event day: the short-entry trap zone)
  upcoming(n)                  -> next n events from today
CLI:  python market_calendar.py   -> writes event_calendar.csv + prints upcoming
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("REGIMELAB_DATA_DIR", r"C:\Users\ASUS\Desktop\claude doc\1")


def _fomc():
    f = pd.read_csv(os.path.join(DATA_DIR, "FOMC_dates.csv"), parse_dates=["fomc_date"])["fomc_date"]
    kept, last = [], None
    for d in f.sort_values():                       # scheduled-only: drop emergency intermeeting
        if last is None or (d - last).days >= 20:
            kept.append(d); last = d
    return pd.DataFrame({"date": kept, "event": "FOMC", "source": "scheduled"})


def _nfp(start="1999-01-01", months_ahead=12):
    end = pd.Timestamp.now().normalize() + pd.DateOffset(months=months_ahead)
    firsts = pd.date_range(start, end, freq="MS")
    fridays = [d + pd.Timedelta(days=(4 - d.dayofweek) % 7) for d in firsts]
    return pd.DataFrame({"date": fridays, "event": "NFP", "source": "rule"})


def _cpi(months_ahead=12):
    c = pd.read_csv(os.path.join(DATA_DIR, "CPI_dates.csv"), parse_dates=["cpi_date"])["cpi_date"]
    c = c.sort_values()
    out = pd.DataFrame({"date": c, "event": "CPI", "source": "scheduled"})
    # project future months from the trailing-24 median release day-of-month (weekday-adjusted)
    dom = int(c.tail(24).dt.day.median())
    last = c.max()
    proj = []
    cur = (last + pd.DateOffset(months=1)).replace(day=1)
    end = pd.Timestamp.now().normalize() + pd.DateOffset(months=months_ahead)
    while cur <= end:
        d = cur.replace(day=min(dom, 28))
        while d.dayofweek >= 5:                     # releases are weekdays
            d += pd.Timedelta(days=1)
        proj.append(d)
        cur = cur + pd.DateOffset(months=1)
    if proj:
        out = pd.concat([out, pd.DataFrame({"date": proj, "event": "CPI", "source": "projected"})])
    return out


def calendar_df(start="1999-01-01", end=None):
    cal = pd.concat([_fomc(), _nfp(), _cpi()], ignore_index=True)
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    cal = cal.drop_duplicates(["date", "event"]).sort_values(["date", "event"]).reset_index(drop=True)
    if start:
        cal = cal[cal["date"] >= pd.Timestamp(start)]
    if end:
        cal = cal[cal["date"] <= pd.Timestamp(end)]
    return cal


def event_flags(idx):
    """Align events to a trading DatetimeIndex. Returns DataFrame(fomc,nfp,cpi,any,any_next,pre_any)."""
    idx = pd.DatetimeIndex(idx)
    cal = calendar_df(start=str(idx.min().date()), end=str((idx.max() + pd.Timedelta(days=7)).date()))
    out = pd.DataFrame(0, index=idx, columns=["fomc", "nfp", "cpi"])
    for ev, grp in cal.groupby("event"):
        loc = idx.get_indexer(grp["date"].values, method="bfill")
        loc = loc[(loc >= 0) & (loc < len(idx))]
        out.iloc[loc, out.columns.get_loc(ev.lower())] = 1
    out["any"] = out[["fomc", "nfp", "cpi"]].max(axis=1)
    out["any_next"] = out["any"].shift(1).fillna(0).astype(int)   # overnight landing day
    out["pre_any"] = out["any"].shift(-1).fillna(0).astype(int)   # the day BEFORE an event
    return out


def upcoming(n=12):
    today = pd.Timestamp.now().normalize()
    cal = calendar_df(start=str(today.date()))
    return cal.head(n)


if __name__ == "__main__":
    cal = calendar_df()
    cal.to_csv(os.path.join(DATA_DIR, "event_calendar.csv"), index=False)
    print(f"event_calendar.csv -> {len(cal)} events "
          f"({cal['date'].min().date()} .. {cal['date'].max().date()})")
    print("\nUPCOMING:")
    for _, r in upcoming(12).iterrows():
        tag = " (projected)" if r["source"] == "projected" else ""
        print(f"  {r['date'].date()}  {r['event']}{tag}")
