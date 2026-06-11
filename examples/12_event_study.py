"""
Example 12 — macro-event catalyst study on NQ: the pre-FOMC drift (and NFP).

Tests the one catalyst with a documented prior — the PRE-FOMC ANNOUNCEMENT DRIFT
(Lucca & Moench 2015: equities drift up in the ~24h before FOMC decisions). Their sample
was 1994-2011; we test 2013-2026, a genuine POST-PUBLICATION out-of-sample window.

Event dates (auditable):
  FOMC announcements  : federalreserve.gov per-year pages, the 2-day scheduled meetings
                        (announcement = 2nd day), 2013+ where the 2-day parser is complete.
                        Cached to data dir as fomc_dates.csv.
  NFP (jobs report)   : first Friday of each month (derived; the standard BLS release day).

Method: around each event, mean NQ close-to-close return on D-1 / D0 / D+1 vs the
unconditional daily mean; t-stats; a PLACEBO null (random non-event dates, same count) so
we know the event window beats random; and the share of NQ's total return earned on FOMC days.

Run:  python examples/12_event_study.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\claude doc\1"
PROXY = "http://127.0.0.1:7897"
FOMC_CSV = os.path.join(DATA_DIR, "fomc_dates.csv")
MONTHS = {m: i + 1 for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"])}


def _two_day_meetings(year, text):
    import re
    out = []
    for mo, d1, d2 in re.findall(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})[-–](\d{1,2})", text):
        out.append(f"{year}-{MONTHS[mo]:02d}-{int(d2):02d}")
    for mo1, d1, mo2, d2 in re.findall(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}[-–]"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})", text):
        out.append(f"{year}-{MONTHS[mo2]:02d}-{int(d2):02d}")
    return set(out)


def _get(s, url, tries=5):
    import time, requests
    for i in range(tries):
        try:
            r = s.get(url, timeout=40)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None


def fetch_fomc_dates(start_year=2013, end_year=2026):
    if os.path.exists(FOMC_CSV):
        return pd.to_datetime(pd.read_csv(FOMC_CSV)["fomc_announcement"]).sort_values()
    import requests
    s = requests.Session(); s.proxies = {"http": PROXY, "https": PROXY}; s.headers["User-Agent"] = "Mozilla/5.0"
    dates = set()
    for y in range(start_year, end_year + 1):
        t = _get(s, f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm")
        if t:
            dates |= _two_day_meetings(y, t)
    # current/upcoming years live on the calendar page; parse with year context
    t = _get(s, "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    if t:
        import re
        cur = None
        for tok in re.finditer(r"(20[1-3]\d)|((January|February|March|April|May|June|July|August|September|"
                               r"October|November|December)\s+\d{1,2}[-–]\d{1,2})", t):
            if tok.group(1):
                cur = int(tok.group(1))
            elif cur and start_year <= cur <= end_year:
                dates |= _two_day_meetings(cur, tok.group(2))
    except Exception:
        pass
    d = pd.to_datetime(sorted(dates))
    pd.DataFrame({"fomc_announcement": d}).to_csv(FOMC_CSV, index=False)
    return pd.Series(d)


def nfp_dates(start, end):
    """First Friday of each month (standard NFP release day)."""
    out = []
    for ts in pd.date_range(start, end, freq="MS"):
        d = ts + pd.Timedelta(days=(4 - ts.dayofweek) % 7)   # first Friday
        out.append(d)
    return pd.to_datetime(out)


def event_window(ret, events):
    """Map events to trading-day positions; return DataFrames of D-1,D0,D+1 returns (bps)."""
    idx = ret.index
    rows = {"D-1": [], "D0": [], "D+1": []}
    for e in events:
        pos = idx.searchsorted(e)
        if pos <= 0 or pos >= len(idx) - 1:
            continue
        # align D0 to the event's trading day (first trading day >= e)
        rows["D-1"].append(ret.iloc[pos - 1]); rows["D0"].append(ret.iloc[pos]); rows["D+1"].append(ret.iloc[pos + 1])
    return {k: np.array(v) for k, v in rows.items()}


def t_of(a):
    a = a[np.isfinite(a)]
    return float(a.mean() / (a.std() / np.sqrt(len(a)))) if len(a) > 2 and a.std() > 0 else float("nan")


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "NQ_F_all_history.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
    ret = df["Close"].pct_change(fill_method=None).dropna()
    base_bps = ret.mean() * 1e4

    fomc = fetch_fomc_dates()
    fomc = fomc[(fomc >= ret.index.min()) & (fomc <= ret.index.max())]
    print(f"NQ {ret.index.min().date()}..{ret.index.max().date()}  | unconditional mean {base_bps:.1f} bps/day")
    print(f"FOMC announcements used: {len(fomc)} ({fomc.min().date()}..{fomc.max().date()}; source federalreserve.gov)\n")

    print("=" * 80)
    print("PRE-FOMC DRIFT — mean NQ return by day relative to the announcement (bps)")
    print("=" * 80)
    w = event_window(ret, fomc)
    print(f"  {'window':<8}{'mean bps':>10}{'t-stat':>9}{'vs base':>10}{'n':>6}")
    for k in ["D-1", "D0", "D+1"]:
        a = w[k]
        print(f"  {k:<8}{a.mean()*1e4:>9.1f}{t_of(a):>9.1f}{(a.mean()*1e4-base_bps):>+9.1f}{len(a):>6}")
    pre = w["D-1"] + w["D0"]
    print(f"  {'[D-1+D0]':<8}{pre.mean()*1e4:>9.1f}{t_of(pre):>9.1f}{'':>10}{len(pre):>6}  (the ~24h drift window)")

    # placebo null: random non-FOMC dates, same count, distribution of mean D-1 return
    rng = np.random.default_rng(7); pool = ret.index[5:-5]
    plac = []
    for _ in range(2000):
        samp = pd.DatetimeIndex(rng.choice(pool, size=len(fomc), replace=False))
        plac.append(event_window(ret, samp)["D-1"].mean())
    plac = np.array(plac)
    pct = (plac < w["D-1"].mean()).mean() * 100
    print(f"\n  PLACEBO null (2000x random dates): real D-1 mean {w['D-1'].mean()*1e4:.1f} bps beats "
          f"{pct:.0f}% of random sets")

    # share of total NQ return earned on FOMC D-1 & D0 days
    fomc_pos = [ret.index.searchsorted(e) for e in fomc]
    mask = np.zeros(len(ret), bool)
    for p in fomc_pos:
        if 0 < p < len(ret):
            mask[p] = True; mask[p - 1] = True
    share = (ret.values[mask].sum()) / ret.values.sum() * 100
    print(f"  FOMC D-1&D0 are {mask.mean()*100:.1f}% of days but earned {share:.0f}% of NQ's total return")

    # NFP
    print("\n" + "=" * 80)
    print("NFP (first Friday) — event-day behavior")
    print("=" * 80)
    wn = event_window(ret, nfp_dates(ret.index.min(), ret.index.max()))
    print(f"  {'window':<8}{'mean bps':>10}{'t-stat':>9}{'vs base':>10}{'n':>6}")
    for k in ["D-1", "D0", "D+1"]:
        a = wn[k]
        print(f"  {k:<8}{a.mean()*1e4:>9.1f}{t_of(a):>9.1f}{(a.mean()*1e4-base_bps):>+9.1f}{len(a):>6}")

    print("\nNOTE: FOMC 2013-2026 is post-publication (Lucca-Moench used 1994-2011) -> a real OOS")
    print("test. t-stats are single-hypothesis; the placebo is the key check. NFP first-Friday is")
    print("a derived approximation (rare 2nd-Friday releases not corrected).")


if __name__ == "__main__":
    main()
