"""Scrape US CPI release dates from the Investing.com economic-calendar API.

POST https://www.investing.com/economic-calendar/Service/getCalendarFilteredData
filtered to US (country 5) inflation events; parse the headline "Consumer Price Index (CPI)"
rows (MoM/YoY release the same day). Chunked over the full history. Saves CPI_dates.csv.
"""
import os
import re
import datetime as dt
import time
import requests
import pandas as pd

PROXY = "http://127.0.0.1:7897"
URL = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
s = requests.Session()
s.proxies = {"http": PROXY, "https": PROXY}
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest", "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "https://www.investing.com/economic-calendar/", "Origin": "https://www.investing.com"})

SEG = re.compile(r'class="theDay" id="theDay(\d+)"')
EVT = re.compile(r'class="[^"]*event"[^>]*>(.*?)</td>', re.S)
HEADLINE_CPI = re.compile(r"^CPI \((MoM|YoY)\)")     # exclude Core/Cleveland/Index variants


def cpi_dates(html):
    out = set()
    parts = SEG.split(html)               # [pre, epoch, content, epoch, content, ...]
    for i in range(1, len(parts), 2):
        epoch = int(parts[i]); content = parts[i + 1]
        for m in EVT.finditer(content):
            name = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            if HEADLINE_CPI.match(name):
                out.add(dt.datetime.fromtimestamp(epoch, dt.timezone.utc).date())
    return out


def fetch_chunk(d0, d1):
    """Return the data HTML, or None on persistent failure (never raises)."""
    body = {"country[]": "5", "category[]": "_inflation",
            "dateFrom": d0, "dateTo": d1, "timeZone": "55",
            "timeFilter": "timeRemain", "currentTab": "custom", "limit_from": "0"}
    for attempt in range(6):
        try:
            r = s.post(URL, data=body, timeout=40)
            if r.status_code == 429:                 # rate-limited -> long backoff
                time.sleep(20 * (attempt + 1)); continue
            r.raise_for_status()
            return r.json().get("data", "")
        except Exception:
            time.sleep(min(5 * (attempt + 1), 30))
    return None


OUT = r"C:\Users\ASUS\Desktop\claude doc\1\CPI_dates.csv"


def main():
    found = set()
    if os.path.exists(OUT):                          # resume: accumulate across runs
        found |= set(pd.read_csv(OUT, parse_dates=["cpi_date"])["cpi_date"].dt.date)
    incomplete = []
    for y in range(2000, 2027):
        have = sum(1 for d in found if d.year == y)
        if have >= 11:                               # year already complete -> skip
            print(f"  {y}: {have} (have, skip)", flush=True); continue
        for d0, d1 in [(f"{y}-01-01", f"{y}-06-30"), (f"{y}-07-01", f"{y}-12-31")]:
            data = fetch_chunk(d0, d1)
            if data is not None:
                found |= cpi_dates(data)
            time.sleep(4)
        pd.Series(sorted(found), name="cpi_date").to_csv(OUT, index=False)
        yr = sum(1 for d in found if d.year == y)
        if yr < 11:
            incomplete.append(y)
        print(f"  {y}: {yr} CPI dates ({'OK' if yr >= 11 else 'PARTIAL'}); total {len(found)}", flush=True)
    print(f"\nincomplete years (re-run to fill): {incomplete}" if incomplete else "\nall years complete")
    dates = sorted(found)
    df = pd.Series(1, index=pd.to_datetime(dates))
    print("\nper-year counts:")
    print(df.groupby(df.index.year).count().to_string())
    print(f"\ntotal {len(dates)} CPI release dates, {dates[0]} -> {dates[-1]}")
    pd.Series(dates, name="cpi_date").to_csv(r"C:\Users\ASUS\Desktop\claude doc\1\CPI_dates.csv", index=False)
    print("saved -> CPI_dates.csv")


if __name__ == "__main__":
    main()
