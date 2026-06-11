"""Scrape FOMC meeting (announcement) dates from the Federal Reserve site.

The announcement date is encoded UNAMBIGUOUSLY in the per-meeting file names:
  - historical transcripts: FOMC{YYYYMMDD}meeting.pdf   (last day of the meeting)
  - recent statements:      /pressreleases/monetary{YYYYMMDD}a.htm
Both give the exact announcement day and exclude unscheduled conference calls.

Sources (authoritative, reachable here):
  - recent: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  - historical by year: https://www.federalreserve.gov/monetarypolicy/fomchistorical{YYYY}.htm

Saves FOMC_dates.csv.
"""
import re
import datetime as dt
import requests
import pandas as pd

PROXY = "http://127.0.0.1:7897"
s = requests.Session()
s.proxies = {"http": PROXY, "https": PROXY}
s.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

DATE_PATS = [re.compile(r"FOMC(\d{8})meeting\.pdf"),
             re.compile(r"monetary(\d{8})a\d?\.htm")]

_MON = {m: i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August",
     "September","October","November","December"], 1)}
# "January 26-27" or cross-month "April 30-May 1" -> announcement = second day
_RANGE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"(\d{1,2})\s*[-–]\s*(?:(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+)?(\d{1,2})")


def dates_in(htmltext):
    out = set()
    for pat in DATE_PATS:
        for m in pat.finditer(htmltext):
            try:
                out.add(dt.datetime.strptime(m.group(1), "%Y%m%d").date())
            except ValueError:
                pass
    return out


def future_dates_in(calendar_html):
    """Parse upcoming scheduled meetings (date ranges) from the calendar page text,
    which lists future meetings that have no statement link yet."""
    txt = re.sub(r"<[^>]+>", " ", calendar_html)
    txt = re.sub(r"\s+", " ", txt)
    out = set()
    heads = list(re.finditer(r"(20\d{2}) FOMC Meetings", txt))
    for i, h in enumerate(heads):
        year = int(h.group(1))
        seg = txt[h.end(): heads[i + 1].start() if i + 1 < len(heads) else h.end() + 1500]
        for m in _RANGE.finditer(seg):
            mon = m.group(3) or m.group(1); day = int(m.group(4))
            try:
                out.add(dt.date(year, _MON[mon], day))
            except (ValueError, KeyError):
                pass
    return out


def main():
    found = set()
    rec = s.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", timeout=30).text
    found |= dates_in(rec)
    found |= future_dates_in(rec)            # upcoming scheduled meetings (no statement yet)
    for year in range(1999, 2021):
        try:
            h = s.get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm", timeout=30)
            if h.status_code == 200:
                found |= dates_in(h.text)
        except Exception as e:
            print("skip", year, type(e).__name__)
    dates = sorted(found)
    # FOMC has ~8 meetings/yr; flag years far from that as parse warnings
    df = pd.Series(1, index=pd.to_datetime(dates))
    by_year = df.groupby(df.index.year).count()
    print("meetings per year:")
    print(by_year.to_string())
    print(f"\ntotal {len(dates)} dates, {dates[0]} -> {dates[-1]}")
    pd.Series(dates, name="fomc_date").to_csv(r"C:\Users\ASUS\Desktop\claude doc\1\FOMC_dates.csv", index=False)
    print("saved -> FOMC_dates.csv")


if __name__ == "__main__":
    main()
