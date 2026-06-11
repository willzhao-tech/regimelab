import re, requests
s = requests.Session(); s.proxies = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}
s.headers["User-Agent"] = "Mozilla/5.0"
MONTHS = {m: i+1 for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"])}

def meetings_for_year(year, text):
    """Scheduled FOMC meetings are 2-day -> 'Month D-D' (announcement = 2nd day)."""
    out = []
    for mo, d1, d2 in re.findall(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})[-–](\d{1,2})", text):
        out.append(f"{year}-{MONTHS[mo]:02d}-{int(d2):02d}")
    # cross-month meetings e.g. 'October 31-November 1'
    for mo1, d1, mo2, d2 in re.findall(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})[-–](January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})", text):
        out.append(f"{year}-{MONTHS[mo2]:02d}-{int(d2):02d}")
    return sorted(set(out))

for y in [2008, 2013, 2016, 2019]:
    t = s.get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm", timeout=30).text
    mtgs = meetings_for_year(y, t)
    print(f"{y}: {len(mtgs)} two-day meetings -> {mtgs}")
