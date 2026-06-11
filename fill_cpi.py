"""Drive fetch_cpi repeatedly (it's resumable) until every year 2000-2026 is complete.

Each round only re-fetches incomplete years (the proxy drops ~10-15% of chunks per pass),
sleeping between rounds to let the connection recover. Stops when all years have >=11 CPI
dates or after MAX_ROUNDS.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import fetch_cpi

OUT = fetch_cpi.OUT
MAX_ROUNDS = 10


def incomplete_years():
    if not os.path.exists(OUT):
        return list(range(2000, 2027))
    d = pd.read_csv(OUT, parse_dates=["cpi_date"])["cpi_date"]
    cnt = d.groupby(d.dt.year).count()
    return [y for y in range(2000, 2027) if cnt.get(y, 0) < 11]


for rnd in range(1, MAX_ROUNDS + 1):
    miss = incomplete_years()
    if not miss:
        break
    print(f"\n=== ROUND {rnd}: {len(miss)} incomplete years -> {miss} ===", flush=True)
    try:
        fetch_cpi.main()
    except Exception as e:
        print("round error (continuing):", type(e).__name__, str(e)[:80], flush=True)
    time.sleep(15)

miss = incomplete_years()
total = len(pd.read_csv(OUT)) if os.path.exists(OUT) else 0
print(f"\nDONE. {total} CPI dates. {'ALL YEARS COMPLETE' if not miss else 'still incomplete: ' + str(miss)}", flush=True)
