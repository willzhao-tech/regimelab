"""Shared daily-OHLCV market-data pipeline for the market study.

GENERAL RULE: every dataset is stored as daily OHLCV with the standard columns
    Date,Open,High,Low,Close,Volume
The instrument's identity is the file name; the schema is always the same.

One core (`update_dataset`) handles load -> incremental fetch -> merge -> atomic
write for any source. Sources are pluggable fetchers:
    - investing(pair_id)   -> Investing.com financial-data API OHLCV  [DEFAULT]
    - yahoo(ticker)        -> Yahoo Finance OHLCV  (alternative/fallback)

DEFAULT SOURCE = Investing.com. It serves OHLCV for everything in this study
(equity-index futures incl. China A50, rates, FX, commodities), paginates to any
history length, and matches the user's own Investing.com exports. Yahoo is kept
as a fallback for symbols Investing lacks. Stooq was REMOVED — its CSV endpoint
is now captcha-walled (needs a manual API key) and unusable headlessly.

Datasets are declared once in DATASETS; adding a new instrument is a new entry
there (and, if it needs its own schedule, a thin wrapper like update_nq.py).
To find an Investing pairId: GET api.investing.com/api/search/v2/search?q=<name>
with the browser headers in _INVESTING_HEADERS, or see find_investing_pair().

CLI:
    python market_data.py            # update every dataset
    python market_data.py NQ A50     # update named datasets
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta

import pandas as pd

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
DEFAULT_PROXY = os.environ.get("REGIMELAB_PROXY", "http://127.0.0.1:7897")
DATA_DIR = os.environ.get("REGIMELAB_DATA_DIR", r"C:\Users\ASUS\Desktop\claude doc\1")


# ---------------------------------------------------------------------------
# source fetchers: each returns fetch(start, proxy) -> OHLCV DataFrame
# ---------------------------------------------------------------------------
def yahoo(ticker: str):
    """Yahoo Finance daily OHLCV for `ticker`."""
    def fetch(start: str, proxy: str | None) -> pd.DataFrame:
        if proxy:
            os.environ["HTTP_PROXY"] = proxy
            os.environ["HTTPS_PROXY"] = proxy
        import yfinance as yf

        data = yf.download(ticker, start=start, auto_adjust=False, progress=False)
        if data is None or data.empty:
            return pd.DataFrame(columns=OHLCV_COLS)
        if hasattr(data.columns, "levels"):       # single ticker -> flatten (Price, Ticker)
            data.columns = [c[0] for c in data.columns]
        data = data[OHLCV_COLS].copy()
        data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
        data.index.name = "Date"
        return data.sort_index()
    return fetch


# Investing.com's API is Cloudflare-fronted and rejects thin clients with 403;
# it also caps each response at ~5000 rows, so long histories need pagination.
_INVESTING_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "domain-id": "www",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.investing.com/",
    "Origin": "https://www.investing.com",
}
_INVESTING_PAGE_CAP = 5000   # API returns at most ~5000 rows (oldest-first) per request


def _get_retry(sess, url, tries=4, backoff=2.0):
    """GET with retries on transient network/proxy errors (the China proxy drops
    connections intermittently). Raises the last error if all tries fail."""
    import requests
    last = None
    for i in range(tries):
        try:
            r = sess.get(url, timeout=60)
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ProxyError,
                requests.exceptions.Timeout) as e:
            last = e
            if i < tries - 1:
                time.sleep(backoff * (i + 1))
    raise last


def _investing_chunk(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLS)
    df = pd.DataFrame({
        "Open":   [float(x["last_openRaw"]) for x in rows],
        "High":   [float(x["last_maxRaw"]) for x in rows],
        "Low":    [float(x["last_minRaw"]) for x in rows],
        "Close":  [float(x["last_closeRaw"]) for x in rows],
        "Volume": [x.get("volumeRaw") for x in rows],
    }, index=pd.to_datetime([x["rowDateTimestamp"][:10] for x in rows]))
    df.index.name = "Date"
    return df.sort_index()


def investing(pair_id: int):
    """Investing.com financial-data API daily OHLCV for `pair_id` (paginated)."""
    api = ("https://api.investing.com/api/financialdata/historical/{pid}"
           "?start-date={start}&end-date={end}&time-frame=Daily&add-missing-rows=false")

    def fetch(start: str, proxy: str | None) -> pd.DataFrame:
        import requests

        sess = requests.Session()
        if proxy:
            sess.proxies = {"http": proxy, "https": proxy}
        sess.headers.update(_INVESTING_HEADERS)
        end = date.today().isoformat()

        # Paginate forward by DATE, not by page count: the API returns oldest-first
        # from start-date and caps each page (~4999-5000 rows), so we advance the
        # cursor to the day after each page's newest bar until a page reaches the
        # present or stops progressing. (Count-based termination is unreliable
        # because a "full" page is sometimes 4999, not exactly 5000.)
        frames, cursor, last_max, guard = [], start, None, 0
        while guard < 60:                       # backstop; ~60 pages is far beyond any series
            guard += 1
            url = api.format(pid=pair_id, start=cursor, end=end)
            r = _get_retry(sess, url)
            chunk = _investing_chunk(r.json().get("data", []))
            if chunk.empty:
                break
            frames.append(chunk)
            pmax = chunk.index.max()
            if last_max is not None and pmax <= last_max:   # no forward progress -> done
                break
            last_max = pmax
            nxt = (pmax + timedelta(days=1)).strftime("%Y-%m-%d")
            if nxt > end:                       # reached the present
                break
            cursor = nxt

        if not frames:
            return pd.DataFrame(columns=OHLCV_COLS)
        out = pd.concat(frames)
        return out[~out.index.duplicated(keep="last")].sort_index()
    return fetch


def find_investing_pair(query: str, proxy: str | None = DEFAULT_PROXY) -> list[dict]:
    """Search Investing.com for an instrument; return [{pairId, symbol, name, type}].

    Use this to discover the pairId for a new dataset, then add it to DATASETS as
    investing(<pairId>). Verify the price level/history before committing.
    """
    import requests

    sess = requests.Session()
    if proxy:
        sess.proxies = {"http": proxy, "https": proxy}
    sess.headers.update(_INVESTING_HEADERS)
    r = sess.get("https://api.investing.com/api/search/v2/search",
                 params={"q": query}, timeout=30)
    r.raise_for_status()
    out = []
    for q in r.json().get("quotes", []):
        out.append({"pairId": q.get("pairId") or q.get("id"),
                    "symbol": q.get("symbol"), "name": q.get("name"),
                    "type": q.get("type") or q.get("pair_type")})
    return out


# ---------------------------------------------------------------------------
# dataset registry
# ---------------------------------------------------------------------------
# Default source is Investing.com (pairId). Use yahoo(...) only as a fallback for
# symbols Investing lacks.
DATASETS = {
    "NQ": {
        "csv": os.path.join(DATA_DIR, "NQ_F_all_history.csv"),
        "fetcher": investing(8874),   # Nasdaq-100 future; OHLCV from 1999-06-22
        "earliest": "1999-01-01",
        "overlap_days": 3,
    },
    "A50": {
        "csv": os.path.join(DATA_DIR, "A50_all_history.csv"),
        "fetcher": investing(44486),  # SGX FTSE China A50 future; OHLCV from 2013-07-26
        "earliest": "2005-01-01",
        "overlap_days": 5,            # absorb late settlement / night-session revisions
    },
    # the rest of the 7-instrument research panel (NQ, A50, EURUSD, US10Y, WTI, XAU, VIX)
    "EURUSD": {
        "csv": os.path.join(DATA_DIR, "EURUSD_all_history.csv"),
        "fetcher": investing(1),      # EUR/USD spot
        "earliest": "1980-01-01",
        "overlap_days": 3,
    },
    "US10Y": {
        "csv": os.path.join(DATA_DIR, "US10Y_all_history.csv"),
        "fetcher": investing(8880),   # US 10-Year T-Note futures (price, not yield)
        "earliest": "1980-01-01",
        "overlap_days": 3,
    },
    "WTI": {
        "csv": os.path.join(DATA_DIR, "WTI_all_history.csv"),
        "fetcher": investing(8849),   # Crude Oil WTI futures
        "earliest": "1980-01-01",
        "overlap_days": 3,
    },
    "XAU": {
        "csv": os.path.join(DATA_DIR, "XAU_all_history.csv"),
        "fetcher": investing(68),     # Gold XAU/USD spot
        "earliest": "1980-01-01",
        "overlap_days": 3,
    },
    "VIX": {
        "csv": os.path.join(DATA_DIR, "VIX_all_history.csv"),
        "fetcher": investing(44336),  # CBOE Volatility Index
        "earliest": "1980-01-01",
        "overlap_days": 3,
    },
    "VXN": {
        "csv": os.path.join(DATA_DIR, "VXN_all_history.csv"),
        "fetcher": investing(44369),  # CBOE Nasdaq-100 Volatility Index (vol-arb signal input)
        "earliest": "2001-01-01",
        "overlap_days": 3,
    },
    "VIX1D": {
        "csv": os.path.join(DATA_DIR, "VIX1D_all_history.csv"),
        "fetcher": yahoo("^VIX1D"),   # CBOE 1-Day VIX (real 0-1DTE implied; ex-39 validation)
        "earliest": "2022-01-01",
        "overlap_days": 3,
    },
    "VIX9D": {
        "csv": os.path.join(DATA_DIR, "VIX9D_all_history.csv"),
        "fetcher": yahoo("^VIX9D"),   # CBOE 9-Day VIX (short-dated term-structure context)
        "earliest": "2010-01-01",
        "overlap_days": 3,
    },
    # ---- vol-book inputs (8-market floor book; ids EMPIRICALLY VERIFIED against the
    # existing CSVs by discover_pairs.py — median overlap |reldiff| 0.000-0.014%) ----
    "SPX":      {"csv": os.path.join(DATA_DIR, "SPX_all_history.csv"),
                 "fetcher": investing(166),   "earliest": "1980-01-01", "overlap_days": 3},
    "EEM":      {"csv": os.path.join(DATA_DIR, "EEM_all_history.csv"),
                 "fetcher": investing(505),   "earliest": "2003-01-01", "overlap_days": 3},
    "DAX":      {"csv": os.path.join(DATA_DIR, "DAX_all_history.csv"),
                 "fetcher": investing(172),   "earliest": "1980-01-01", "overlap_days": 3},
    "SX5E":     {"csv": os.path.join(DATA_DIR, "SX5E_all_history.csv"),
                 "fetcher": investing(175),   "earliest": "1987-01-01", "overlap_days": 3},
    "N225":     {"csv": os.path.join(DATA_DIR, "N225_all_history.csv"),
                 "fetcher": investing(178),   "earliest": "1980-01-01", "overlap_days": 3},
    "HSI":      {"csv": os.path.join(DATA_DIR, "HSI_all_history.csv"),
                 "fetcher": investing(179),   "earliest": "1986-01-01", "overlap_days": 3},
    "NSEI":     {"csv": os.path.join(DATA_DIR, "NSEI_all_history.csv"),
                 "fetcher": investing(17940), "earliest": "1994-01-01", "overlap_days": 3},
    "INDIAVIX": {"csv": os.path.join(DATA_DIR, "INDIAVIX_all_history.csv"),
                 "fetcher": investing(17942), "earliest": "2007-01-01", "overlap_days": 3},
    "VDAX":     {"csv": os.path.join(DATA_DIR, "VDAX_all_history.csv"),
                 "fetcher": investing(19133), "earliest": "1992-01-01", "overlap_days": 3},
    "VSTOXX":   {"csv": os.path.join(DATA_DIR, "VSTOXX_all_history.csv"),
                 "fetcher": investing(40817), "earliest": "1999-01-01", "overlap_days": 3},
    "JNIV":     {"csv": os.path.join(DATA_DIR, "JNIV_all_history.csv"),
                 "fetcher": investing(28878), "earliest": "1998-01-01", "overlap_days": 3},
    "VHSI":     {"csv": os.path.join(DATA_DIR, "VHSI_all_history.csv"),
                 "fetcher": investing(49577), "earliest": "2001-01-01", "overlap_days": 3},
    "VXEEM":    {"csv": os.path.join(DATA_DIR, "VXEEM_all_history.csv"),
                 "fetcher": investing(44340), "earliest": "2011-01-01", "overlap_days": 3},
}


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
def _load_existing(csv_path: str) -> pd.DataFrame:
    """Return the stored OHLCV frame (empty if absent or schema doesn't match)."""
    empty = pd.DataFrame(columns=OHLCV_COLS)
    empty.index = pd.DatetimeIndex([], name="Date")
    if not os.path.exists(csv_path):
        return empty
    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
    if not set(OHLCV_COLS).issubset(df.columns):    # older/foreign schema -> rebuild
        return empty
    return df[OHLCV_COLS].sort_index()


def update_dataset(csv_path, fetcher, earliest, overlap_days=1, proxy=DEFAULT_PROXY) -> dict:
    """Update an OHLCV CSV in place with bars newer than what it holds.

    Re-fetches the last `overlap_days` so revised/late bars are picked up; fresh
    bars win on overlap. A missing file or schema change triggers a full rebuild.

    Industry-discipline guards (prop-shop P1/P2 analogues):
      - REVISION DETECTION: any overlap bar whose Close changes vs what is on disk
        is logged loudly (vendors must not rewrite history silently);
      - QUALITY GATE: the newly fetched rows are audited (data_quality.audit_ohlcv)
        BEFORE the merge is committed — a HARD violation (High<Low, non-positive
        Close, duplicate dates) aborts the write, so a corrupt tick never reaches disk.
    """
    existing = _load_existing(csv_path)

    if existing.empty:
        start = earliest
    else:
        start = (existing.index.max() - timedelta(days=overlap_days)).strftime("%Y-%m-%d")

    fetched = fetcher(start, proxy)

    # ---- revision detection on the overlap window --------------------------------
    revisions = []
    if not existing.empty and not fetched.empty:
        common = existing.index.intersection(fetched.index)
        if len(common):
            old_c = existing.loc[common, "Close"].astype(float)
            new_c = fetched.loc[common, "Close"].astype(float)
            chg = ((new_c - old_c).abs() / old_c.abs().clip(lower=1e-12)) > 1e-4   # >1bp = real revision
            for d in common[chg]:
                revisions.append((d.date().isoformat(),
                                  float(old_c.loc[d]), float(new_c.loc[d])))
            for d, o, n in revisions:
                print(f"  [REVISED] {os.path.basename(csv_path)} {d}: Close {o:g} -> {n:g}")

    # ---- quality gate on the NEW rows (historical quirks on disk aren't re-litigated)
    if not fetched.empty:
        try:
            from data_quality import audit_ohlcv
            hard = [i for i in audit_ohlcv(fetched.dropna(subset=["Close"]),
                                           os.path.basename(csv_path))
                    if i.startswith("HARD")]
        except ImportError:
            hard = []
        if hard:
            print(f"  [GATE] REFUSING to commit {os.path.basename(csv_path)} — "
                  f"corrupt fetch:\n    " + "\n    ".join(hard))
            return {"rows_added": 0, "last_date": (existing.index.max().date().isoformat()
                                                   if len(existing) else None),
                    "total_rows": len(existing), "path": csv_path,
                    "blocked": hard, "revisions": len(revisions)}

    combined = fetched if existing.empty else pd.concat([existing, fetched])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined = combined.dropna(subset=["Close"])
    combined["Volume"] = combined["Volume"].astype("Int64")

    rows_added = len(combined) - len(existing)

    if rows_added > 0 or not existing.equals(combined):
        tmp = csv_path + ".tmp"
        combined.to_csv(tmp, index_label="Date")
        os.replace(tmp, csv_path)

    return {
        "rows_added": max(rows_added, 0),
        "last_date": (combined.index.max().date().isoformat() if len(combined) else None),
        "total_rows": len(combined),
        "path": csv_path,
        "revisions": len(revisions),
    }


def update_one(name: str, proxy: str | None = DEFAULT_PROXY) -> dict:
    """Update a single registered dataset by name (e.g. "NQ", "A50")."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(DATASETS)}")
    d = DATASETS[name]
    return update_dataset(d["csv"], d["fetcher"], d["earliest"],
                          d.get("overlap_days", 1), proxy)


def _report(name: str, info: dict) -> None:
    if info.get("blocked"):
        print(f"[{name}] BLOCKED by quality gate ({len(info['blocked'])} HARD violation(s)); "
              f"file unchanged at {info['total_rows']} rows through {info['last_date']}")
    elif info["rows_added"]:
        rev = f", {info['revisions']} revised" if info.get("revisions") else ""
        print(f"[{name}] added {info['rows_added']} row(s){rev}; now {info['total_rows']} "
              f"rows through {info['last_date']}")
    else:
        print(f"[{name}] already up to date: {info['total_rows']} rows through {info['last_date']}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(DATASETS)
    for nm in names:
        _report(nm, update_one(nm))
