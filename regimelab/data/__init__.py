"""
regimelab.data
==============

The data layer turns heterogeneous free sources into one common `Panel`.

Design:
  - Every source subclasses `DataSource` and implements `.fetch() -> SourceResult`.
  - `build_panel([...sources...])` fetches each, aligns them on a shared date
    index, and assembles a single Panel (returns + macro + events).
  - Live backends (fredapi, pandas-datareader, yfinance) are imported LAZILY
    inside `.fetch()`, so importing regimelab never requires them. A missing
    backend raises a clear, actionable error naming the pip extra to install.
  - Results are cached to disk (Parquet) keyed by source signature, so repeated
    runs are offline and the free APIs are not hammered.

Sources implemented here (all free, per the data-sourcing study):
  - InvestingSource     : daily OHLCV, the DEFAULT source           (requests)
  - FredSource          : macro / rates / FX / commodities / VIX  (fredapi)
  - FamaFrenchSource    : factor returns 1926+                     (pandas-datareader)
  - YahooSource         : adjusted prices + dividends              (yfinance)
  - LocalCSVSource      : a directory of CSVs (offline / bring-your-own)
  - LegacyJsonSource    : the original panel.json (for continuity & tests)

DEFAULT SOURCE: InvestingSource. It serves OHLCV for every instrument in the
study and paginates to any history length. StooqSource was REMOVED — Stooq's CSV
endpoint is now captcha-walled (a manual API key is required) and cannot be used
headlessly.

Adding a new source = one new subclass. Nothing above the data layer changes.
"""

from __future__ import annotations

import os
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Mapping, Sequence, Any

import numpy as np
import pandas as pd

from ..panel import Panel


# ---------------------------------------------------------------------------
# result container + caching
# ---------------------------------------------------------------------------
@dataclass
class SourceResult:
    """What a source returns: any subset of returns / prices / macro / events."""
    returns: Optional[pd.DataFrame] = None
    prices: Optional[pd.DataFrame] = None
    macro: Optional[pd.DataFrame] = None
    events: Optional[pd.DataFrame] = None
    provenance: dict = field(default_factory=dict)


_CACHE_DIR = os.environ.get("REGIMELAB_CACHE", os.path.expanduser("~/.regimelab_cache"))


def _cache_path(signature: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    h = hashlib.sha1(signature.encode()).hexdigest()[:16]
    return os.path.join(_CACHE_DIR, f"{h}.parquet")


def _missing_backend(pkg: str, extra: str) -> "ImportError":
    return ImportError(
        f"The '{pkg}' package is required for this data source but is not installed.\n"
        f"Install the live-data backends with:  pip install 'regimelab[{extra}]'\n"
        f"(or: pip install {pkg})"
    )


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------
class DataSource(ABC):
    """Abstract source. Subclasses implement `_fetch_uncached`; caching is shared."""

    #: short name used in provenance + cache signature
    name: str = "source"

    def signature(self) -> str:
        """Stable string identifying this source's request, for cache keying."""
        return f"{self.name}:{json.dumps(self._signature_payload(), sort_keys=True, default=str)}"

    @abstractmethod
    def _signature_payload(self) -> dict: ...

    @abstractmethod
    def _fetch_uncached(self) -> SourceResult: ...

    def fetch(self, use_cache: bool = True) -> SourceResult:
        path = _cache_path(self.signature())
        if use_cache and os.path.exists(path):
            df = pd.read_parquet(path)
            kind = df.attrs.get("regimelab_kind", "returns")
            return SourceResult(**{kind: df}, provenance={"cached": True, "name": self.name})
        result = self._fetch_uncached()
        # cache the primary frame (returns preferred, else prices, else macro)
        primary = result.returns if result.returns is not None else (
            result.prices if result.prices is not None else result.macro)
        if use_cache and primary is not None:
            kind = "returns" if result.returns is not None else (
                "prices" if result.prices is not None else "macro")
            primary = primary.copy()
            primary.attrs["regimelab_kind"] = kind
            try:
                primary.to_parquet(path)
            except Exception:
                pass  # caching is best-effort; never fail a fetch on cache write
        return result


# ---------------------------------------------------------------------------
# FRED  (macro / rates / FX / commodities / VIX)
# ---------------------------------------------------------------------------
class FredSource(DataSource):
    """
    Fetch series from FRED. `series` maps your instrument/macro name -> FRED code.
    `as_macro=True` routes them to Panel.macro (regime conditioning); otherwise
    they are treated as levels and converted to returns (e.g. a price series).

    Yields (DGS10 etc.) are LEVELS, not returns — keep them as macro, or set
    `diff=True` to use first differences. Requires a free FRED API key:
        https://fred.stlouisfed.org/docs/api/api_key.html
    """
    name = "fred"

    def __init__(
        self,
        series: Mapping[str, str],
        api_key: Optional[str] = None,
        as_macro: bool = True,
        diff: bool = False,
        start: Optional[str] = None,
    ):
        self.series = dict(series)
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.as_macro = as_macro
        self.diff = diff
        self.start = start

    def _signature_payload(self) -> dict:
        return {"series": self.series, "as_macro": self.as_macro,
                "diff": self.diff, "start": self.start}

    def _fetch_uncached(self) -> SourceResult:
        try:
            from fredapi import Fred
        except ImportError:
            raise _missing_backend("fredapi", "data")
        if not self.api_key:
            raise ValueError(
                "FredSource needs a FRED API key. Pass api_key=... or set FRED_API_KEY. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        fred = Fred(api_key=self.api_key)
        cols = {}
        for name, code in self.series.items():
            s = fred.get_series(code, observation_start=self.start)
            cols[name] = s
        df = pd.DataFrame(cols)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if self.diff:
            df = df.diff().iloc[1:]
        prov = {"name": self.name, "series": self.series}
        if self.as_macro:
            return SourceResult(macro=df, provenance=prov)
        # treat as price levels -> fractional returns
        returns = df.pct_change().iloc[1:]
        return SourceResult(returns=returns, prices=df, provenance=prov)


# ---------------------------------------------------------------------------
# Fama-French factor library  (returns, 1926+)
# ---------------------------------------------------------------------------
class FamaFrenchSource(DataSource):
    """
    Fetch factor returns from the Kenneth French Data Library via pandas-datareader.
    `dataset` selects the French dataset; `factors` selects columns to keep.
    Values arrive in percent and are converted to fractional returns.
    """
    name = "famafrench"

    def __init__(
        self,
        factors: Optional[Sequence[str]] = None,
        dataset: str = "F-F_Research_Data_5_Factors_2x3_daily",
        start: Optional[str] = "1990-01-01",
    ):
        self.factors = list(factors) if factors else None
        self.dataset = dataset
        self.start = start

    def _signature_payload(self) -> dict:
        return {"dataset": self.dataset, "factors": self.factors, "start": self.start}

    def _fetch_uncached(self) -> SourceResult:
        try:
            import pandas_datareader.data as web
        except ImportError:
            raise _missing_backend("pandas-datareader", "data")
        raw = web.DataReader(self.dataset, "famafrench", start=self.start)
        df = raw[0].copy()  # table 0 is the factor returns
        df.index = pd.to_datetime(df.index.astype(str))
        df = df / 100.0  # percent -> fractional
        df.columns = [c.strip() for c in df.columns]
        if self.factors:
            keep = [c for c in df.columns if c in self.factors]
            df = df[keep]
        return SourceResult(returns=df, provenance={"name": self.name, "dataset": self.dataset})


# ---------------------------------------------------------------------------
# Investing.com  (daily OHLCV via the financial-data API) — DEFAULT SOURCE
# ---------------------------------------------------------------------------
class InvestingSource(DataSource):
    """
    Daily prices from Investing.com's financial-data API. `pairs` maps your
    instrument name -> Investing pairId (e.g. {"NQ": 8874, "A50": 44486}).
    Find a pairId via the search endpoint (see regimelab root market_data.py
    `find_investing_pair`). Returns close levels -> fractional returns (the panel
    works in returns); full OHLCV is available in the raw export pipeline.

    This is the default source: it serves every instrument in the study, needs no
    API key, and paginates to any history length (the API caps ~5000 rows/page).
    Requires the local proxy from blocked regions; pass `proxy=` or rely on the
    HTTP(S)_PROXY environment variables.
    """
    name = "investing"

    _HEADERS = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "domain-id": "www",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.investing.com/",
        "Origin": "https://www.investing.com",
    }
    _PAGE_CAP = 5000

    def __init__(self, pairs: Mapping[str, int], start: str = "1990-01-01",
                 end: Optional[str] = None, proxy: Optional[str] = None):
        self.pairs = dict(pairs)
        self.start = start
        self.end = end
        self.proxy = proxy

    def _signature_payload(self) -> dict:
        return {"pairs": self.pairs, "start": self.start, "end": self.end}

    def _fetch_pair(self, sess, pid: int) -> pd.Series:
        import datetime as _dt
        api = ("https://api.investing.com/api/financialdata/historical/{pid}"
               "?start-date={start}&end-date={end}&time-frame=Daily&add-missing-rows=false")
        end = self.end or _dt.date.today().isoformat()
        frames, cursor, guard = [], self.start, 0
        while guard < 50:
            guard += 1
            r = sess.get(api.format(pid=pid, start=cursor, end=end), timeout=60)
            r.raise_for_status()
            rows = r.json().get("data", [])
            if not rows:
                break
            s = pd.Series(
                [float(x["last_closeRaw"]) for x in rows],
                index=pd.to_datetime([x["rowDateTimestamp"][:10] for x in rows]),
            ).sort_index()
            frames.append(s)
            if len(rows) < self._PAGE_CAP:
                break
            cursor = (s.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if cursor > end:
                break
        if not frames:
            return pd.Series(dtype="float64")
        out = pd.concat(frames)
        return out[~out.index.duplicated(keep="last")].sort_index()

    def _fetch_uncached(self) -> SourceResult:
        import requests
        sess = requests.Session()
        proxy = self.proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if proxy:
            sess.proxies = {"http": proxy, "https": proxy}
        sess.headers.update(self._HEADERS)
        closes = {name: self._fetch_pair(sess, pid) for name, pid in self.pairs.items()}
        prices = pd.DataFrame(closes).sort_index()
        return Panel_from_prices_result(prices, self.name, {"pairs": self.pairs})


# ---------------------------------------------------------------------------
# Yahoo  (adjusted prices; convenient but fragile)
# ---------------------------------------------------------------------------
class YahooSource(DataSource):
    """Adjusted daily prices from Yahoo via yfinance. Fragile; cache aggressively."""
    name = "yahoo"

    def __init__(self, tickers: Mapping[str, str], start: str = "1990-01-01", end: Optional[str] = None):
        self.tickers = dict(tickers)
        self.start = start
        self.end = end

    def _signature_payload(self) -> dict:
        return {"tickers": self.tickers, "start": self.start, "end": self.end}

    def _fetch_uncached(self) -> SourceResult:
        try:
            import yfinance as yf
        except ImportError:
            raise _missing_backend("yfinance", "data")
        syms = list(self.tickers.values())
        data = yf.download(syms, start=self.start, end=self.end, auto_adjust=True, progress=False)
        close = data["Close"] if "Close" in data else data
        if isinstance(close, pd.Series):
            close = close.to_frame()
        # map Yahoo symbols back to friendly names
        inv = {v: k for k, v in self.tickers.items()}
        close = close.rename(columns=inv)
        return Panel_from_prices_result(close.sort_index(), self.name, {"tickers": self.tickers})


# ---------------------------------------------------------------------------
# Local CSV directory  (offline / bring-your-own)
# ---------------------------------------------------------------------------
class LocalCSVSource(DataSource):
    """
    Read a directory of CSVs (one per instrument, a Date column + a value column),
    or a single wide CSV. Pure-offline; the escape hatch when an API is down or
    you have your own data. `kind` routes the result ('returns'|'prices'|'macro').
    """
    name = "localcsv"

    def __init__(self, path: str, kind: str = "prices", date_col: str = "Date", value_col: str = "Close"):
        self.path = path
        self.kind = kind
        self.date_col = date_col
        self.value_col = value_col

    def _signature_payload(self) -> dict:
        return {"path": os.path.abspath(self.path), "kind": self.kind}

    def _fetch_uncached(self) -> SourceResult:
        if os.path.isdir(self.path):
            cols = {}
            for fn in sorted(os.listdir(self.path)):
                if not fn.lower().endswith(".csv"):
                    continue
                name = os.path.splitext(fn)[0]
                df = pd.read_csv(os.path.join(self.path, fn))
                df[self.date_col] = pd.to_datetime(df[self.date_col])
                cols[name] = df.set_index(self.date_col)[self.value_col]
            wide = pd.DataFrame(cols).sort_index()
        else:
            wide = pd.read_csv(self.path)
            wide[self.date_col] = pd.to_datetime(wide[self.date_col])
            wide = wide.set_index(self.date_col).sort_index()
        if self.kind == "prices":
            return Panel_from_prices_result(wide, self.name, {"path": self.path})
        return SourceResult(**{self.kind: wide}, provenance={"name": self.name, "path": self.path})


# ---------------------------------------------------------------------------
# Legacy panel.json  (continuity with the original research)
# ---------------------------------------------------------------------------
class LegacyJsonSource(DataSource):
    """Load the original panel.json (values in percent) as a returns source."""
    name = "legacyjson"

    def __init__(self, path: str, key: str = "series"):
        self.path = path
        self.key = key

    def _signature_payload(self) -> dict:
        return {"path": os.path.abspath(self.path), "key": self.key}

    def _fetch_uncached(self) -> SourceResult:
        with open(self.path) as fh:
            blob = json.load(fh)
        series = blob[self.key]
        frames = {
            inst: pd.Series({pd.to_datetime(d): v / 100.0 for d, v in obs.items()})
            for inst, obs in series.items()
        }
        returns = pd.DataFrame(frames).sort_index()
        return SourceResult(returns=returns, provenance={"name": self.name, "path": self.path})


# ---------------------------------------------------------------------------
# helper + assembler
# ---------------------------------------------------------------------------
def Panel_from_prices_result(prices: pd.DataFrame, name: str, prov: dict) -> SourceResult:
    """Build a SourceResult (returns + prices) from a price-level frame."""
    prices = prices.sort_index()
    returns = prices.pct_change().iloc[1:]
    return SourceResult(returns=returns, prices=prices, provenance={"name": name, **prov})


def build_panel(
    sources: Sequence[DataSource],
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_cache: bool = True,
    how: str = "outer",
) -> Panel:
    """
    Fetch each source and assemble one Panel.

    Returns/prices frames are concatenated column-wise; macro and events frames
    are concatenated separately. Dates are unioned (`how='outer'`) or intersected
    (`how='inner'`). The result is the single object the rest of the platform uses.
    """
    ret_frames, price_frames, macro_frames, event_frames = [], [], [], []
    provenance = []
    for src in sources:
        res = src.fetch(use_cache=use_cache)
        provenance.append(res.provenance)
        if res.returns is not None:
            ret_frames.append(res.returns)
        if res.prices is not None:
            price_frames.append(res.prices)
        if res.macro is not None:
            macro_frames.append(res.macro)
        if res.events is not None:
            event_frames.append(res.events)

    def _join(frames: list[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if not frames:
            return None
        out = pd.concat(frames, axis=1, join=("inner" if how == "inner" else "outer"))
        out = out.sort_index()
        return out

    returns = _join(ret_frames)
    if returns is None:
        raise ValueError("No source produced returns; a Panel needs at least one returns frame.")
    panel = Panel(
        returns=returns,
        prices=_join(price_frames),
        macro=_join(macro_frames),
        events=_join(event_frames),
        meta={"sources": provenance, "assembled_how": how},
    )
    if start or end:
        panel = panel.subset(start=start, end=end)
    return panel
