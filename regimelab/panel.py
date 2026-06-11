"""
regimelab.panel
===============

The `Panel` is the single shared data structure that flows between every layer
of the platform. A data source produces a Panel; strategies consume one; the
evaluation layer runs strategies against one. Keeping this contract small and
stable is what lets the layers compose without coupling.

A Panel holds:
  - `returns`  : DataFrame (index = dates, columns = instruments) of daily
                 fractional returns (0.013 == +1.3%). This is the canonical
                 representation used by the engine.
  - `prices`   : optional DataFrame of levels, same shape, if the source had them.
  - `macro`    : optional DataFrame of macro/regime-conditioning series
                 (CPI, unemployment, yield-curve spread, VIX level, etc.).
                 These are NOT tradable; they feed the regime layer.
  - `events`   : optional DataFrame of catalyst/event data, indexed by date,
                 with columns describing scheduled releases / surprises.
  - `meta`     : free-form dict (source provenance, frequency, currency notes).

Construction is deliberately permissive: a source can supply any subset, and the
Panel exposes helpers (`common_dates`, `subset`, `align`) that the layers above
rely on. Nothing here computes strategy logic — that lives in `strategies/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Mapping, Any

import numpy as np
import pandas as pd


@dataclass
class Panel:
    """A cross-asset panel of returns plus optional prices, macro, and events."""

    returns: pd.DataFrame
    prices: Optional[pd.DataFrame] = None
    macro: Optional[pd.DataFrame] = None
    events: Optional[pd.DataFrame] = None
    meta: dict = field(default_factory=dict)

    # -- validation -------------------------------------------------------
    def __post_init__(self) -> None:
        if not isinstance(self.returns, pd.DataFrame):
            raise TypeError("Panel.returns must be a pandas DataFrame")
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            # be forgiving: coerce a parseable index to datetime
            self.returns.index = pd.to_datetime(self.returns.index)
        self.returns = self.returns.sort_index()

    # -- basic descriptors ------------------------------------------------
    @property
    def instruments(self) -> list[str]:
        return list(self.returns.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.returns.index

    def __repr__(self) -> str:
        n_macro = 0 if self.macro is None else self.macro.shape[1]
        span = (
            f"{self.dates[0].date()}..{self.dates[-1].date()}"
            if len(self.dates) else "empty"
        )
        return (
            f"Panel(instruments={len(self.instruments)}, rows={len(self.dates)}, "
            f"span={span}, macro_series={n_macro}, "
            f"events={'yes' if self.events is not None else 'no'})"
        )

    # -- selection / alignment -------------------------------------------
    def common_dates(self, instruments: Optional[Sequence[str]] = None) -> pd.DatetimeIndex:
        """Dates on which *all* requested instruments have a (non-NaN) return."""
        cols = list(instruments) if instruments is not None else self.instruments
        sub = self.returns[cols]
        return sub.dropna(how="any").index

    def subset(
        self,
        instruments: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> "Panel":
        """Return a new Panel restricted to instruments and/or a date window."""
        cols = list(instruments) if instruments is not None else self.instruments
        r = self.returns[cols]
        if start is not None:
            r = r.loc[r.index >= pd.to_datetime(start)]
        if end is not None:
            r = r.loc[r.index <= pd.to_datetime(end)]
        idx = r.index

        def _clip(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None:
                return None
            return df.loc[df.index.isin(idx)] if len(idx) else df.iloc[0:0]

        return Panel(
            returns=r,
            prices=None if self.prices is None else self.prices[cols].loc[self.prices.index.isin(idx)],
            macro=_clip(self.macro),
            events=_clip(self.events),
            meta={**self.meta, "subset_of": self.meta.get("name", "panel")},
        )

    def to_matrix(self, instruments: Optional[Sequence[str]] = None) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
        """
        Dense (n_instruments x n_dates) ndarray of returns on the common dates,
        plus the column order and the date index. This is the fast form the
        engine consumes in Monte-Carlo loops.
        """
        cols = list(instruments) if instruments is not None else self.instruments
        idx = self.common_dates(cols)
        mat = self.returns.loc[idx, cols].to_numpy().T  # shape (n_inst, n_dates)
        return mat, cols, idx

    # -- construction helpers --------------------------------------------
    @classmethod
    def from_price_frame(cls, prices: pd.DataFrame, **kw: Any) -> "Panel":
        """Build a Panel from a price-level DataFrame (returns computed as pct change)."""
        prices = prices.sort_index()
        returns = prices.pct_change().iloc[1:]
        return cls(returns=returns, prices=prices, **kw)

    @classmethod
    def from_legacy_json(cls, path: str, key: str = "series") -> "Panel":
        """
        Load the legacy panel.json format used by the original research scripts:
        `{ "series": { instrument: { "YYYY-MM-DD": pct_return_in_percent } } }`
        where values are returns in *percent* (1.3 == +1.3%). Converts to the
        fractional convention used throughout regimelab.
        """
        import json
        with open(path) as fh:
            blob = json.load(fh)
        series = blob[key]
        frames = {
            inst: pd.Series({pd.to_datetime(d): v / 100.0 for d, v in obs.items()})
            for inst, obs in series.items()
        }
        returns = pd.DataFrame(frames).sort_index()
        return cls(returns=returns, meta={"name": "legacy_panel", "source": path})


def annualization_factor(freq: str = "D") -> int:
    """Periods per year for common frequencies."""
    return {"D": 252, "W": 52, "M": 12, "Q": 4, "Y": 1}.get(freq.upper(), 252)
