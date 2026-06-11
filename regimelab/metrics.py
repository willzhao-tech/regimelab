"""
regimelab.metrics
=================

Shared performance statistics, computed identically wherever they are needed
(strategy reporting, in-sample evaluation, forward Monte-Carlo). Keeping one
implementation guarantees that every number in the platform is comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .panel import annualization_factor


@dataclass
class Stats:
    cagr: float
    vol: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    total: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def compute_stats(daily, rf_annual: float = 0.0, freq: str = "D") -> Stats:
    """Annualized stats from a daily fractional-return series/array."""
    d = np.asarray(daily, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return Stats(0, 0, 0, 0, 0, 0, 0, 0)
    tdays = annualization_factor(freq)
    cum = float(np.prod(1.0 + d))
    yrs = n / tdays
    cagr = cum ** (1.0 / yrs) - 1.0 if cum > 0 else -1.0
    vol = float(np.std(d)) * np.sqrt(tdays)
    sharpe = (cagr - rf_annual) / vol if vol > 0 else 0.0
    downs = d[d < 0]
    dd_sd = np.sqrt((downs ** 2).sum() / n) * np.sqrt(tdays) if downs.size else 0.0
    sortino = (cagr - rf_annual) / dd_sd if dd_sd > 0 else 0.0
    eq = np.cumprod(1.0 + d)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    return Stats(cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino,
                 max_dd=mdd, calmar=calmar, total=cum - 1.0, n=n)


def yearly_returns(returns: pd.Series) -> pd.Series:
    """Calendar-year compounded returns from a daily return series."""
    grp = (1.0 + returns).groupby(returns.index.year).prod() - 1.0
    grp.index.name = "year"
    return grp
