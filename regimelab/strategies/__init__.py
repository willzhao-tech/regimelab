"""
regimelab.strategies
=====================

The strategy layer turns "an allocation rule" into a pluggable object and runs
any such object through ONE shared, look-ahead-free, volatility-targeted engine.

Design:
  - A `Strategy` implements `weights(mat, i, ctx) -> np.ndarray` returning the
    target weights for the period beginning at column `i` of the returns matrix.
    It sees only data strictly before `i` (no look-ahead).
  - `run(strategy, panel, ...)` is the engine. It rebalances on a schedule,
    asks the strategy for weights, scales the book to a target volatility on a
    trailing window, and returns a daily return series + the realized weights.
  - Strategies register themselves via `@register("name")`, so `get("name")`
    and `available()` expose them; adding a strategy is a new decorated class.

Built-ins reproduce the original research exactly:
  risk_parity (inverse-vol), equal_weight (1/N), fixed (hand weights), trend (TSMOM).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Mapping, Any

import numpy as np
import pandas as pd

from ..panel import Panel, annualization_factor


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, type["Strategy"]] = {}


def register(name: str) -> Callable[[type["Strategy"]], type["Strategy"]]:
    def deco(cls: type["Strategy"]) -> type["Strategy"]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str, **kwargs: Any) -> "Strategy":
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy '{name}'. Available: {available()}")
    return _REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# base interface
# ---------------------------------------------------------------------------
@dataclass
class EngineContext:
    """Read-only context handed to a strategy at each rebalance."""
    vol_win: int
    instruments: list[str]
    rf_annual: float = 0.0


class Strategy(ABC):
    """Abstract allocation rule. Subclasses implement `weights`."""
    name: str = "strategy"

    @abstractmethod
    def weights(self, mat: np.ndarray, i: int, ctx: EngineContext) -> np.ndarray:
        """
        Target weights (sum convention is the strategy's own; the engine handles
        vol-scaling). MUST use only mat[:, :i] (strictly before i) — no look-ahead.
        `mat` is (n_instruments x n_dates) fractional returns.
        """
        ...

    def __repr__(self) -> str:
        return f"<Strategy {self.name}>"


# ---------------------------------------------------------------------------
# shared estimators (trailing, no look-ahead)
# ---------------------------------------------------------------------------
def trailing_vol(series: np.ndarray, i: int, win: int) -> float:
    seg = series[max(0, i - win):i]
    return float(np.std(seg)) if seg.size >= 10 else float("nan")


def inverse_vol_weights(mat: np.ndarray, i: int, win: int) -> np.ndarray:
    n = mat.shape[0]
    iv = np.zeros(n)
    for k in range(n):
        v = trailing_vol(mat[k], i, win)
        iv[k] = (1.0 / v) if (v and v > 0 and np.isfinite(v)) else 0.0
    tot = iv.sum()
    return iv / tot if tot > 0 else np.ones(n) / n


# ---------------------------------------------------------------------------
# built-in strategies
# ---------------------------------------------------------------------------
@register("risk_parity")
class RiskParity(Strategy):
    """Inverse-volatility weights: w_i ∝ 1/σ_i on a trailing window."""
    def weights(self, mat, i, ctx):
        return inverse_vol_weights(mat, i, ctx.vol_win)


@register("equal_weight")
class EqualWeight(Strategy):
    """1/N across all instruments."""
    def weights(self, mat, i, ctx):
        n = mat.shape[0]
        return np.ones(n) / n


@register("fixed")
class Fixed(Strategy):
    """
    Static hand-chosen weights. Pass `weights_map={inst: w}`; the engine column
    order is ctx.instruments, so we project the map onto that order.
    """
    def __init__(self, weights_map: Optional[Mapping[str, float]] = None):
        self.weights_map = dict(weights_map) if weights_map else None

    def weights(self, mat, i, ctx):
        if self.weights_map is None:
            n = mat.shape[0]
            return np.ones(n) / n
        w = np.array([self.weights_map.get(name, 0.0) for name in ctx.instruments], float)
        tot = w.sum()
        return w / tot if tot > 0 else np.ones(len(w)) / len(w)


@register("trend")
class Trend(Strategy):
    """
    Time-series momentum (trend following). Signal = sign of trailing-`lookback`
    cumulative return per instrument; legs inverse-vol weighted; long OR short.
    """
    def __init__(self, lookback: int = 126):
        self.lookback = lookback

    def weights(self, mat, i, ctx):
        n = mat.shape[0]
        sig = np.zeros(n)
        for k in range(n):
            if i >= self.lookback:
                cum = np.prod(1.0 + mat[k, i - self.lookback:i]) - 1.0
                sig[k] = np.sign(cum)
        legs = inverse_vol_weights(mat, i, ctx.vol_win)
        return legs * sig  # signed, risk-weighted


@register("static_mix")
class StaticMix(Fixed):
    """Alias of `fixed` for readability when expressing benchmark allocations."""
    pass


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """Daily returns of a strategy run, plus realized weights and provenance."""
    returns: pd.Series                # daily fractional returns, indexed by date
    weights: pd.DataFrame             # realized target weights per rebalance-day, by instrument
    instruments: list[str]
    strategy: str
    config: dict

    def equity_curve(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()


_REBAL = {"D": 1, "W": 5, "M": 21, "Q": 63}  # approx trading-day cadence


def _is_rebal(i: int, dates: pd.DatetimeIndex, rebal: str, prev_idx: int) -> bool:
    if rebal == "M":
        return dates[i].month != dates[prev_idx].month or dates[i].year != dates[prev_idx].year
    if rebal == "Q":
        return (dates[i].quarter, dates[i].year) != (dates[prev_idx].quarter, dates[prev_idx].year)
    if rebal == "W":
        return dates[i].isocalendar()[1] != dates[prev_idx].isocalendar()[1]
    if rebal == "D":
        return True
    raise ValueError(f"unknown rebal '{rebal}'")


def run(
    strategy: Strategy,
    panel: Panel,
    instruments: Optional[Sequence[str]] = None,
    target_vol: float = 0.10,
    vol_win: int = 63,
    rebal: str = "M",
    rf_annual: float = 0.0,
    freq: str = "D",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> BacktestResult:
    """
    Run `strategy` through the shared engine on `panel`.

    At each rebalance: get target weights (using only past data), then scale the
    weighted book to `target_vol` annualized using trailing realized vol on
    `vol_win`. Returns are then accrued daily until the next rebalance.
    """
    cols = list(instruments) if instruments is not None else panel.instruments
    sub = panel.subset(cols, start=start, end=end)
    mat, cols, idx = sub.to_matrix(cols)               # (n_inst x n_dates)
    n_inst, n_dates = mat.shape
    tdays = annualization_factor(freq)
    ctx = EngineContext(vol_win=vol_win, instruments=cols, rf_annual=rf_annual)

    port = np.empty(n_dates - 1)
    w = None
    scale = 1.0
    prev_idx = 0
    rebal_rows: dict[pd.Timestamp, np.ndarray] = {}

    for i in range(1, n_dates):
        if w is None or _is_rebal(i, idx, rebal, prev_idx):
            w = strategy.weights(mat, i, ctx)
            # scale combined book to target vol on trailing window of the weighted series
            lo = max(0, i - vol_win)
            seg = (w[:, None] * mat[:, lo:i]).sum(axis=0)
            pv = float(np.std(seg)) if seg.size > 10 else float("nan")
            scale = (target_vol / np.sqrt(tdays)) / pv if (pv and pv > 0 and np.isfinite(pv)) else 1.0
            prev_idx = i
            rebal_rows[idx[i]] = w * scale
        port[i - 1] = float((scale * w * mat[:, i]).sum())

    returns = pd.Series(port, index=idx[1:], name=strategy.name)
    weights = pd.DataFrame.from_dict(rebal_rows, orient="index", columns=cols).sort_index()
    config = dict(target_vol=target_vol, vol_win=vol_win, rebal=rebal,
                  rf_annual=rf_annual, instruments=cols, freq=freq)
    return BacktestResult(returns=returns, weights=weights,
                          instruments=cols, strategy=strategy.name, config=config)
