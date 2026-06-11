"""
regimelab.regime
================

The regime layer is the platform's analytical core. It does two things:

  1. IDENTIFY regimes from data. Given a Panel's macro series (and, where
     available, catalyst/event signals), label each date with a regime. This is
     the novel, extensible piece: rules map observable macro state -> regime,
     and new identifiers (HMM, k-means, supervised) plug in behind one interface.

  2. SIMULATE forward. A `RegimeModel` holds, per regime, asset drifts, vol
     multipliers, an inflation flag, and fat-tail parameters, plus a Markov
     transition matrix and an inflation-conditional stock/bond correlation.
     `simulate()` draws persistent, fat-tailed forward paths — the engine behind
     the possibility-weighted forward evaluation.

Both halves are configured, not hard-coded: regimes, priors, drifts and
correlations live in a `RegimeModel` you can edit or fit, so the platform
supports new regime menus and new identification schemes without core changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Mapping, Sequence, Callable, Any

import numpy as np
import pandas as pd

from ..panel import Panel


# ===========================================================================
# PART 1 — regime definitions + the forward model
# ===========================================================================
@dataclass
class Regime:
    """One macroeconomic regime's parameters for simulation."""
    name: str
    drift: dict[str, float]          # annualized drift per instrument
    eq_vol_mult: float = 1.0         # equity vol scaler
    bond_vol_mult: float = 1.0       # bond vol scaler
    inflationary: bool = False       # flips the stock/bond correlation sign
    prior: float = 0.0               # stationary probability (normalized later)


@dataclass
class RegimeModel:
    """
    A full forward environment: a regime menu, a Markov transition matrix whose
    stationary distribution matches the priors, an inflation-conditional
    stock/bond correlation, an equity cross-correlation, and per-instrument
    fat-tail (Student-t) degrees of freedom.
    """
    regimes: list[Regime]
    instruments: list[str]
    base_vol: dict[str, float]                       # per-instrument daily vol (from data)
    corr_stock_bond_infl: float = 0.55
    corr_stock_bond_noinfl: float = -0.35
    corr_equities: float = 0.72                      # e.g. US-China equity co-move
    tail_df: dict[str, float] = field(default_factory=dict)   # Student-t dof per instrument
    p_stay: float = 0.72                             # Markov self-transition
    equity_names: tuple[str, ...] = ("NQ", "A50")
    bond_name: str = "US10Y"
    seed: Optional[int] = None

    # -- derived ----------------------------------------------------------
    def __post_init__(self):
        tot = sum(max(r.prior, 0.0) for r in self.regimes)
        if tot > 0:
            for r in self.regimes:
                r.prior = max(r.prior, 0.0) / tot
        self._names = [r.name for r in self.regimes]
        self._prior = np.array([r.prior for r in self.regimes])
        self._T = self._build_transition()
        self._rng = np.random.default_rng(self.seed)

    def _build_transition(self) -> np.ndarray:
        n = len(self.regimes)
        T = np.zeros((n, n))
        for i in range(n):
            denom = sum(self.regimes[j].prior for j in range(n) if j != i)
            for j in range(n):
                if i == j:
                    T[i, j] = self.p_stay
                else:
                    T[i, j] = (1 - self.p_stay) * (self.regimes[j].prior / denom if denom > 0 else 0)
        return T

    # -- simulation -------------------------------------------------------
    def _t_std(self, shape, df: float) -> np.ndarray:
        """Standardized Student-t (unit variance) draws."""
        z = self._rng.standard_normal(shape)
        chi = self._rng.chisquare(df, shape)
        t = z / np.sqrt(chi / df)
        return t / np.sqrt(df / (df - 2)) if df > 2 else t

    def _gen_year(self, regime: Regime, periods: int = 252) -> np.ndarray:
        """One year of daily returns (n_instruments x periods) for a regime."""
        cse = self.corr_stock_bond_infl if regime.inflationary else self.corr_stock_bond_noinfl
        out = np.empty((len(self.instruments), periods))
        # factor draws
        eq0 = self.equity_names[0]
        zc = self._t_std(periods, self.tail_df.get(eq0, 5))
        for k, inst in enumerate(self.instruments):
            sig = self.base_vol[inst]
            if inst in self.equity_names:
                sig *= regime.eq_vol_mult
            elif inst == self.bond_name:
                sig *= regime.bond_vol_mult
            mu = regime.drift.get(inst, 0.0) / periods
            if inst == eq0:
                shock = zc
            elif inst in self.equity_names:
                za = self._t_std(periods, self.tail_df.get(inst, 3))
                rho = self.corr_equities
                shock = rho * zc + np.sqrt(max(0, 1 - rho ** 2)) * za
            elif inst == self.bond_name:
                zb = self._t_std(periods, self.tail_df.get(inst, 6))
                shock = cse * zc + np.sqrt(max(0, 1 - cse ** 2)) * zb
            else:  # independent (e.g. gold) with its own fat tail
                shock = self._t_std(periods, self.tail_df.get(inst, 4))
            out[k] = mu + sig * shock
        return out

    def draw_path(self, horizon_years: int) -> tuple[np.ndarray, list[str]]:
        """A multi-year daily return matrix following the Markov regime chain."""
        cur = self._rng.choice(len(self.regimes), p=self._prior)
        seq = [cur]
        for _ in range(horizon_years - 1):
            cur = self._rng.choice(len(self.regimes), p=self._T[cur])
            seq.append(cur)
        blocks = [self._gen_year(self.regimes[r]) for r in seq]
        mat = np.concatenate(blocks, axis=1)
        return mat, [self._names[r] for r in seq]

    def simulate(self, horizon_years: int = 5) -> "SimPath":
        mat, regime_seq = self.draw_path(horizon_years)
        return SimPath(matrix=mat, instruments=list(self.instruments), regime_sequence=regime_seq)


@dataclass
class SimPath:
    """One simulated forward path: a return matrix + the regimes it traversed."""
    matrix: np.ndarray               # (n_instruments x n_days)
    instruments: list[str]
    regime_sequence: list[str]


# ---- a ready-to-use, history-anchored default model -----------------------
def default_model(instruments: Sequence[str], base_vol: Mapping[str, float],
                  seed: Optional[int] = None) -> RegimeModel:
    """
    The history-anchored 7-regime model from the research, ready to simulate.
    Priors reflect long-run US base rates; correlations are inflation-conditional;
    A50/gold carry fatter tails. Pass per-instrument `base_vol` measured from data.
    """
    regimes = [
        Regime("Goldilocks",   {"NQ":0.18,"A50":0.14,"US10Y":0.04,"XAU":0.06}, 0.8, 0.8, False, 0.34),
        Regime("Muddle",       {"NQ":0.06,"A50":0.04,"US10Y":0.02,"XAU":0.05}, 1.0, 1.0, False, 0.27),
        Regime("Gold-drawdown",{"NQ":0.08,"A50":0.05,"US10Y":-0.02,"XAU":-0.25},1.0,1.0, False, 0.06),
        Regime("Deflationary", {"NQ":-0.32,"A50":-0.28,"US10Y":0.12,"XAU":0.04},1.6,1.3, False, 0.07),
        Regime("Stagflation",  {"NQ":-0.18,"A50":-0.14,"US10Y":-0.10,"XAU":0.20},1.3,1.5, True,  0.10),
        Regime("Risk-off",     {"NQ":-0.28,"A50":-0.34,"US10Y":0.06,"XAU":0.08},1.7,1.2, False, 0.08),
        Regime("Crash",        {"NQ":-0.42,"A50":-0.45,"US10Y":-0.06,"XAU":-0.12},1.8,1.4,True,  0.045),
    ]
    return RegimeModel(
        regimes=regimes,
        instruments=list(instruments),
        base_vol=dict(base_vol),
        tail_df={"NQ":5, "A50":3, "US10Y":6, "XAU":4},
        seed=seed,
    )


def base_vol_from_panel(panel: Panel, instruments: Sequence[str]) -> dict[str, float]:
    """Per-instrument empirical daily vol, to anchor a RegimeModel to real data."""
    return {inst: float(np.std(panel.returns[inst].dropna())) for inst in instruments}


# ===========================================================================
# PART 2 — regime IDENTIFICATION from macro + catalyst signals
# ===========================================================================
class RegimeIdentifier(ABC):
    """Map a Panel's macro/event state to a per-date regime label."""
    @abstractmethod
    def label(self, panel: Panel) -> pd.Series:
        """Return a Series (index = dates) of regime-name strings."""
        ...


class RuleBasedIdentifier(RegimeIdentifier):
    """
    Transparent rule-based regime labeller driven by macro series in Panel.macro.
    Expected (optional) macro columns, any subset:
        'recession'   : NBER flag (FRED USREC), 1 during recession
        'cpi_yoy'     : YoY inflation (%)
        'curve'       : 10y-2y or 10y-3m spread
        'vix'         : VIX level
        'equity_dd'   : trailing equity drawdown (fraction, negative)
    Rules are applied in priority order; the first match wins. This is a
    deliberately simple, auditable baseline — subclass or swap for an HMM /
    clustering / supervised identifier without touching the rest of the platform.
    """
    def __init__(self, inflation_threshold: float = 4.0, vix_stress: float = 28.0,
                 crash_dd: float = -0.20):
        self.inflation_threshold = inflation_threshold
        self.vix_stress = vix_stress
        self.crash_dd = crash_dd

    def label(self, panel: Panel) -> pd.Series:
        if panel.macro is None or panel.macro.empty:
            raise ValueError("RuleBasedIdentifier needs Panel.macro series to label regimes.")
        m = panel.macro
        idx = m.index
        out = pd.Series(index=idx, dtype=object)

        def col(name):
            return m[name] if name in m.columns else pd.Series(np.nan, index=idx)

        rec = col("recession"); cpi = col("cpi_yoy"); curve = col("curve")
        vix = col("vix"); dd = col("equity_dd")

        for t in idx:
            inflationary = (pd.notna(cpi[t]) and cpi[t] >= self.inflation_threshold)
            in_rec = (pd.notna(rec[t]) and rec[t] >= 0.5)
            stressed = (pd.notna(vix[t]) and vix[t] >= self.vix_stress)
            deep_dd = (pd.notna(dd[t]) and dd[t] <= self.crash_dd)

            if deep_dd and stressed:
                label = "Crash"
            elif in_rec and inflationary:
                label = "Stagflation"
            elif in_rec:
                label = "Deflationary"
            elif stressed:
                label = "Risk-off"
            elif inflationary:
                label = "Stagflation"
            elif pd.notna(vix[t]) and vix[t] < 15:
                label = "Goldilocks"
            else:
                label = "Muddle"
            out[t] = label
        return out

    def regime_frequencies(self, panel: Panel) -> pd.Series:
        """Empirical share of days in each regime, per the rules — a data-driven prior."""
        lab = self.label(panel)
        return lab.value_counts(normalize=True).sort_values(ascending=False)


class RichRuleBasedIdentifier(RuleBasedIdentifier):
    """
    A finer identifier that uses the yield curve in addition to recession / CPI /
    VIX / drawdown, so the regime menu carves the long 'Muddle' middle into more
    cycle-specific states. New regimes beyond the base set:
        'LateCycle' : curve inverted (10y-2y < 0) but not yet stressed/recession
        'Reflation' : steep curve (>= `steep`) and calm VIX (early-cycle recovery)
    Same auditable, first-match-wins design — added macro column expected:
        'curve' : 10y minus 2y yield (negative = inverted)
    """
    def __init__(self, inflation_threshold: float = 4.0, vix_stress: float = 28.0,
                 crash_dd: float = -0.20, calm_vix: float = 15.0, steep: float = 1.5):
        super().__init__(inflation_threshold, vix_stress, crash_dd)
        self.calm_vix = calm_vix
        self.steep = steep

    def label(self, panel: Panel) -> pd.Series:
        if panel.macro is None or panel.macro.empty:
            raise ValueError("RichRuleBasedIdentifier needs Panel.macro series to label regimes.")
        m = panel.macro
        idx = m.index
        out = pd.Series(index=idx, dtype=object)

        def col(name):
            return m[name] if name in m.columns else pd.Series(np.nan, index=idx)

        rec = col("recession"); cpi = col("cpi_yoy"); vix = col("vix")
        dd = col("equity_dd"); curve = col("curve")

        for t in idx:
            inflationary = (pd.notna(cpi[t]) and cpi[t] >= self.inflation_threshold)
            in_rec = (pd.notna(rec[t]) and rec[t] >= 0.5)
            stressed = (pd.notna(vix[t]) and vix[t] >= self.vix_stress)
            deep_dd = (pd.notna(dd[t]) and dd[t] <= self.crash_dd)
            inverted = (pd.notna(curve[t]) and curve[t] < 0)
            steep = (pd.notna(curve[t]) and curve[t] >= self.steep)
            calm = (pd.notna(vix[t]) and vix[t] < self.calm_vix)

            if deep_dd and stressed:
                label = "Crash"
            elif in_rec and inflationary:
                label = "Stagflation"
            elif in_rec:
                label = "Deflationary"
            elif stressed:
                label = "Risk-off"
            elif inflationary:
                label = "Stagflation"
            elif inverted:
                label = "LateCycle"
            elif steep and calm:
                label = "Reflation"
            elif calm:
                label = "Goldilocks"
            else:
                label = "Muddle"
            out[t] = label
        return out


# ---- catalyst / event helpers --------------------------------------------
def attach_event_flags(panel: Panel, event_dates: Mapping[str, Sequence[str]]) -> Panel:
    """
    Attach scheduled-catalyst flags to a Panel as an `events` frame, one boolean
    column per event family (e.g. {'fomc': [...dates...], 'cpi': [...]}). These
    feed event-conditional analysis and can sharpen regime identification.
    """
    idx = panel.returns.index
    cols = {}
    for fam, dates in event_dates.items():
        dts = pd.to_datetime(list(dates))
        flag = pd.Series(0, index=idx, name=fam)
        flag.loc[flag.index.isin(dts)] = 1
        cols[fam] = flag
    events = pd.DataFrame(cols, index=idx)
    return Panel(returns=panel.returns, prices=panel.prices, macro=panel.macro,
                 events=events, meta={**panel.meta, "events_attached": list(event_dates)})
