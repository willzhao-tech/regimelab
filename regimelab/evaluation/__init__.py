"""
regimelab.evaluation
=====================

The evaluation layer is what makes the platform produce *research* rather than
just backtests. It provides:

  - `in_sample(...)`  : run a strategy on realized history, return Stats.
  - `forward(...)`    : run a strategy across many simulated regime paths,
                        return the distribution of outcomes (the possibility-
                        weighted lens).
  - `compare(...)`    : run a set of strategies through both lenses and return a
                        tidy table — the in-sample-vs-forward ranking.
  - out-of-sample protocol (`oos_split`, `walk_forward`): fit/evaluate on
    genuinely held-out data so results are not in-sample artifacts.
  - inference: `deflated_sharpe` (Bailey & López de Prado) to discount the best
    of many tried strategies, and `sharpe_pvalue` for a single Sharpe.
  - `inversion_study(...)`: the platform's headline novel experiment —
    quantifying how the in-sample→forward ranking inversion grows with how
    REGIME-CONCENTRATED the in-sample window was.

Everything here sits on top of the strategy engine and the regime model; it adds
no new market logic, only measurement and inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Mapping, Any

import numpy as np
import pandas as pd

from ..panel import Panel
from ..metrics import Stats, compute_stats
from .. import strategies as strat_layer
from .. import regime as regime_layer


# ===========================================================================
# the two lenses
# ===========================================================================
def in_sample(
    strategy,
    panel: Panel,
    instruments: Optional[Sequence[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    **engine_kw: Any,
) -> Stats:
    """Run a strategy on realized history; return annualized Stats."""
    if isinstance(strategy, str):
        strategy = strat_layer.get(strategy)
    res = strat_layer.run(strategy, panel, instruments=instruments,
                          start=start, end=end, **engine_kw)
    return compute_stats(res.returns.values,
                         rf_annual=engine_kw.get("rf_annual", 0.0))


@dataclass
class ForwardResult:
    """Distribution of outcomes for a strategy across simulated regime paths."""
    cagr: np.ndarray
    sharpe: np.ndarray
    max_dd: np.ndarray
    strategy: str
    n_paths: int
    horizon: int

    def summary(self) -> dict:
        def pc(a, q): return float(np.percentile(a, q))
        return {
            "E_cagr": float(np.mean(self.cagr)),
            "med_cagr": float(np.median(self.cagr)),
            "E_sharpe": float(np.mean(self.sharpe)),
            "cagr_5pct": pc(self.cagr, 5),
            "maxdd_med": float(np.median(self.max_dd)),
            "maxdd_5pct": pc(self.max_dd, 5),
            "p_loss": float(np.mean(self.cagr < 0)),
        }


def forward(
    strategy,
    model: "regime_layer.RegimeModel",
    instruments: Optional[Sequence[str]] = None,
    n_paths: int = 1000,
    horizon: int = 5,
    target_vol: float = 0.10,
    vol_win: int = 63,
    rebal: str = "M",
) -> ForwardResult:
    """
    Run a strategy across `n_paths` simulated forward paths from `model`,
    each `horizon` years, through the same engine used in-sample.
    """
    if isinstance(strategy, str):
        strategy = strat_layer.get(strategy)
    cols = list(instruments) if instruments is not None else list(model.instruments)
    cagrs, sharpes, mdds = [], [], []
    for _ in range(n_paths):
        sim = model.simulate(horizon_years=horizon)
        dates = pd.bdate_range("2030-01-01", periods=sim.matrix.shape[1])
        p = Panel(returns=pd.DataFrame(sim.matrix.T, index=dates, columns=cols))
        res = strat_layer.run(strategy, p, instruments=cols,
                              target_vol=target_vol, vol_win=vol_win, rebal=rebal)
        s = compute_stats(res.returns.values)
        cagrs.append(s.cagr); sharpes.append(s.sharpe); mdds.append(s.max_dd)
    return ForwardResult(np.array(cagrs), np.array(sharpes), np.array(mdds),
                         getattr(strategy, "name", "strategy"), n_paths, horizon)


def compare(
    strategy_specs: Sequence[tuple[str, dict]],
    panel: Panel,
    model: "regime_layer.RegimeModel",
    instruments: Sequence[str],
    n_paths: int = 800,
    horizon: int = 5,
    is_start: Optional[str] = None,
    **engine_kw: Any,
) -> pd.DataFrame:
    """
    Run each (strategy_name, kwargs) through both lenses; return a tidy table
    with in-sample and forward metrics side by side, plus both rankings.
    """
    rows = []
    for name, kw in strategy_specs:
        label = name + (f"({kw['lookback']})" if name == "trend" and "lookback" in kw else "")
        s_is = in_sample(strat_layer.get(name, **kw), panel, instruments=instruments,
                         start=is_start, **engine_kw)
        fr = forward(strat_layer.get(name, **kw), model, instruments=instruments,
                     n_paths=n_paths, horizon=horizon, **{k: v for k, v in engine_kw.items()
                                                          if k in {"target_vol", "vol_win", "rebal"}})
        fs = fr.summary()
        rows.append({
            "strategy": label,
            "is_cagr": s_is.cagr, "is_sharpe": s_is.sharpe, "is_maxdd": s_is.max_dd,
            "fwd_cagr": fs["E_cagr"], "fwd_sharpe": fs["E_sharpe"],
            "fwd_cagr_5pct": fs["cagr_5pct"], "fwd_ploss": fs["p_loss"],
        })
    df = pd.DataFrame(rows).set_index("strategy")
    df["is_rank"] = df["is_sharpe"].rank(ascending=False).astype(int)
    df["fwd_rank"] = df["fwd_sharpe"].rank(ascending=False).astype(int)
    return df


# ===========================================================================
# out-of-sample protocol
# ===========================================================================
def oos_split(panel: Panel, split_date: str) -> tuple[Panel, Panel]:
    """Split a panel into in-sample (<= split) and out-of-sample (> split)."""
    is_p = panel.subset(end=split_date)
    # OOS begins the day after split
    oos_dates = panel.returns.index[panel.returns.index > pd.to_datetime(split_date)]
    oos_p = panel.subset(start=str(oos_dates[0].date())) if len(oos_dates) else panel.subset(start="2999-01-01")
    return is_p, oos_p


def walk_forward(
    strategy,
    panel: Panel,
    instruments: Sequence[str],
    n_folds: int = 4,
    **engine_kw: Any,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward: split history into `n_folds` sequential test
    blocks; for each, the strategy is evaluated on the test block (the engine
    only uses trailing data within it, so this is genuinely out-of-sample for
    any rule whose parameters were chosen on earlier data). Returns per-fold Stats.
    """
    if isinstance(strategy, str):
        strategy = strat_layer.get(strategy)
    idx = panel.common_dates(instruments)
    bounds = np.linspace(0, len(idx), n_folds + 1).astype(int)
    rows = []
    for f in range(n_folds):
        lo, hi = bounds[f], bounds[f + 1]
        if hi - lo < 60:
            continue
        seg_start, seg_end = str(idx[lo].date()), str(idx[hi - 1].date())
        s = in_sample(strategy, panel, instruments=instruments,
                      start=seg_start, end=seg_end, **engine_kw)
        rows.append({"fold": f + 1, "start": seg_start, "end": seg_end,
                     "cagr": s.cagr, "sharpe": s.sharpe, "max_dd": s.max_dd, "n": s.n})
    return pd.DataFrame(rows).set_index("fold")


# ===========================================================================
# inference — discounting luck and multiple testing
# ===========================================================================
def sharpe_pvalue(sharpe: float, n_periods: int, freq_per_year: int = 252) -> float:
    """
    Two-sided p-value for a single annualized Sharpe under H0: true Sharpe = 0,
    using the large-sample SR standard error ~ sqrt(1/n) on the per-period SR.
    """
    from math import erfc, sqrt
    sr_period = sharpe / np.sqrt(freq_per_year)
    se = np.sqrt(1.0 / max(n_periods, 2))
    z = abs(sr_period) / se
    return float(erfc(z / np.sqrt(2)))  # two-sided normal tail


def deflated_sharpe(
    observed_sharpe: float,
    n_periods: int,
    n_trials: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    freq_per_year: int = 252,
    sharpe_variance_across_trials: Optional[float] = None,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014): the probability that
    the observed (annualized) Sharpe exceeds what you'd expect to find as the
    MAXIMUM across `n_trials` independent strategy configurations under a true
    Sharpe of zero. A high DSR (-> 1) means the result survives multiple testing;
    a low DSR means it is plausibly the best of many lucky tries.

    Returns DSR in [0, 1]. Per-period quantities are used internally.
    """
    from math import sqrt, log, erf, erfc
    sr = observed_sharpe / np.sqrt(freq_per_year)        # per-period SR

    # expected maximum SR across N trials under H0 (variance of SR estimates across trials)
    var_sr = sharpe_variance_across_trials
    if var_sr is None:
        var_sr = 1.0 / n_periods                          # default: sampling variance
    sigma = np.sqrt(var_sr)
    # Euler-Mascheroni + extreme-value approximation for E[max of N standard normals]
    EM = 0.5772156649
    N = max(n_trials, 1)
    if N > 1:
        z1 = _norm_ppf(1 - 1.0 / N)
        z2 = _norm_ppf(1 - 1.0 / (N * np.e))
        e_max = (1 - EM) * z1 + EM * z2
    else:
        e_max = 0.0
    sr0 = sigma * e_max                                   # the "expected best by luck" hurdle

    # DSR: prob that estimated SR exceeds sr0, accounting for non-normality (PSR form)
    num = (sr - sr0) * np.sqrt(n_periods - 1)
    den = np.sqrt(1 - skew * sr + (kurtosis - 1) / 4.0 * sr ** 2)
    if den <= 0:
        return float("nan")
    z = num / den
    return float(0.5 * (1 + erf(z / np.sqrt(2))))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ===========================================================================
# THE HEADLINE NOVEL EXPERIMENT: regime concentration -> ranking inversion
# ===========================================================================
@dataclass
class InversionResult:
    """Output of the inversion study: per-trial concentration vs ranking divergence."""
    table: pd.DataFrame              # one row per simulated in-sample window
    correlation: float               # corr(concentration, inversion)
    slope: float                     # OLS slope of inversion on concentration

    def __repr__(self):
        return (f"InversionResult(corr={self.correlation:+.3f}, slope={self.slope:+.3f}, "
                f"n_trials={len(self.table)})")


def _regime_concentration(regime_seq: Sequence[str]) -> float:
    """
    Herfindahl-style concentration of a regime sequence in [0,1]: 1 = the window
    was a single regime; ->1/k = perfectly balanced across k regimes. This is the
    independent variable of the inversion study.
    """
    s = pd.Series(list(regime_seq)).value_counts(normalize=True)
    return float((s ** 2).sum())


def _ranking_inversion(is_ranks: pd.Series, fwd_ranks: pd.Series) -> float:
    """
    Inversion magnitude in [0,1]: 1 - normalized Spearman, so 0 = identical
    rankings, 1 = perfectly reversed. Measures how badly the in-sample ranking
    misleads about the forward ranking.
    """
    n = len(is_ranks)
    if n < 2:
        return 0.0
    rho = np.corrcoef(is_ranks.values, fwd_ranks.values)[0, 1]
    return float((1 - rho) / 2)  # rho in [-1,1] -> inversion in [0,1]


def inversion_study(
    panel: Panel,
    model: "regime_layer.RegimeModel",
    strategy_specs: Optional[Sequence[tuple[str, dict]]] = None,
    instruments: Optional[Sequence[str]] = None,
    n_trials: int = 40,
    is_horizon: int = 7,
    fwd_paths: int = 200,
    fwd_horizon: int = 5,
    seed: int = 0,
    **engine_kw: Any,
) -> InversionResult:
    """
    THE platform's headline finding, as a reusable experiment.

    Hypothesis: the more REGIME-CONCENTRATED an in-sample window is, the more its
    strategy ranking (by Sharpe) inverts relative to the regime-balanced forward
    ranking. We test it by simulating many synthetic in-sample windows of varying
    regime composition, ranking strategies within each, comparing to the forward
    ranking, and regressing inversion magnitude on concentration.

    Returns the per-trial table plus the correlation and OLS slope. A positive,
    significant slope is the contribution: it turns the one-off observation
    "the 2019-2026 ranking inverted" into a quantified, testable law.
    """
    if strategy_specs is None:
        strategy_specs = [
            ("risk_parity", {}), ("equal_weight", {}),
            ("fixed", {"weights_map": {"NQ": 0.4, "A50": 0.2, "US10Y": 0.2, "XAU": 0.2}}),
            ("trend", {"lookback": 126}),
            ("fixed", {"weights_map": {"NQ": 0.6, "US10Y": 0.4}}),  # 60/40
            ("fixed", {"weights_map": {"NQ": 1.0}}),                 # all-equity
        ]
    cols = list(instruments) if instruments is not None else list(model.instruments)

    # one stable forward ranking (the "truth": regime-balanced, possibility-weighted)
    fwd_sharpe = {}
    for name, kw in strategy_specs:
        label = name + ("_" + "_".join(f"{k}{v}" for k, v in kw.items()) if kw else "")
        fr = forward(strat_layer.get(name, **kw), model, instruments=cols,
                     n_paths=fwd_paths, horizon=fwd_horizon,
                     **{k: v for k, v in engine_kw.items() if k in {"target_vol", "vol_win", "rebal"}})
        fwd_sharpe[label] = float(np.mean(fr.sharpe))
    fwd_rank = pd.Series(fwd_sharpe).rank(ascending=False)

    # many synthetic in-sample windows of varying regime concentration
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_trials):
        # draw an in-sample window from the model (its own seed advances per call)
        model._rng = np.random.default_rng(seed + 1000 + t)
        sim = model.simulate(horizon_years=is_horizon)
        conc = _regime_concentration(sim.regime_sequence)
        dates = pd.bdate_range("2000-01-01", periods=sim.matrix.shape[1])
        p_is = Panel(returns=pd.DataFrame(sim.matrix.T, index=dates, columns=cols))

        is_sharpe = {}
        for name, kw in strategy_specs:
            label = name + ("_" + "_".join(f"{k}{v}" for k, v in kw.items()) if kw else "")
            s = in_sample(strat_layer.get(name, **kw), p_is, instruments=cols,
                          **{k: v for k, v in engine_kw.items() if k in {"target_vol", "vol_win", "rebal"}})
            is_sharpe[label] = s.sharpe
        is_rank = pd.Series(is_sharpe).rank(ascending=False)
        common = is_rank.index.intersection(fwd_rank.index)
        inv = _ranking_inversion(is_rank[common], fwd_rank[common])
        rows.append({"trial": t, "concentration": conc, "inversion": inv,
                     "dominant_regime": pd.Series(sim.regime_sequence).mode()[0]})

    table = pd.DataFrame(rows)
    corr = float(table["concentration"].corr(table["inversion"]))
    # OLS slope of inversion on concentration
    x = table["concentration"].values; y = table["inversion"].values
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 and np.std(x) > 0 else float("nan")
    return InversionResult(table=table, correlation=corr, slope=slope)


# ---------------------------------------------------------------------------
# the headline experiment on REAL data, end to end (no simulated windows)
# ---------------------------------------------------------------------------
def inversion_study_real(
    panel: Panel,
    instruments: Sequence[str],
    strategy_specs: Optional[Sequence[tuple[str, dict]]] = None,
    window_years: float = 2.0,
    step_months: int = 2,
    min_days: int = 250,
    identifier: "regime_layer.RegimeIdentifier | None" = None,
    reference: str = "full_sample",
    model: "regime_layer.RegimeModel | None" = None,
    fwd_paths: int = 200,
    fwd_horizon: int = 5,
    **engine_kw: Any,
) -> InversionResult:
    """
    Real-data version of `inversion_study`. Instead of simulating in-sample
    windows from the regime model, it slides REAL rolling windows over the
    realized panel, computes each window's regime concentration from REAL regime
    labels (an identifier on Panel.macro, e.g. VIX), ranks the strategies on the
    REAL returns in the window, and measures how that ranking inverts vs a
    reference ranking. Regresses inversion magnitude on real concentration.

    `reference`:
      - "full_sample" (default): the strategies' ranking over the ENTIRE realized
        history — the broadest available real regime mix, the model-free proxy for
        the "regime-balanced" ranking. This makes the experiment real end to end.
      - "forward": the possibility-weighted model ranking (needs `model=`), i.e.
        the same reference the simulated `inversion_study` uses.
    """
    cols = list(instruments)
    if strategy_specs is None:
        strategy_specs = [
            ("risk_parity", {}), ("equal_weight", {}),
            ("fixed", {"weights_map": {"NQ": 0.4, "A50": 0.2, "US10Y": 0.2, "XAU": 0.2}}),
            ("trend", {"lookback": 126}),
            ("fixed", {"weights_map": {"NQ": 0.6, "US10Y": 0.4}}),
            ("fixed", {"weights_map": {"NQ": 1.0}}),
        ]

    def _label(name, kw):
        return name + ("_" + "_".join(f"{k}{v}" for k, v in kw.items()) if kw else "")

    eng = {k: v for k, v in engine_kw.items() if k in {"target_vol", "vol_win", "rebal", "rf_annual"}}

    # reference ("balanced") ranking
    if reference == "forward":
        if model is None:
            raise ValueError("reference='forward' requires a fitted model=")
        ref = {}
        for name, kw in strategy_specs:
            fr = forward(strat_layer.get(name, **kw), model, instruments=cols,
                         n_paths=fwd_paths, horizon=fwd_horizon, **eng)
            ref[_label(name, kw)] = float(np.mean(fr.sharpe))
        ref_rank = pd.Series(ref).rank(ascending=False)
    else:  # full-sample real ranking — model-free
        ref = {}
        for name, kw in strategy_specs:
            ref[_label(name, kw)] = in_sample(strat_layer.get(name, **kw), panel,
                                              instruments=cols, **eng).sharpe
        ref_rank = pd.Series(ref).rank(ascending=False)

    # real regime labels over the panel
    if identifier is None:
        identifier = regime_layer.RuleBasedIdentifier()
    labels = identifier.label(panel).dropna()

    idx = panel.common_dates(cols)
    if len(idx) == 0:
        raise ValueError("no common dates for the requested instruments")
    end0 = idx.max()
    wlen = pd.DateOffset(months=int(round(window_years * 12)))
    step = pd.DateOffset(months=step_months)

    rows = []
    w0 = idx.min()
    while w0 + wlen <= end0 + pd.Timedelta(days=1):
        w1 = w0 + wlen
        win_idx = idx[(idx >= w0) & (idx < w1)]
        if len(win_idx) >= min_days:
            wl = labels.loc[(labels.index >= w0) & (labels.index < w1)]
            if len(wl):
                conc = _regime_concentration(wl.values)
                is_sharpe = {}
                for name, kw in strategy_specs:
                    is_sharpe[_label(name, kw)] = in_sample(
                        strat_layer.get(name, **kw), panel, instruments=cols,
                        start=str(w0.date()), end=str(w1.date()), **eng).sharpe
                is_rank = pd.Series(is_sharpe).rank(ascending=False)
                common = is_rank.index.intersection(ref_rank.index)
                rows.append({
                    "start": str(w0.date()), "end": str(w1.date()),
                    "n": len(win_idx), "concentration": conc,
                    "inversion": _ranking_inversion(is_rank[common], ref_rank[common]),
                    "dominant_regime": wl.mode().iloc[0],
                })
        w0 = w0 + step

    table = pd.DataFrame(rows).dropna(subset=["concentration", "inversion"])
    corr = float(table["concentration"].corr(table["inversion"])) if len(table) > 1 else float("nan")
    x, y = table["concentration"].values, table["inversion"].values
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 and np.std(x) > 0 else float("nan")
    return InversionResult(table=table, correlation=corr, slope=slope)
