"""
Tests for regimelab. Run with:  pytest -q   (from the package root)

These lock in the behaviors we validated during the build: the engine
reproduces the known risk-parity result, the registry works, no-look-ahead
holds, the regime identifier labels correctly, and the layers integrate.
"""
import os
import numpy as np
import pandas as pd
import pytest

import regimelab as rl
from regimelab.metrics import compute_stats
from regimelab.regime import default_model, base_vol_from_panel, RuleBasedIdentifier

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_JSON = os.path.join(os.path.dirname(HERE), "examples", "legacy_panel.json")
CORE = ["NQ", "A50", "US10Y", "XAU"]


@pytest.fixture(scope="module")
def panel():
    return rl.data.build_panel([rl.data.LegacyJsonSource(PANEL_JSON)])


def test_panel_loads(panel):
    assert panel.returns.shape[1] == 7
    assert len(panel.common_dates(CORE)) > 1800


def test_registry_lists_strategies():
    avail = rl.strategies.available()
    for s in ["risk_parity", "equal_weight", "fixed", "trend"]:
        assert s in avail


def test_risk_parity_reproduces_known_result(panel):
    res = rl.strategies.run(rl.strategies.get("risk_parity"), panel, instruments=CORE,
                            target_vol=0.10, vol_win=63, rebal="M", start="2019-01-01")
    s = compute_stats(res.returns.values)
    # known: CAGR ~+10.2%, Sharpe ~0.84, maxDD ~-28.6%
    assert 0.08 < s.cagr < 0.12
    assert 0.78 < s.sharpe < 0.90
    assert -0.32 < s.max_dd < -0.25


def test_equal_weight_beats_risk_parity_in_sample(panel):
    rp = compute_stats(rl.strategies.run(rl.strategies.get("risk_parity"), panel,
                       instruments=CORE, start="2019-01-01").returns.values)
    ew = compute_stats(rl.strategies.run(rl.strategies.get("equal_weight"), panel,
                       instruments=CORE, start="2019-01-01").returns.values)
    # the documented in-sample inversion premise
    assert ew.sharpe > rp.sharpe


def test_no_lookahead_first_weight_uses_only_past():
    # construct a panel where the last day is a huge spike; weights at a rebalance
    # must not depend on future returns. We check determinism w.r.t. truncation.
    dates = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 0.01, size=(200, 4)), index=dates, columns=CORE)
    p_full = rl.Panel(returns=df)
    p_trunc = rl.Panel(returns=df.iloc[:150])
    r_full = rl.strategies.run(rl.strategies.get("risk_parity"), p_full, instruments=CORE, rebal="M")
    r_trunc = rl.strategies.run(rl.strategies.get("risk_parity"), p_trunc, instruments=CORE, rebal="M")
    # overlapping early period must match exactly (future data cannot leak back)
    common = r_full.returns.index.intersection(r_trunc.returns.index)
    assert np.allclose(r_full.returns.loc[common].values,
                       r_trunc.returns.loc[common].values, atol=1e-12)


def test_regime_identifier_labels(panel):
    dates = pd.bdate_range("2020-01-01", periods=4)
    macro = pd.DataFrame({"recession": [1, 0, 0, 0], "cpi_yoy": [1, 7, 2, 2],
                          "vix": [40, 20, 12, 32], "equity_dd": [-0.25, 0, 0, 0]}, index=dates)
    p = rl.Panel(returns=pd.DataFrame({"NQ": [0.0] * 4}, index=dates), macro=macro)
    labels = RuleBasedIdentifier().label(p)
    assert labels.iloc[0] == "Crash"        # recession + vix40 + deep dd
    assert labels.iloc[1] == "Stagflation"  # cpi 7
    assert labels.iloc[2] == "Goldilocks"   # calm, vix<15
    assert labels.iloc[3] == "Risk-off"     # vix 32


def test_forward_simulation_runs(panel):
    model = default_model(CORE, base_vol_from_panel(panel, CORE), seed=1)
    sim = model.simulate(horizon_years=3)
    assert sim.matrix.shape[0] == 4
    assert sim.matrix.shape[1] == 3 * 252
    assert len(sim.regime_sequence) == 3


def test_deflated_sharpe_monotonic():
    from regimelab.evaluation import deflated_sharpe
    # more trials -> harder to clear -> lower DSR for the same observed Sharpe
    high = deflated_sharpe(0.8, n_periods=2000, n_trials=5)
    low = deflated_sharpe(0.8, n_periods=2000, n_trials=200)
    assert high > low
