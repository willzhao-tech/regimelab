# -*- coding: utf-8 -*-
"""
Unit tests for the vol-arb harness (volarb_harness.py).

Covers the five invariants that protect against the failure modes seen in the
study (3 fake Sharpes were caught during development):
  1. Causality   — signal/forecast value at t is unchanged when future rows are removed.
  2. Park formula — fcast_vol("parkW") matches a hand-computed example exactly.
  3. P&L arithmetic — backtest() reproduces hand-computed pnl on a tiny synthetic case.
  4. Walk-forward — parameter picks use ONLY trailing train data (pick lags a regime flip).
  5. Costs       — transaction cost is charged on position CHANGES only.

All tests run on synthetic data; no CSV files are required.

Run:
  "C:\\Users\\ASUS\\Desktop\\claude doc\\market study\\regimelab\\regimelab\\regimelab\\.venv\\Scripts\\python.exe" -m pytest tests\\test_volarb.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import volarb_harness as H

SQ = np.sqrt(252.0)


# ---------------------------------------------------------------- fixtures
def make_synth(n=300, seed=7):
    """Synthetic OHLC frame + VXN series with a business-day index."""
    rs = np.random.RandomState(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    close = pd.Series(100.0 * np.exp(np.cumsum(rs.normal(0, 0.01, n))), index=idx)
    spread = np.abs(rs.normal(0.01, 0.003, n)) + 1e-4
    high = close * (1 + spread / 2)
    low = close * (1 - spread / 2)
    df = pd.DataFrame({"High": high, "Low": low, "Close": close}, index=idx)
    ret = close.pct_change()
    vxn = pd.Series(20.0 + 5.0 * np.sin(np.arange(n) / 17.0) + rs.normal(0, 1, n),
                    index=idx).abs()
    return df, ret, vxn


def signal_regime_combo(df, ret, vxn, r_hi=2.0, r_lo=-2.0, d=0.0):
    """Family-A signal exactly as in va_regime_combo.py (built from harness fcasts)."""
    fc21 = H.fcast_vol(df, ret, "park21")
    fc10 = H.fcast_vol(df, ret, "park10")
    fc42 = H.fcast_vol(df, ret, "park42")
    richness = vxn - fc21
    trend = fc10 - fc42
    s = pd.Series(0.0, index=ret.index)
    s[((richness >= r_hi) & (trend <= -d)).fillna(False)] = 1.0
    s[((richness <= r_lo) & (trend >= d)).fillna(False)] = -1.0
    return s


# ---------------------------------------------------------------- 1. causality
def test_causality_truncation_fcast():
    """fcast_vol value at t must be identical when all rows after t are deleted."""
    df, ret, vxn = make_synth(300)
    for kind in ("park10", "park21", "park42", "ewma94", "rv21"):
        full = H.fcast_vol(df, ret, kind)
        for cut in (120, 200, 290):
            trunc = H.fcast_vol(df.iloc[:cut], ret.iloc[:cut], kind)
            overlap_full = full.iloc[:cut]
            # every defined value on the overlap must match exactly
            # (assert_series_equal raises with kind/cut visible in the parametrization)
            pd.testing.assert_series_equal(trunc, overlap_full, check_names=False,
                                           obj=f"fcast_vol({kind}) cut={cut}")


def test_causality_truncation_signal():
    """Family-A signal value at t unchanged when future rows are removed."""
    df, ret, vxn = make_synth(300)
    full = signal_regime_combo(df, ret, vxn)
    for cut in (150, 250):
        trunc = signal_regime_combo(df.iloc[:cut], ret.iloc[:cut], vxn.iloc[:cut])
        assert (trunc == full.iloc[:cut]).all(), \
            f"look-ahead detected in regime_combo signal at cut={cut}"


def test_fcast_is_shifted():
    """The forecast at t must NOT use row t: changing ONLY row t's High/Low leaves
    fcast at t untouched (it may only affect t+1 onward)."""
    df, ret, vxn = make_synth(120)
    base = H.fcast_vol(df, ret, "park10")
    df2 = df.copy()
    t = 100
    df2.iloc[t, df2.columns.get_loc("High")] = df2["High"].iloc[t] * 1.5  # huge range shock at t
    bumped = H.fcast_vol(df2, ret, "park10")
    assert bumped.iloc[t] == pytest.approx(base.iloc[t]), \
        "fcast at t reacted to same-day data: missing shift(1)"
    assert bumped.iloc[t + 1] != pytest.approx(base.iloc[t + 1]), \
        "fcast never saw the shock: rolling window broken"


# ---------------------------------------------------------------- 2. park formula
def test_park_forecast_hand_computed():
    """park(2) on a 3-row hand example.

    H/L ratios: 1.1, 1.2, 1.05  ->  ln^2 = 0.009084030374332749,
    0.03324115007177121, 0.0023804801196801303.
    park2_t = sqrt( rolling_2 mean of ln(H/L)^2 / (4 ln 2) ) * sqrt(252) * 100,
    then SHIFTED 1 day, so:
      fcast[2] = sqrt( (ln(1.1)^2 + ln(1.2)^2)/2 / (4 ln 2) ) * sqrt(252) * 100
               = 138.68898045379197
      fcast[3] = sqrt( (ln(1.2)^2 + ln(1.05)^2)/2 / (4 ln 2) ) * sqrt(252) * 100
               = 127.23290457291321
    """
    idx = pd.bdate_range("2020-01-01", periods=4)
    df = pd.DataFrame({
        "High":  [110.0, 120.0, 105.0, 108.0],
        "Low":   [100.0, 100.0, 100.0, 100.0],
        "Close": [105.0, 110.0, 102.0, 104.0],
    }, index=idx)
    ret = df["Close"].pct_change()
    f = H.fcast_vol(df, ret, "park2")
    assert np.isnan(f.iloc[0]) and np.isnan(f.iloc[1]), \
        "first window-incomplete + shifted values must be NaN"
    assert f.iloc[2] == pytest.approx(138.68898045379197, rel=1e-12)
    assert f.iloc[3] == pytest.approx(127.23290457291321, rel=1e-12)


# ---------------------------------------------------------------- 3. pnl arithmetic
def test_pnl_arithmetic_hand_computed():
    """5-day synthetic case with VXN=20 flat, known returns and signal.

    s   = [0, 1, 1, -1, -1]  -> pos = s.shift(1) = [0, 0, 1, 1, -1]
    ret = [nan, 0.01, 0.02, 0.00, 0.01]
    iv  = (20/100)^2/252 = 1.5873015873015873e-4 every day (strike at prior close)
    cost unit = 2*20*0.5/1e4/252 = 7.936507936507937e-6 per |dpos|
    Hand-computed daily pnl (day0 dropped, iv undefined):
      day1: pos=0                      -> 0.0
      day2: 1*(iv - 0.0004) - 1*cost   -> -2.4920634920634920e-4
      day3: 1*(iv - 0)                 -> +1.5873015873015876e-4
      day4: -1*(iv - 0.0001) - 2*cost  -> -7.4603174603174630e-5
    """
    idx = pd.bdate_range("2021-01-01", periods=5)
    ret = pd.Series([np.nan, 0.01, 0.02, 0.00, 0.01], index=idx)
    vxn = pd.Series(20.0, index=idx)
    s = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0], index=idx)

    pnl = H.backtest(s, ret, vxn, cost_volpt=0.5)

    expected = pd.Series([0.0,
                          -0.0002492063492063492,
                          0.00015873015873015876,
                          -7.460317460317463e-05], index=idx[1:])
    assert len(pnl) == 4, "day 0 (iv undefined) must be dropped"
    pd.testing.assert_series_equal(pnl, expected, rtol=1e-12, check_names=False)


def test_signal_is_clipped_and_shifted():
    """backtest clips s to [-1,1] and trades it with a 1-day lag."""
    idx = pd.bdate_range("2021-01-01", periods=4)
    ret = pd.Series([np.nan, 0.0, 0.0, 0.0], index=idx)
    vxn = pd.Series(20.0, index=idx)
    s = pd.Series([5.0, 0.0, 0.0, 0.0], index=idx)  # oversized signal on day 0
    pnl = H.backtest(s, ret, vxn, cost_volpt=0.0)
    iv = (20 / 100.0) ** 2 / 252.0
    # day1 holds pos=clip(5)=1 (not 5): pnl = 1*iv
    assert pnl.iloc[0] == pytest.approx(iv, rel=1e-12)
    # day2/3 flat
    assert (pnl.iloc[1:] == 0).all()


# ---------------------------------------------------------------- 4. walk-forward
def test_walk_forward_uses_train_only():
    """Best param flips mid-sample; the walk-forward pick must LAG the flip.

    Construction: 350 days, regime flips at index 200 (a test-block boundary
    with train=100, test=50).  Param 'g1' has Sharpe >> 0 before the flip and
    << 0 after; 'g2' is the mirror image.
      block starts: 100, 150, 200, 250, 300
      - picks[0] (train 0..99,   all pre-flip)  -> g1
      - picks[1] (train 50..149, all pre-flip)  -> g1
      - picks[2] (train 100..199, all pre-flip) -> g1, even though its TEST
        window 200..249 is post-flip where g2 wins.  A peeking implementation
        would pick g2 here.
      - picks[4] (train 200..299, all post-flip) -> g2
    """
    n, flip = 350, 200
    rs = np.random.RandomState(0)
    idx = pd.bdate_range("2010-01-01", periods=n)
    noise1 = rs.normal(0, 0.001, n)
    noise2 = rs.normal(0, 0.001, n)
    g1 = pd.Series(np.where(np.arange(n) < flip, 0.01, -0.01) + noise1, index=idx)
    g2 = pd.Series(np.where(np.arange(n) < flip, -0.01, 0.01) + noise2, index=idx)
    series = {"g1": g1, "g2": g2}

    oos = H.walk_forward(lambda g: series[g], ["g1", "g2"], train=100, test=50)
    picks = oos.attrs["picks"]

    assert len(picks) == 5
    assert picks[0] == "g1"
    assert picks[1] == "g1"
    assert picks[2] == "g1", ("pick for the test block starting AT the flip used "
                              "post-flip (future) information")
    assert picks[4] == "g2", "walk-forward never adapted after the flip entered train"

    # and the OOS stream is the concatenation of the PICKED series on TEST blocks only
    expected_first_block = g1.iloc[100:150]
    pd.testing.assert_series_equal(oos.iloc[:50], expected_first_block,
                                   check_names=False)


def test_walk_forward_oos_starts_after_train():
    """No OOS pnl may come from the first `train` days."""
    n = 350
    idx = pd.bdate_range("2010-01-01", periods=n)
    s = pd.Series(np.random.RandomState(1).normal(0.001, 0.01, n), index=idx)
    oos = H.walk_forward(lambda g: s, ["only"], train=100, test=50)
    assert oos.index[0] == idx[100]
    assert oos.index[-1] <= idx[-1]


# ---------------------------------------------------------------- 5. costs
def test_cost_charged_on_position_changes_only():
    """pnl(with cost) differs from pnl(zero cost) ONLY on days the position changes,
    and by exactly 2*VXN_(t-1)*0.5/1e4/252 * |dpos|."""
    df, ret, vxn = make_synth(200, seed=3)
    rs = np.random.RandomState(11)
    s = pd.Series(rs.choice([-1.0, 0.0, 1.0], size=200), index=ret.index)

    pnl_free = H.backtest(s, ret, vxn, cost_volpt=0.0)
    pnl_cost = H.backtest(s, ret, vxn, cost_volpt=0.5)
    diff = (pnl_free - pnl_cost).reindex(pnl_cost.index)

    pos = s.clip(-1, 1).shift(1).fillna(0.0)
    dpos = pos.diff().abs().fillna(0.0).reindex(pnl_cost.index)
    expected_cost = (2 * vxn.shift(1) * 0.5 / 1e4 / 252).reindex(pnl_cost.index) * dpos

    # exact cost on change days
    pd.testing.assert_series_equal(diff, expected_cost, rtol=1e-12, check_names=False)
    # zero cost on no-change days, positive on change days (sanity, not tautology)
    assert (diff[dpos == 0].abs() < 1e-18).all(), "cost charged on a day without a trade"
    assert (diff[dpos > 0] > 0).all(), "no cost charged on a trading day"
    assert (dpos > 0).sum() > 10, "synthetic case degenerate: too few position changes"


def test_constant_position_pays_entry_cost_once():
    """An always-short book trades once (entry) and never again."""
    idx = pd.bdate_range("2021-01-01", periods=50)
    ret = pd.Series(0.0, index=idx); ret.iloc[0] = np.nan
    vxn = pd.Series(20.0, index=idx)
    s = pd.Series(1.0, index=idx)
    pnl_free = H.backtest(s, ret, vxn, cost_volpt=0.0)
    pnl_cost = H.backtest(s, ret, vxn, cost_volpt=0.5)
    diff = pnl_free - pnl_cost
    charged_days = diff[diff.abs() > 1e-18]
    assert len(charged_days) == 1, f"cost charged on {len(charged_days)} days, expected 1 (entry)"
    assert charged_days.index[0] == idx[1]  # pos goes 0 -> 1 on day 1 (signal lag)
    assert charged_days.iloc[0] == pytest.approx(2 * 20 * 0.5 / 1e4 / 252, rel=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
