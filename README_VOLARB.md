# Vol-Arb on NQ/VXN — Long-Short Variance Strategy (walk-forward OOS study)

A long-short variance-risk-premium study on the Nasdaq-100 future (NQ) against the
CBOE Nasdaq-100 Volatility Index (VXN). Two independent causal signal families
(A `regime_combo`, B `range_forecast`) gate a daily +/-1 variance position; parameters
are chosen **only** by rolling walk-forward (1260-day train -> 252-day test). The study's
claim is a **timing alpha vs static short-vol with inverted (positive) skew** — not the
absolute Sharpe level, which is inflated by the proxy instrument (see Limitations).

OOS 2006-02-13 .. 2026-03-12 (proxy P&L): A Sharpe 1.57 (skew +20), B 1.18 (skew +19),
50/50 blend 1.74 (skew +16, alpha-t vs static ~+11 OLS); static short-vol on the same
dates 1.01 with skew -14.

---

## 1. Repository layout

| Path | Role |
|---|---|
| `volarb_harness.py` | Core library: `load()`, `fcast_vol()`, `iv_rvar()`, `backtest()`, `metrics()`, `walk_forward()` |
| `va_regime_combo.py` | Family A signal + walk-forward run + look-ahead self-audit |
| `va_range_forecast.py` | Family B signal + walk-forward run + look-ahead self-audit |
| `examples/34_reproduce_volarb.py` | Full reproduction package: A, B, blend, static baseline, ledgers, trade logs, equity plot, spec |
| `market_data.py`, `update_nq.py`, `fetch_nq.py` | Data pipeline (Investing.com financialdata API, Yahoo fallback) |
| `tests/test_volarb.py` | Pytest suite: causality, park formula, pnl arithmetic, walk-forward hygiene, cost accounting |
| `C:\Users\ASUS\Desktop\claude doc\1\` | DATA + ARTIFACTS directory (CSVs in, ledgers/plots/spec out) |

Key artifacts (in `C:\Users\ASUS\Desktop\claude doc\1`):
`VOLARB_SPEC.md` (recreation spec), `volarb_ledger_A.csv` / `_B.csv` / `_blend.csv`
(daily ledgers: every input, signal, position, pnl), `volarb_trades_A.csv` / `_B.csv`
(position-change logs), `volarb_equity.png` (equity / drawdown / rolling Sharpe).

---

## 2. EXACT runtime instructions

**Python interpreter (always use the project venv):**

```
"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab\.venv\Scripts\python.exe" <script>
```

**Working directory for scripts:** `C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab`
(scripts use absolute paths internally, so cwd is not critical, but run them from here).

**Network:** data fetching requires the local proxy at `http://127.0.0.1:7897`
(China geo-block). The pipeline sets it by default (`market_data.DEFAULT_PROXY`);
no env vars needed if the proxy is running.

**Run order:**

```powershell
$py = "C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab\.venv\Scripts\python.exe"
$proj = "C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab"

# 1) Refresh data (NQ + VXN; appends to the CSVs in ...\claude doc\1)
& $py "$proj\update_nq.py"

# 2) Reproduce the headline result (writes ledgers, trade logs, plot, VOLARB_SPEC.md)
& $py "$proj\examples\34_reproduce_volarb.py"

# 3) Per-family runs with diagnostics + programmatic look-ahead audits
& $py "$proj\va_regime_combo.py"
& $py "$proj\va_range_forecast.py"

# 4) Any P0/P1 follow-up scripts (robustness, sensitivity, documentation tasks)
#    run AFTER step 2, since they read the volarb_* artifacts and/or import
#    volarb_harness on the refreshed CSVs.

# 5) Unit tests
& $py -m pytest "$proj\tests\test_volarb.py" -v
```

Step 1 is optional if the CSVs in `C:\Users\ASUS\Desktop\claude doc\1` are already
current — every backtest step runs fully offline from those CSVs.

---

## 3. Dependencies

Python **3.13.13** (venv at `.venv\` in the project root). Versions installed and tested:

| Package | Version | Used for |
|---|---|---|
| pandas | 2.3.3 | all data handling |
| numpy | 2.4.6 | math |
| matplotlib | 3.10.9 | `volarb_equity.png` and study plots |
| scikit-learn | 1.9.0 | auxiliary studies (regime models) |
| optuna | 4.9.0 | auxiliary studies (not used in the headline result) |
| plotly | 6.8.0 | interactive reports (auxiliary) |
| scipy | 1.17.1 | stats helpers |
| requests | 2.34.2 | Investing.com / Yahoo fetchers |
| pytest | 9.0.3 | test suite |

The vol-arb core (`volarb_harness.py` + the two `va_*` scripts + example 34) needs only
pandas, numpy, matplotlib.

---

## 4. Data sources

All series are daily `Date,Open,High,Low,Close,Volume` CSVs in
`C:\Users\ASUS\Desktop\claude doc\1`. Primary source is the **Investing.com
financialdata API** (Stooq was removed/CAPTCHA-walled); fallback for VXN is
**Yahoo Finance `^VXN`**. To discover a new pairId:
`GET api.investing.com/api/search/v2/search?q=<name>` (see
`market_data.find_investing_pair`). All HTTP goes through the local proxy
`127.0.0.1:7897`.

| CSV | Instrument | Investing pairId | Coverage |
|---|---|---|---|
| `NQ_F_all_history.csv` | Nasdaq-100 future (continuous) | **8874** | 1999-06-22 .. present |
| `VXN_all_history.csv` | CBOE Nasdaq-100 Volatility Index | **44369** (Yahoo `^VXN` fallback) | 2001-02-05 .. present |
| `SPX_all_history.csv` | S&P 500 index | **166** | 1999-01-04 .. present |
| `VIX_all_history.csv` | CBOE Volatility Index | **44336** | 1990-01-03 .. present |
| `SX5E_all_history.csv` | EURO STOXX 50 | **175** | 2011-08-15 .. present |
| `VSTOXX_all_history.csv` | VSTOXX (futures) | **40817** | 2013-06-24 .. present |
| `SKEW_all_history.csv` | CBOE SKEW Index | **961123** | 2013-10-16 .. present |

The headline strategy uses only NQ + VXN; the rest support cross-asset / robustness
studies.

---

## 5. Strategy specification

(Verbatim from `VOLARB_SPEC.md`, which example 34 regenerates on every run.)

### Data alignment
Align NQ and VXN on common dates. `ret_t = Close_t/Close_(t-1) - 1`.

### Instrument (PROXY — the key caveat)
Daily variance-swap proxy, P&L per unit notional:

```
P&L_t  = pos_t * (iv_t - rvar_t) - cost_t
iv_t   = (VXN_(t-1)/100)^2 / 252            # strike set at prior close
rvar_t = ret_t^2
cost_t = 2*VXN_(t-1)*0.5/10^4/252 * |pos_t - pos_(t-1)|   # 0.5 vol-pt spread
```

This 1-day-variance-at-VXN-strike is NOT directly tradable; it inflates Sharpe
LEVELS for strategy and baseline alike. Expect a real NDX var-swap/option
implementation around 0.8–1.2 Sharpe. The relative results (alpha vs static,
skew inversion) are the robust claims.

### Causal signals
`s_t` uses data through close `t`; the harness shifts it, so the position held on
day `t+1` is `s_t`. Vol forecast (value at `t` uses data through `t-1` — shifted
inside `fcast_vol`):

```
park(w)_t = sqrt( mean_(i=t-w..t-1) ln(High_i/Low_i)^2 / (4 ln 2) ) * sqrt(252) * 100
```

**A) regime_combo** — `richness_t = VXN_t - park(21)_t`; `volTrend_t = park(10)_t - park(42)_t`

```
s = +1 (short vol) if richness >= r_hi AND volTrend <= -d
s = -1 (long  vol) if richness <= r_lo AND volTrend >= +d
else 0
grid: r_hi in (2,4,6), r_lo in (0,-2), d in (0,1,2)     # 18 combos
```

Dominant walk-forward pick: `(r_hi, r_lo, d) = (2, -2, 0)`.

**B) range_forecast** — `range_t = ln(High_t/Low_t)*100`; `breakeven_t = VXN_t/sqrt(252)`

```
s = +1 if range < b1*breakeven      # tape quieter than implied -> sell vol
s = -1 if range > b2*breakeven      # tape hotter than implied  -> buy vol
else 0
grid: b1 in (0.8,1.0,1.2), b2 in (1.3,1.6,2.0)          # 9 combos
```

### Walk-forward (NO in-sample parameter choice)
Train = 1260 trading days, Test = 252, rolling. In each block pick the grid params
with the highest **TRAIN** Sharpe; apply to the next 252-day TEST block; concatenate
TEST blocks only. Blend = `0.5*A + 0.5*B` daily P&L. Total combos for deflation
accounting: 18 (A) + 9 (B) = 27, plus the implicit choice to blend.

---

## 6. Results summary (proxy P&L, OOS 2006-02-13 .. 2026-03-12)

| Strategy | Sharpe | Skew | Notes |
|---|---|---|---|
| A regime_combo | **1.57** | **+20** | long-short, mostly flat/short |
| B range_forecast | **1.18** | **+19** | independent gating variable |
| A+B 50/50 blend | **1.74** | **+16** | alpha-t vs static ~ +11 (OLS, same dates) |
| static short-vol (s=+1) | 1.01 | **-14** | baseline on identical dates |

**Where the Sharpe comes from, and what breaks it.** The P&L instrument is a daily
variance swap struck at the prior VXN close — not a tradable contract. The favorable
assumptions are (i) you can roll a 1-day var swap at the VXN level every close for
0.5 vol-pt round-trip cost, and (ii) zero margin/funding drag. If those fail, the
**level** of every Sharpe above drops (expect ~0.8–1.2 for a real NDX var-swap or
option-replication implementation). What survives proxy failure: the **relative**
claims — both families beat static short-vol on the same dates with a highly
significant timing alpha, and they flip the skew from -14 (classic short-vol crash
profile) to positive ~+16..+20 (the strategy is long vol into stress). Verify any
number against `volarb_ledger_*.csv`.

**Anti-overfitting hygiene enforced** (3 fake Sharpes were caught and discarded
during development):

- Causal only: all trailing stats; signals shifted one extra day by the harness.
- Parameters via walk-forward only; train numbers are never reported as results.
- No tail deletion / capping / winsorizing — the ledgers include 2008-10-13 raw.
- Combo counts reported for deflated-Sharpe reasoning (27 total).
- Programmatic look-ahead audits (truncation tests) run inside both `va_*` scripts
  and in `tests/test_volarb.py`.

---

## 7. Limitations

1. **Proxy instrument (the big one).** Daily variance at the VXN strike is not
   tradable. VXN is a 30-day implied-vol index; a real implementation needs NDX
   variance swaps, VolTarget futures, or delta-hedged options, each with basis,
   term-structure and roll effects that this study does not model. Sharpe LEVELS
   here are inflated for strategy and baseline alike.
2. **Cost model is crude.** Flat 0.5 vol-pt spread charged as
   `2*VXN*0.5/1e4/252` per unit turnover. Real var-swap spreads widen exactly when
   family A trades (high richness episodes); stress-period costs are understated.
3. **Single market, single regime history.** NQ/VXN 2001-2026 only. The OOS window
   covers 2008 and 2020, but it is still one instrument; the walk-forward picks
   (dominant `(2,-2,0)`) could be regime-specific.
4. **Selection within this study.** A and B are the 2 survivors of a wider family
   search; the blend doubles down on them. Walk-forward protects the parameter
   choice but not the family choice — deflate accordingly.
5. **VXN data quirks.** Investing.com VXN history starts 2001-02-05 and occasionally
   lags a day vs NQ (alignment drops those dates); pre-2003 VXN had a different
   methodology (old VXN was at-the-money implied vol).
6. **No intraday execution modeling.** Signals are computed at the close and assumed
   filled at the same close one day later (via the harness shift). Gaps between
   signal and execution closes are uncosted beyond the spread.

---

## 8. Tests

```powershell
& "C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab\.venv\Scripts\python.exe" `
  -m pytest "C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab\tests\test_volarb.py" -v
```

10 tests, all on synthetic data (no CSVs required): truncation/causality of
`fcast_vol` and the family-A signal, shift discipline of the forecast,
hand-computed Parkinson values, hand-computed P&L on a 5-day case, signal clip+lag,
walk-forward train-only picks (regime-flip lag test), OOS-starts-after-train, cost
charged on position changes only (exact amount), single entry cost for a constant
position.
