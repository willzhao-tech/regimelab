# Multi-Market Equity-Vol Book — the floor book

A gated, multi-market, short-dated **equity-volatility timing** strategy, backtested under
realistic execution frictions. This is the productionized end-state of the regimelab vol research:
the honest, look-ahead-free, friction-realistic version of "ride the dominant vol regime, step
aside before the break."

> **Paper only. Educational. Not investment advice.** No real orders are placed anywhere in this code.

---

## 1. What it is

- **Universe (8 markets):** SPX/VIX, NQ/VXN, EEM/VXEEM, DAX/VDAX, SX5E/VSTOXX, N225/JNIV,
  HSI/VHSI, NIFTY/INDIAVIX. Each = an equity index paired with its own listed vol index.
- **Instrument:** a 1-day-to-expiry, delta-hedged ATM **straddle** (a near-pure *gamma* expression,
  ~zero vega). Premium modeled as `prem = 2·(2·Φ(0.5·k·σ·√dt) − 1)`, short P&L `= prem − |ret|`.
  `k = 0.82` is the **measured** ratio of 1-day IV to the 30-day index (from SPX VIX1D/VIX).
- **Signals (per market, walk-forward):** two regime families —
  (A) `regime_combo` (vol-richness vs Parkinson forecast + vol trend) and
  (B) `range_forecast` (realized H/L range vs the straddle breakeven). Both are `.shift(1)`-causal;
  params are re-selected every 252 days from the prior 1260 days (no global fit).
- **The book = the "selection-free floor":** sleeves combined by
  `weight = inverse-trailing-vol × cost-coverage-gate` — **no return information is used** in the
  weighting (the most defensible, hardest-to-overfit choice), then a causal 10%-vol target (cap 4×).

The cost-coverage gate only trades a market while its *trailing* net-of-friction edge ≥ 0, so the
book **steps fully out** of markets/regimes that stop paying for their frictions.

## 2. Frictions modeled (the "L4" model)

Cumulative, all charged causally:
1. proportional bid-ask (2.5% of premium US / 4% non-US, per in-market day);
2. **stress-widening** spread: `× (1 + max(0, VolIdx/trailing-median − 1))` (high vol ⇒ wider);
3. **long-vol legs pay 2×** the spread (bid-ask widens precisely when you buy vol);
4. discrete-strike slippage (±0.125% off-ATM payoff offset);
5. cost floor (1.5 bp of notional/day) + EEM American-assignment penalty;
6. optional **margin-funding** drag (rate × 15% margin, per in-market day) — see `examples/47`.

## 3. Honest results (full L4 frictions, 2005–2026, walk-forward)

| metric | value | note |
|---|---|---|
| Sharpe (active days) | **1.26** | the headline; days the book is deployed |
| Sharpe (**calendar basis**, idle = 0) | **1.01** | the honest live number — book is only **64% deployed** |
| maxDD | −20% | vs −39% for naive equal-risk |
| skew / worst day | −1.86 / −6.9% | best tail of the finalists (still < 0) |
| Calmar | 0.52 | |

**Robustness (`examples/47`):** Sharpe ranges **1.07 (adverse: short 1000/200 window + 5% funding)
→ 1.26 (central) → 1.54 (long 1500/252 window)**; starting later only helps (1.69 from 2013).
Margin funding at 5% costs only ~0.04 Sharpe.

### The caveats that matter (read these)

- **k-sensitivity is the #1 model risk (`examples/49`).** ±0.03 in `k` swings active Sharpe
  0.88 ↔ 1.98. `k` was measured as an *average* 0.82 (0.79 vol-falling / 0.87 vol-rising) on SPX
  and extrapolated to all markets/all history. The book trades mostly in **calm/falling-vol**
  regimes where the effective `k` is nearer **0.79** → treat **calendar Sharpe ~0.6–1.0** as the
  prudent planning band, not 1.26. Only real per-strike option quotes resolve this.
- **Spread-fragile.** Profitable to ~1.5× quoted spreads; near zero by 2×. Live spread > 1.5×
  quoted is the kill criterion.
- **SPX carries it (`examples/48`).** SPX = **60%** of book P&L (inverse-vol concentrates in the
  calmest sleeve); the ex-US sleeves add little net of their wider frictions.
- **Crisis "protection" is absence, not a hedge.** In 2008 the book traded **1 day** (gate fully
  off); 2020 just 44 days. It sidesteps crises by stepping out — and thereby forgoes the rich
  post-crash premium.
- **Timing is the product.** The unconditional (always-on) short-vol book *loses* under these
  frictions (Sharpe −0.97). All of the value is *when* it trades. Gross-leg attribution shows the
  **long-vol legs out-earn the short-vol legs (71% vs 29%)** — the edge is timing vol *expansions*,
  not harvesting premium.

## 4. Run it

```powershell
# from the package root, using the bundled venv
$py = ".\.venv\Scripts\python.exe"

& $py examples\45_optimized_book.py     # THE BOOK: equity curve + ledger -> optimized_book.png/.csv
& $py examples\47_funding_robust.py     # P0.2/P0.3 margin-funding + window/start-date robustness
& $py examples\48_attribution.py        # P1.5 leg / regime / per-market / time-in-market attribution
& $py examples\49_k_sensitivity.py      # P1.6 k-sensitivity sweep (the key fragility)
& $py examples\44_book_stress.py        # friction ladder + spread sweep + leave-one-out + OOS split
& $py examples\46_skew_and_placebo.py   # 4-finalist same-axes table + timing-shuffle placebo

# tests
& $py tests\test_book_causality.py      # gold-standard no-look-ahead guarantee (future-perturbation)
& $py -m pytest tests\ -q               # full suite
```

Shared code: `bookopt_harness.py` (sleeve P&L under L4 frictions, walk-forward, `book_of`) and
`bookopt_floor.py` (the single source of truth for the floor book — reused by ex 47–49 and the test).

## 5. Dependencies

- Python 3.13, `numpy`, `pandas` (2.3.x — **not** 3.0, which breaks `pandas-datareader`),
  `matplotlib`. All present in `.venv`.
- Data fetching only: see `market_data.py`. Requires a local proxy at `127.0.0.1:7897`
  (the research host is behind a China geo-block).

## 6. Data sources

- **OHLCV + vol indices:** Investing.com (default; browser-header API, date-paginated), Yahoo
  fallback (used for single stocks — Investing is not split-adjusted there). All series are
  full-history daily OHLCV, stored as `*_all_history.csv` in the data dir.
- **`k` calibration:** real CBOE **VIX1D** / VIX (`examples/39`).
- Pair IDs and the fetch/registry live in `market_data.py` (`DATASETS`); daily auto-updates run via
  scheduled tasks (`update_nq.py`, `update_a50.py`).

## 7. Reproducibility & integrity

- Every number above regenerates from the scripts in §4 against the cached CSVs.
- **No look-ahead is machine-checked** (`tests/test_book_causality.py`): corrupting all input data
  after a cutoff leaves pre-cutoff book P&L byte-identical.
- This research deliberately logged its own negative results and retracted artifacts; see
  `volbook_working_paper.md` (§ "What didn't work") and the regimelab memory notes.

## 8. Publishing to SSRN

SSRN has **no submission API** and its Terms of Use prohibit automated submission, so the pipeline is
**prepare-and-handoff** — it never posts to SSRN.

```powershell
$env:PYTHONIOENCODING="utf-8"
& $py build_paper_pdf.py            # volbook_working_paper.md -> ssrn_submission/timing_is_the_product.pdf
& $py build_ssrn_submission.py      # + ssrn_metadata.json + SSRN_CHECKLIST.md, then pre-flight
& $py build_ssrn_submission.py --open   # same, then opens the SSRN login for YOU to drive the upload
```

- `build_paper_pdf.py` renders the working paper to an SSRN-grade PDF (ReportLab; title page, abstract,
  JEL/keywords, AI-disclosure, references) with the headline numbers injected from `results.json`
  (render-from-data) and an auto-generated key-results appendix.
- `build_ssrn_submission.py` assembles `ssrn_submission/` (PDF + metadata JSON + a copy-paste portal
  checklist) and runs a pre-flight (PDF-only, abstract 250–400 words, title+author on page 1,
  AI-disclosure present, 1–7 classifications). It will not open the browser if a HARD check fails, and
  never auto-fills or submits the form — a human drives the upload.
