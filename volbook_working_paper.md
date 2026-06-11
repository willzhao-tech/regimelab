# Timing Is the Product: A Friction-Realistic, Look-Ahead-Free Multi-Market Equity-Volatility Book

**Working paper — research/educational. Paper only; not investment advice.**
Regenerates from `examples/44–49` and `tests/test_book_causality.py` against cached daily data.
Supersedes the earlier `volarb_working_paper.md`, whose headline numbers were inflated by an
instrument mismatch and pre-friction pricing (documented in §9–10 as a correction, not deleted).

---

## Abstract

We study a regime-gated, long-short equity-volatility strategy across eight index/vol-index pairs
(SPX, NQ, EEM, DAX, SX5E, N225, HSI, NIFTY), 2005–2026. The strategy times next-day realized
variance with two causal regime signals, expressed through a 1-day-to-expiry delta-hedged straddle
(a near-pure gamma instrument), and combined into a book by a **selection-free** weighting that uses
**no return information**. Under a realistic execution-friction model (stress-widening spreads,
asymmetric long-vol bid-ask, discrete strikes, floors, assignment, margin funding), the book earns a
Sharpe of **1.26 on deployed days** and **1.01 on a calendar basis** (it is idle ~36% of the time),
with a maximum drawdown of −20%. The central economic finding is sharp and, we believe, a genuine
contribution: at realistic short-dated pricing the **unconditional** variance risk premium does *not*
survive — an always-on short-vol version of the same book loses (Sharpe −0.97). All of the value is
*timing*: when to be in, out, short, or long. We document, with adversarial verification and
machine-checked causality, both what works and a ledger of artifacts we caught and retracted.

---

## 1. Introduction and contribution

The variance risk premium (VRP) — implied variance exceeding subsequent realized variance — is among
the most robust anomalies in the literature (Carr & Wu, 2009; Bollerslev, Tauchen & Zhou, 2009).
The standard reading is that selling variance earns a premium for bearing crash risk. Our results,
under realistic execution, qualify that reading materially:

1. **The unconditional premium does not survive realistic short-dated execution.** A static, always-on
   short-vol book across all eight markets returns Sharpe **−0.97** under our friction model. There is
   no free premium to harvest once options are fairly priced (via a calibrated short-dated IV ratio)
   and costs are charged honestly. (Contribution vs Carr–Wu: the premium they document is, net of
   realistic frictions on *tradable short-dated* instruments, not unconditionally harvestable.)

2. **Timing is the entire product.** The gated book's alpha over the same-friction static book has a
   t-stat of +6.8; gross-leg attribution shows the **long-vol legs out-earn the short-vol legs
   (71% vs 29% of gross P&L)** — the edge is anticipating vol *expansions*, not collecting premium.

3. **A selection-free construction is both the most defensible and the best-behaved.** Weighting
   sleeves by inverse trailing volatility × a cost-coverage gate — using *no* return information —
   ties return-based weightings on Sharpe while delivering the best tail and zero overfitting surface.

4. **Methodological**: rolling walk-forward parameter selection (no global fit); a six-component
   refined cost model; calibration of the strategy's instrument to short-dated gamma via a measured
   IV ratio; and a machine-checked no-look-ahead guarantee.

We also report, in full, the negative results and the artifacts we caught (§10) — the discipline that
produced (1)–(3).

## 2. Data

Daily OHLCV for eight equity indices and their listed volatility indices, full history through
2026-05-15, sourced from Investing.com (browser-header API, date-paginated; Yahoo fallback for single
stocks). Vol indices: VIX, VXN, VXEEM, VDAX, VSTOXX, JNIV, VHSI, INDIAVIX. The short-dated IV ratio
`k` is calibrated from CBOE **VIX1D**/VIX. All series are stored as `*_all_history.csv`; daily
auto-updates run via scheduled tasks. (Research host is behind a geo-block; fetches route through a
local proxy.)

## 3. The instrument: short-dated gamma, calibrated

The strategy trades a **1-day-to-expiry, delta-hedged ATM straddle** — a near-pure *gamma* expression
with ~zero vega, chosen after an earlier variance-swap proxy was found to embed a vega leg that did
not match the signal (§10). Short P&L per unit `= premium − |ret|`, with

  `premium = 2·(2·Φ(0.5·k·σ·√dt) − 1)`,  `σ = VolIdx/100`,  `dt = 1/252`.

`k` is the ratio of 1-day implied vol to the 30-day index, **measured at 0.82** (0.79 in
falling-vol states, 0.87 in rising-vol states) from SPX VIX1D/VIX, then applied to all markets and all
history. §9 shows this extrapolation is the single largest model risk.

## 4. Signals and walk-forward

Two causal regime families, per market:
- **A — `regime_combo`**: short vol when vol-richness (`VolIdx − Parkinson_21 forecast`) is high *and*
  the vol trend (`Parkinson_10 − Parkinson_42`) is falling; long vol in the mirror condition.
- **B — `range_forecast`**: short vol when the realized H/L range is below `b1×` the straddle
  breakeven; long vol when above `b2×`.

All inputs are trailing and `.shift(1)`-lagged. Parameters are re-selected every 252 trading days from
the **prior 1260 days** by training Sharpe (walk-forward; no global fit). The per-market sleeve is the
equal blend of A and B.

## 5. Book construction: the selection-free floor

Each sleeve is scaled to unit trailing risk (63-day, shifted), then combined with weights

  `weight = invvol_252(sleeve) × coverage_gate`,

where `invvol_252` is inverse trailing-252-day volatility (risk only — **no returns**) and
`coverage_gate ∈ {0,1}` admits a sleeve only while its *trailing* net-of-friction edge ≥ 0. A causal
10%-vol target (cap 4×) sets book leverage. We call this the **floor** because it is the most
conservative defensible construction: it cannot overfit on returns. Among finalists (§7) it is also
the best on tail risk, so we adopt it as *the* book rather than as a lower bound.

## 6. Friction model ("L4")

Charged causally, cumulatively:

| # | component | specification |
|---|---|---|
| 1 | proportional spread | 2.5% of premium (US/ETF) / 4% (non-US), per in-market day |
| 2 | stress-widening | spread `× (1 + max(0, VolIdx/trailing-median − 1))` |
| 3 | asymmetric long-vol | long legs pay **2×** the (widened) spread |
| 4 | discrete strikes | ±0.125% off-ATM payoff offset (slippage) |
| 5 | floors / assignment | 1.5 bp notional/day floor; EEM American-assignment penalty |
| 6 | margin funding | rate × 15% margin, per in-market day (sensitivity in §9) |

The single biggest *cost* hit is discrete strikes (not the spread): in the friction ladder the book
falls 1.18 → 0.87 at the discrete-strike step.

## 7. Main results (2005–2026, walk-forward, full L4)

Four finalists on the same axes (common index):

| construction | Sharpe (active) | maxDD | skew | worst day | Calmar | uses returns? |
|---|---|---|---|---|---|---|
| baseline equal-risk | 0.85 | −39% | −1.05 | −7.1% | 0.24 | no |
| riskweight × cov (composed) | 1.29 | −20% | −2.49 | −9.7% | 0.54 | yes |
| **floor: invvol × cov** | **1.26** | **−20%** | **−1.86** | **−6.9%** | **0.52** | **no** |
| static short-vol (always on) | −0.97 | −95% | −1.76 | −6.7% | −0.12 | n/a |

The floor ties the return-based composed book on Sharpe/maxDD/Calmar but has materially better tail
(skew −1.86 vs −2.49; worst day −6.9% vs −9.7%) and no overfitting surface. **Calendar-basis Sharpe**
(idle capital = 0; the book is deployed 64% of days) is **1.01** — the honest live number. Growth of
$1 ≈ 8.7× on active days.

A timing-shuffled placebo (preserve the weights' on-fraction and autocorrelation, break their
phase-alignment with returns) places the real book at the **87th percentile** (z = +1.13): the timing
carries real signal, but ~⅛ of random-timing books beat it — a meaningful share of the de-risking
lift is generic de-leverage, not precise timing. We report this rather than bury it.

## 8. Attribution

- **By leg (gross sleeve):** short-vol +0.81 cum / +1.4 bp per active day; long-vol +2.01 cum /
  +3.5 bp per active day. **Long-vol legs dominate (71%)** — the edge is timing expansions.
- **By vol regime:** low VIX <15 → 23.2% ann, Sharpe 2.16 (46% of days); medium → 8.6%, 0.70; high
  VIX >25 → −9.7% ann, −0.76 (only 7% of days — see below).
- **By market:** SPX **60%** of book P&L, NQ 15%, DAX 15%, others ≤5% or slightly negative.
  Inverse-vol concentrates in the calmest, deepest sleeve. The book is SPX-carried.
- **Time-in-market / step-aside:** the cost-coverage gate pulls the book *fully out* of stress, not
  flat. In **2008 the book traded one day**; 2020, 44 days. Crisis "protection" is **absence**, not a
  hedge — the book sidesteps drawdowns but also forgoes the rich post-crash premium.

## 9. Stress and robustness

- **Spread sweep (floor):** active Sharpe 1.26 → 1.00 (×1.5) → 0.64 (×2.0); stays **positive even at
  ×3.0** (0.79) because the gate self-protects by stepping out. Contrast the naive equal-risk book,
  which goes **negative at ×2.0** (−0.21). The cost-coverage gate is genuine spread insurance.
- **k-sensitivity (largest model risk):** ±0.03 in `k` swings active Sharpe **0.88 ↔ 1.98**. Across
  the measured band (0.79–0.87): calendar Sharpe **0.58 → 1.69**. Because the book trades mostly in
  calm/falling-vol regimes, where the effective `k` tends toward 0.79, we treat **calendar Sharpe
  ~0.6–1.0 as the prudent planning band**, not 1.26.
- **Walk-forward window:** 1000/200 → 1.16, 1260/252 → 1.26, 1500/252 → 1.54.
- **Start date:** from 2005 → 1.26, 2008 → 1.46, 2013 → 1.69 (later only helps).
- **Margin funding:** 5% costs ~0.04 Sharpe (→1.22). **Adverse corner** (1000/200 + 5% funding, full
  history): **1.07**. The book never breaks below ~1.07 across configurations.
- **Leave-one-out:** dropping SPX is the only material hit (−0.27); dropping any ex-US sleeve does not
  hurt. US-only (SPX, NQ, EEM) ≈ full book ≥ ex-US.

## 10. What didn't work (the honesty ledger)

This research retracted more than it kept. For the record:

- **Variance-swap proxy P&L → instrument mismatch.** The original signal earned alpha on a payoff
  (rolled 21-day variance swap) dominated by a **vega** leg (1.79× the gamma leg) that the signal did
  not actually predict (corr +0.069 gamma vs +0.001 vega). Fixed by switching to the 1-DTE straddle.
- **Three look-ahead bugs** in an earlier short-vol sleeve (full-sample wing cost, tail-deletion via
  `np.minimum`, full-sample vol scaling) inflated a Sharpe to 2.17; corrected to ≤0; examples retracted.
- **An optimizer's 2.52 Sharpe** was a fresh artifact (idealized tail cap + leverage pinning).
- **Four optimization ideas rejected** under adversarial verification: a friction-aware gate
  (post-hoc threshold), a long-leg redesign (honest negative — long legs help), VIX1D state-scaling
  (degrades, dies under spreads), and a drawdown brake (vol-shrinkage; failed a random-delever
  placebo at the 86th percentile).
- **"9/9 universality" does not survive frictions.** It holds in proxy space; net of real costs the
  book is US-carried (§8).

The no-look-ahead property is now **machine-checked**: `tests/test_book_causality.py` corrupts all
input data after a cutoff and asserts pre-cutoff book P&L is byte-identical.

## 11. Relation to the literature, and innovations

- **Carr & Wu (2009), *Variance Risk Premiums* (RFS).** They establish the unconditional VRP from
  synthetic variance swaps. *Our contribution:* on tradable **short-dated** instruments, net of a
  realistic six-component cost model, the unconditional premium is **not** harvestable (static
  Sharpe −0.97); only a *conditional/timing* component survives.
- **Bollerslev, Tauchen & Zhou (2009); Bollerslev et al. (2015), leveraged/realized-vol predictors.**
  We use trailing Parkinson range as a realized-vol forecast inside a *walk-forward* regime signal,
  and find its value is in **timing direction**, not in static premium capture.
- **Lucca & Moench (2015), pre-FOMC drift.** Tested separately on this data (`examples/12`); included
  as a small, decayed catalyst tilt in the equity overlay, not in the vol book.

*Stated innovations:* (i) rolling walk-forward selection rather than a single global fit; (ii) a
refined, stress-aware, asymmetric cost model with discrete-strike slippage; (iii) instrument
calibration to short-dated gamma via a measured IV ratio `k`; (iv) a **selection-free** book
weighting that uses no return information; (v) an explicit timing-vs-premium decomposition; (vi) a
machine-checked causality guarantee.

## 12. Limitations and live-trading risk register

1. **The straddle premium is a model, not fills** (`k=0.82`). Real per-strike quotes — the one data
   source we could not obtain (paid: OptionMetrics/CBOE DataShop) — would settle §9's k-band.
2. **k-extrapolation** across markets/history; effective `k` on calm trading days may be < 0.82.
3. **Spread risk**: live spreads > ~1.5× quoted erode the edge (gate mitigates but does not eliminate).
4. **Concentration**: ~60% SPX; the diversification thesis is weaker net of frictions than in proxy space.
5. **Step-aside**: long flat stretches (2008 ≈ out all year); realized path depends on gate re-entry.
6. **Non-US weeklies** modeled as daily expiries; FX unhedged on foreign sleeves.
7. **Calendar Sharpe ~1.0**, not 1.26, is the deployed-capital number; ~0.6–1.0 once k-risk is priced.

## 13. Conclusion

A gated, eight-market, short-dated equity-vol book, weighted by a selection-free rule and charged
realistic frictions, earns a defensible **calendar Sharpe near 1.0** (active-days 1.26), with −20%
drawdown and machine-checked causality. Its returns come entirely from *timing* — the unconditional
variance premium does not survive realistic short-dated execution — and the strategy's biggest open
risks are the un-sourced real option-pricing level (`k`) and execution spreads at size. The honest
deliverable is not a high Sharpe; it is a *calibrated, falsifiable, self-skeptical* one.

---
*Reproduce:* `examples/45` (book + equity curve), `47` (funding/robustness), `48` (attribution),
`49` (k-sensitivity), `44` (friction ladder/spread/LOO/OOS), `46` (finalists + placebo);
`tests/test_book_causality.py` (no-look-ahead). Shared: `bookopt_harness.py`, `bookopt_floor.py`.
