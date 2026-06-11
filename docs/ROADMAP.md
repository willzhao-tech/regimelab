# regimelab roadmap

## Built (v0.1.0)
- **data/** — pluggable sources (FRED, Fama-French, Stooq, Yahoo, LocalCSV, LegacyJson), lazy backends, Parquet cache, `build_panel`.
- **panel.py** — the shared `Panel` type (returns + prices + macro + events) and helpers.
- **strategies/** — registry + shared look-ahead-free vol-targeted engine; `risk_parity`, `equal_weight`, `fixed`, `trend`, `static_mix`. Reproduces the known in-sample result exactly.
- **regime/** — `RegimeModel`/`default_model` forward simulator (Markov-persistent, inflation-conditional correlation, fat tails) + `RuleBasedIdentifier` (macro/catalyst → regime) + `attach_event_flags`.
- **evaluation/** — `in_sample`, `forward`, `compare`; `oos_split`, `walk_forward`; `deflated_sharpe`, `sharpe_pvalue`; `inversion_study`.
- **reporting/** — `format_comparison`, `format_inversion`, `research_brief`.
- **tests/**, **examples/** — 8 passing checks; two runnable demos.

## Findings demonstrated
1. **Ranking inversion** — in-sample vs forward Spearman ≈ −0.2 on the bundled panel (equal-weight #1→#5; risk-parity #3→#1; trend #6→#2). Reproducible via `examples/01`.
2. **Concentration → inversion (threshold-and-saturate)** — inversion rises from balanced→concentrated windows then saturates; weak linear correlation (~+0.14). Honest, qualified result via `examples/02`.

## Next (the research-grade upgrades)
- **Real data**: wire `InvestingSource` (default) + `FredSource` + `FamaFrenchSource` on a networked machine (free FRED key) to get a genuine multi-decade, multi-market panel — the prerequisite for the inversion finding to be tested out-of-sample on naturally balanced *and* concentrated windows. (`StooqSource` was removed: Stooq is now captcha-walled.)
- **Catalyst layer depth**: add FRED Releases (event dates) + FRBSF Bauer–Swanson (FOMC surprises) ingestion; event-conditional regime identification and an event-driven strategy.
- **Inference**: bootstrap confidence intervals on the inversion slope; White/Hansen SPA across the strategy set.
- **HMM/clustering identifier**: a fitted regime identifier alongside the rule-based baseline.
- **Costs/financing layer**: per-strategy turnover and transaction-cost model (the friction work), so net results are reportable.
