# regimelab

A research platform for **regime-conditional evaluation of cross-asset allocation strategies**.

`regimelab` is the generalized, extensible foundation distilled from a line of
cross-asset macro research. Where the original work was a set of one-off scripts
on a fixed seven-instrument panel, this package turns each piece into a pluggable
layer so that new data, new strategies, new regime models, and new evaluation
protocols can be added without rewriting anything below them.

## The five layers

```
┌─────────────────────────────────────────────────────────────┐
│  reporting/   regenerate papers, tables, dashboards          │
├─────────────────────────────────────────────────────────────┤
│  evaluation/  in-sample + possibility-weighted forward;      │
│               out-of-sample protocol; multiple-testing       │
│               deflation; the inversion-magnitude finding     │
├─────────────────────────────────────────────────────────────┤
│  strategies/  registry: any allocation rule behind one API,  │
│               run through one vol-targeted engine            │
├─────────────────────────────────────────────────────────────┤
│  regime/      identify regimes from macro + catalyst signals │
│               (the novel core); + the forward simulator      │
├─────────────────────────────────────────────────────────────┤
│  data/        pluggable ingestion: FRED, Fama-French, Stooq, │
│               local files → one common Panel format          │
└─────────────────────────────────────────────────────────────┘
```

Each layer depends only on the layers below it and on a small set of shared
types (`regimelab.panel.Panel`). Swap the data source and everything above still
runs; add a strategy and the whole evaluation stack applies to it for free.

## Install

```bash
pip install -e .                 # core (numpy, pandas)
pip install -e ".[data]"         # + live data backends (fredapi, pandas-datareader, yfinance)
pip install -e ".[full]"         # + reporting (matplotlib) and dev tooling
```

The live data backends are **optional**. The package imports and runs without
them; they are required only when you actually fetch from FRED/Stooq/Yahoo, and
each adapter raises a clear, actionable error if its backend is missing.

## Quick start

```python
import regimelab as rl

# 1. DATA — pull a real multi-decade, multi-asset panel from free sources
#    (requires `pip install ".[data]"` and a free FRED API key)
panel = rl.data.build_panel(
    sources=[
        rl.data.InvestingSource(pairs={"NQ": 8874, "A50": 44486}),  # default source
        rl.data.FredSource(series={"US10Y": "DGS10", "VIX": "VIXCLS"}, api_key="..."),
        rl.data.FamaFrenchSource(factors=["Mkt-RF", "SMB", "HML", "Mom"]),
    ],
    start="1990-01-01",
)

# 2. STRATEGY — pick any registered allocation rule
strat = rl.strategies.get("risk_parity")          # or "equal_weight", "trend", ...

# 3. EVALUATION — both lenses, plus inference
result = rl.evaluation.in_sample(strat, panel, target_vol=0.10)
forward = rl.evaluation.forward(strat, panel, n_paths=2000, horizon=5)

# 4. the headline finding, as a reusable experiment
inv = rl.evaluation.inversion_study(panel)        # regime concentration → ranking inversion
```

## Design principles

- **Pluggable, not hard-coded.** Data sources, strategies, and regime models are
  registered behind interfaces. Adding one is a new file, not an edit to the core.
- **No look-ahead, anywhere.** All volatility/weight estimates use trailing
  windows only; the data layer preserves point-in-time vintages where the source
  provides them (e.g. FRED/ALFRED).
- **Reproducible.** Every result is regenerated from config + cached raw data.
- **Honest about limits.** Forward results are model-conditional; the package
  reports them as a structured robustness exercise, never as forecasts.

## Status

This is a research scaffold under active construction. See `docs/ROADMAP.md`
for what is built versus planned, and `examples/` for runnable demonstrations.
All outputs are illustrative and educational — **not investment advice**.
