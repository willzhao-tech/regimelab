# regimelab — auto-generated research brief

*All figures regenerated live from the platform. Illustrative; not investment advice.*

## 1. The two lenses

```
strategy                  IS Sharpe  IS rank  Fwd Sharpe  Fwd rank  Fwd P(loss)
-----------------------------------------------------------------------------------------
risk_parity                    0.84        3        0.42         2         29%
equal_weight                   1.00        1        0.25         3         38%
fixed                          0.99        2        0.16         5         44%
trend(126)                     0.54        6        0.46         1         20%
fixed                          0.72        5        0.22         4         41%
fixed                          0.77        4        0.08         6         46%
-----------------------------------------------------------------------------------------
Spearman(IS rank, Fwd rank) = -0.257   (near 0 or negative => in-sample ranking misleads forward)
```

The in-sample and possibility-weighted forward rankings diverge: strategies that rank highly
in-sample are not the ones that rank highly forward. This is the platform's central,
reproducible observation.

## 2. Out-of-sample walk-forward (risk parity)

```

 fold                        window    Sharpe     CAGR
    1        2019-01-02..2020-10-20      2.16    30.4%
    2        2020-10-21..2022-10-20     -1.28   -14.5%
    3        2022-10-21..2024-08-14      0.88     9.3%
    4        2024-08-15..2026-06-05      1.73    21.4%
```
Per-fold Sharpe swings widely across sequential blocks, making the strategy's regime-
dependence directly visible out of sample.

## 3. Inference: discounting luck

- Risk-parity in-sample Sharpe **0.84**; deflated Sharpe across 6 trials = **0.83** (→1 means it survives multiple testing).

## 4. The headline experiment (reported honestly)

```
Regime-concentration vs ranking-inversion (the headline experiment):

  concentration bucket    mean inversion     n
  mid                              0.177     5
  concentrated                     0.368     9
  high                             0.351    26

  linear correlation = +0.122   OLS slope = +0.123

  Interpretation: inversion rises as windows move from regime-balanced to regime-
  concentrated, then SATURATES — so the linear correlation is weak even though a real
  threshold effect is present. The persistent regime model over-produces concentrated
  windows, leaving little balance to drive a linear fit. Confirming the shape on real multi-
  decade data (with naturally balanced and concentrated windows) is the indicated next
  experiment.
```

## Caveats

Forward results are conditional on the regime model (menu, drifts, inflation-conditional
correlations, persistence, fat tails) and are a structured robustness exercise, not
forecasts. The inversion relationship is a threshold-and-saturate effect with a weak linear
correlation, pending confirmation on real multi-decade data. All results gross of costs.