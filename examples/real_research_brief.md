# regimelab ¡ª auto-generated research brief

*All figures regenerated live from the platform. Illustrative; not investment advice.*

## 1. The two lenses

```
strategy                  IS Sharpe  IS rank  Fwd Sharpe  Fwd rank  Fwd P(loss)
-----------------------------------------------------------------------------------------
risk_parity                    0.95        5        0.38         1         29%
equal_weight                   1.06        2        0.16         5         42%
fixed                          1.14        1        0.21         3         39%
trend(126)                     0.60        6        0.31         2         29%
fixed                          0.96        3        0.17         4         42%
fixed                          0.95        4        0.10         6         44%
-----------------------------------------------------------------------------------------
Spearman(IS rank, Fwd rank) = -0.429   (near 0 or negative => in-sample ranking misleads forward)
```

The in-sample and possibility-weighted forward rankings diverge: strategies that rank highly
in-sample are not the ones that rank highly forward. This is the platform's central,
reproducible observation.

## 2. Out-of-sample walk-forward (risk parity)

```

 fold                        window    Sharpe     CAGR
    1        2013-07-29..2016-11-24      0.83     8.9%
    2        2016-11-25..2019-12-20      1.48    15.9%
    3        2019-12-23..2023-03-03      0.04     0.5%
    4        2023-03-06..2026-06-09      1.74    20.1%
```
Per-fold Sharpe swings widely across sequential blocks, making the strategy's regime-
dependence directly visible out of sample.

## 3. Inference: discounting luck

- Risk-parity in-sample Sharpe **0.95**; deflated Sharpe across 6 trials = **0.98** (¡ú1 means it survives multiple testing).

## 4. The headline experiment (reported honestly)

```
Regime-concentration vs ranking-inversion (the headline experiment):

  concentration bucket    mean inversion     n
  mid                              0.177     5
  concentrated                     0.400     9
  high                             0.443    26

  linear correlation = +0.251   OLS slope = +0.241

  Interpretation: inversion rises as windows move from regime-balanced to regime-
  concentrated, then SATURATES ¡ª so the linear correlation is weak even though a real
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