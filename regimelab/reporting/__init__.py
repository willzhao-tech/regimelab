"""
regimelab.reporting
====================

The reporting layer regenerates human-readable artifacts — tables, summaries,
and a markdown research brief — directly from the layers below, so every figure
is reproducible from config + cached data rather than hand-transcribed.

It deliberately reports the inversion result AS MEASURED (a threshold-and-
saturate relationship with a weak linear correlation), not as the cleaner law
originally hypothesized. Honest reporting is part of the platform's contract.
"""

from __future__ import annotations

from typing import Optional, Sequence
import textwrap

import numpy as np
import pandas as pd

from ..panel import Panel
from .. import evaluation as ev
from .. import regime as regime_layer


# ---------------------------------------------------------------------------
# table formatting
# ---------------------------------------------------------------------------
def format_comparison(df: pd.DataFrame) -> str:
    """Pretty-print the two-lens comparison table (from evaluation.compare)."""
    lines = []
    lines.append(f"{'strategy':24}{'IS Sharpe':>11}{'IS rank':>9}"
                 f"{'Fwd Sharpe':>12}{'Fwd rank':>10}{'Fwd P(loss)':>13}")
    lines.append("-" * 89)
    for name, r in df.iterrows():
        lines.append(f"{name:24}{r['is_sharpe']:>11.2f}{int(r['is_rank']):>9}"
                     f"{r['fwd_sharpe']:>12.2f}{int(r['fwd_rank']):>10}{r['fwd_ploss']*100:>11.0f}%")
    # rank-correlation footer
    rho = np.corrcoef(df['is_rank'], df['fwd_rank'])[0, 1]
    lines.append("-" * 89)
    lines.append(f"Spearman(IS rank, Fwd rank) = {rho:+.3f}   "
                 f"(near 0 or negative => in-sample ranking misleads forward)")
    return "\n".join(lines)


def format_inversion(result: "ev.InversionResult") -> str:
    """Summarize the inversion study honestly, including the saturation shape."""
    t = result.table.copy()
    t["bucket"] = pd.cut(t["concentration"], bins=[0, 0.25, 0.35, 0.5, 1.0],
                         labels=["balanced", "mid", "concentrated", "high"])
    g = t.groupby("bucket", observed=True)["inversion"].agg(["mean", "count"])
    lines = ["Regime-concentration vs ranking-inversion (the headline experiment):", ""]
    lines.append(f"  {'concentration bucket':22}{'mean inversion':>16}{'n':>6}")
    for b, r in g.iterrows():
        lines.append(f"  {str(b):22}{r['mean']:>16.3f}{int(r['count']):>6}")
    lines.append("")
    lines.append(f"  linear correlation = {result.correlation:+.3f}   OLS slope = {result.slope:+.3f}")
    lines.append("")
    lines.append(textwrap.fill(
        "Interpretation: inversion rises as windows move from regime-balanced to "
        "regime-concentrated, then SATURATES — so the linear correlation is weak even "
        "though a real threshold effect is present. The persistent regime model "
        "over-produces concentrated windows, leaving little balance to drive a linear "
        "fit. Confirming the shape on real multi-decade data (with naturally balanced "
        "and concentrated windows) is the indicated next experiment.", width=92,
        initial_indent="  ", subsequent_indent="  "))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the research brief
# ---------------------------------------------------------------------------
def research_brief(
    panel: Panel,
    model: "regime_layer.RegimeModel",
    instruments: Sequence[str],
    out_path: Optional[str] = None,
    n_paths: int = 600,
    is_start: Optional[str] = None,
) -> str:
    """
    Regenerate a full markdown brief from the platform: two-lens comparison,
    walk-forward, deflated-Sharpe check, and the inversion study — all computed
    live, all reported with their honest caveats. Optionally written to disk.
    """
    specs = [
        ("risk_parity", {}),
        ("equal_weight", {}),
        ("fixed", {"weights_map": {"NQ": 0.4, "A50": 0.2, "US10Y": 0.2, "XAU": 0.2}}),
        ("trend", {"lookback": 126}),
        ("fixed", {"weights_map": {"NQ": 0.6, "US10Y": 0.4}}),
        ("fixed", {"weights_map": {"NQ": 1.0}}),
    ]
    cmp = ev.compare(specs, panel, model, list(instruments),
                     n_paths=n_paths, is_start=is_start,
                     target_vol=0.10, vol_win=63, rebal="M")
    wf = ev.walk_forward("risk_parity", panel, list(instruments),
                         n_folds=4, target_vol=0.10, vol_win=63, rebal="M")
    inv = ev.inversion_study(panel, model, n_trials=40, is_horizon=7,
                             fwd_paths=120, fwd_horizon=5,
                             target_vol=0.10, vol_win=63, rebal="M", seed=3)

    rp_is = cmp.loc["risk_parity", "is_sharpe"]
    dsr = ev.deflated_sharpe(rp_is, n_periods=len(panel.common_dates(instruments)), n_trials=len(specs))

    md = []
    md.append("# regimelab — auto-generated research brief\n")
    md.append("*All figures regenerated live from the platform. Illustrative; not investment advice.*\n")
    md.append("## 1. The two lenses\n")
    md.append("```\n" + format_comparison(cmp) + "\n```\n")
    md.append(textwrap.fill(
        "The in-sample and possibility-weighted forward rankings diverge: strategies "
        "that rank highly in-sample are not the ones that rank highly forward. This is "
        "the platform's central, reproducible observation.", width=92) + "\n")
    md.append("## 2. Out-of-sample walk-forward (risk parity)\n")
    md.append("```\n")
    md.append(f"{'fold':>5}{'window':>30}{'Sharpe':>10}{'CAGR':>9}")
    for f, r in wf.iterrows():
        md.append(f"{f:>5}{r['start']+'..'+r['end']:>30}{r['sharpe']:>10.2f}{r['cagr']*100:>8.1f}%")
    md.append("```")
    md.append(textwrap.fill(
        "Per-fold Sharpe swings widely across sequential blocks, making the strategy's "
        "regime-dependence directly visible out of sample.", width=92) + "\n")
    md.append("## 3. Inference: discounting luck\n")
    md.append(f"- Risk-parity in-sample Sharpe **{rp_is:.2f}**; deflated Sharpe across "
              f"{len(specs)} trials = **{dsr:.2f}** (→1 means it survives multiple testing).\n")
    md.append("## 4. The headline experiment (reported honestly)\n")
    md.append("```\n" + format_inversion(inv) + "\n```\n")
    md.append("## Caveats\n")
    md.append(textwrap.fill(
        "Forward results are conditional on the regime model (menu, drifts, "
        "inflation-conditional correlations, persistence, fat tails) and are a "
        "structured robustness exercise, not forecasts. The inversion relationship is a "
        "threshold-and-saturate effect with a weak linear correlation, pending "
        "confirmation on real multi-decade data. All results gross of costs.", width=92))

    text = "\n".join(md)
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(text)
    return text
