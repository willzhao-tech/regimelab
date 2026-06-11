# -*- coding: utf-8 -*-
"""
APPROACH: causal_riskweight
Weight each sleeve by its TRAILING realized performance (recomputed daily, then
SHIFTED so today's weight only uses data up to yesterday). Markets that earn
their keep get more weight automatically; HSI gets down-weighted BY THE DATA,
never by our hindsight.

Cardinal rule compliance:
  - We NEVER pick which markets to include using full-sample results.
  - All 8 markets always enter the book.
  - The per-sleeve weight on day t is a function of trailing PnL up to day t-1
    ONLY (rolling window, then .shift(1)). No future info.
  - We report baseline (equal-risk) vs each causal weighting scheme, and an
    explicit argument why this is not post-hoc selection.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import bookopt_harness as H

SQ = H.SQ

# ----------------------------------------------------------------------------
# 1) Build the 8 raw sleeve PnLs (gated blend), exactly as the baseline does.
# ----------------------------------------------------------------------------
sleeves = {}
for name, *_ in H.PAIRS:
    blend, static = H.market(name)       # no gate, mult=1.0 -> honest L4 baseline
    if blend is not None:
        sleeves[name] = blend
print("sleeves built:", list(sleeves.keys()))

# ----------------------------------------------------------------------------
# 2) Baseline: equal-risk book (weights=None) -> must reproduce ~0.84.
# ----------------------------------------------------------------------------
bk_base = H.book_of(sleeves)
s_base = H.sharpe(bk_base)
dd_base = H.maxdd(bk_base)
h1_base, h2_base = H.split_halves(bk_base)
print(f"\nBASELINE equal-risk   Sharpe={s_base:.3f}  maxDD={dd_base:.3f}  "
      f"halves=({h1_base:.2f},{h2_base:.2f})  n={len(bk_base)}")

# ----------------------------------------------------------------------------
# 3) Causal weighting helpers.
#    For each sleeve we compute a TRAILING statistic over a rolling window,
#    then .shift(1) so the weight used on day t is decided from data up to t-1.
#    We floor at 0 (never short a sleeve) and add a small epsilon so a sleeve
#    is never permanently zeroed; weights are renormalized inside book_of.
# ----------------------------------------------------------------------------
def trailing_sharpe_weight(pnl, win=252, floor=0.0, eps=0.05):
    """Trailing realized Sharpe (annualized), clipped at >=0, shifted causal."""
    mu = pnl.rolling(win).mean()
    sd = pnl.rolling(win).std()
    sh = (mu / sd * SQ).replace([np.inf, -np.inf], np.nan)
    w = sh.clip(lower=floor) + eps          # >=eps, more for better trailing Sharpe
    return w.shift(1)                        # CAUSAL: decided from data up to t-1

def trailing_invvol_weight(pnl, win=252, eps=1e-9):
    """Inverse trailing vol, shifted causal (risk-parity-ish, no return info)."""
    sd = pnl.rolling(win).std()
    w = (1.0 / (sd + eps))
    return w.shift(1)

def trailing_coverage_weight(pnl, win=252, eps=0.05):
    """Trailing mean PnL clipped at >=0 (does the sleeve cover its costs?),
       normalized by trailing vol so it is unit-free, shifted causal."""
    mu = pnl.rolling(win).mean()
    sd = pnl.rolling(win).std()
    w = (mu / sd).clip(lower=0.0) + eps
    return w.shift(1)

def build_weights(maker, **kw):
    return {k: maker(v, **kw) for k, v in sleeves.items()}

def evaluate(label, weights):
    bk = H.book_of(sleeves, weights=weights)
    s = H.sharpe(bk); dd = H.maxdd(bk); h1, h2 = H.split_halves(bk)
    print(f"{label:34s} Sharpe={s:.3f}  maxDD={dd:.3f}  "
          f"halves=({h1:.2f},{h2:.2f})  n={len(bk)}")
    return dict(label=label, sharpe=s, maxdd=dd, h1=h1, h2=h2, bk=bk)

print()
results = []
# Trailing-Sharpe weighting at a few windows (all causal):
for win in (126, 252, 504):
    results.append(evaluate(f"trailing-Sharpe w={win}",
                            build_weights(trailing_sharpe_weight, win=win)))
# Inverse trailing vol (pure risk-parity refinement, no return info):
for win in (126, 252):
    results.append(evaluate(f"inverse-trailing-vol w={win}",
                            build_weights(trailing_invvol_weight, win=win)))
# Cost-coverage weighting:
for win in (252, 504):
    results.append(evaluate(f"trailing-coverage w={win}",
                            build_weights(trailing_coverage_weight, win=win)))

# ----------------------------------------------------------------------------
# 4) Causality self-check.
#    Prove the weight on day t cannot see PnL on/after day t. We rebuild the
#    trailing-Sharpe weight from a TRUNCATED PnL (NaN'd from a cutoff onward)
#    and confirm the weight up to the cutoff is byte-identical to the full one.
#    If any future value leaked into an earlier weight, these would differ.
# ----------------------------------------------------------------------------
def causality_audit(win=252):
    name = "HSI" if "HSI" in sleeves else list(sleeves)[0]
    pnl = sleeves[name]
    full = trailing_sharpe_weight(pnl, win=win)
    cut = pnl.index[int(len(pnl) * 0.6)]
    truncated = pnl.copy()
    truncated.loc[cut:] = np.nan               # destroy all info at/after cutoff
    trunc_w = trailing_sharpe_weight(truncated, win=win)
    # weights strictly BEFORE the cutoff must be unchanged by erasing the future
    pre = full.index < cut
    a = full[pre].dropna()
    b = trunc_w.reindex(full.index)[pre].dropna()
    common = a.index.intersection(b.index)
    identical = np.allclose(a.reindex(common).values, b.reindex(common).values)
    print(f"\nCAUSALITY AUDIT ({name}, win={win}): "
          f"pre-cutoff weights identical after erasing future = {identical} "
          f"({len(common)} pts compared)")
    return identical

audit_ok = causality_audit()

# ----------------------------------------------------------------------------
# 5) Pick the best CAUSAL scheme to report. Note: choosing the *family* of
#    weighting (trailing-Sharpe vs inv-vol) by these numbers is itself a mild
#    in-sample choice, so we ALSO report the most theory-default scheme
#    (inverse-trailing-vol, which uses NO return info at all) as the
#    selection-free headline, and flag the best-Sharpe scheme separately.
# ----------------------------------------------------------------------------
best = max(results, key=lambda r: r["sharpe"])
invvol = next(r for r in results if r["label"].startswith("inverse-trailing-vol w=252"))
ts252 = next(r for r in results if r["label"].startswith("trailing-Sharpe w=252"))

print("\n================= SUMMARY vs baseline 0.84 =================")
print(f"baseline equal-risk:            {s_base:.3f}")
print(f"inverse-trailing-vol w=252:     {invvol['sharpe']:.3f}  (no return info -> no selection)")
print(f"trailing-Sharpe w=252:          {ts252['sharpe']:.3f}")
print(f"best causal scheme [{best['label']}]: {best['sharpe']:.3f} "
      f"(maxDD {best['maxdd']:.3f}, halves {best['h1']:.2f}/{best['h2']:.2f})")
print(f"causality audit passed: {audit_ok}")

# Diagnostic: show the realized trailing-Sharpe weight HSI vs SPX get on average,
# to demonstrate the data down-weights HSI without us touching it.
def avg_norm_weight(maker, win=252):
    W = pd.DataFrame(build_weights(maker, win=win)).dropna(how="all")
    Wn = W.div(W.sum(axis=1), axis=0)
    return Wn.mean().sort_values()
print("\nAvg normalized trailing-Sharpe weight per sleeve (data's own verdict):")
print(avg_norm_weight(trailing_sharpe_weight).to_string())
