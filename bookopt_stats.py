# -*- coding: utf-8 -*-
"""Inference toolkit for the vol-book (ported from the 'Is There Alpha in Semiconductors?' study).
Holds the honest ~1.0 Sharpe to the same bar that dissolved the semiconductor alpha:
 - PSR / Deflated Sharpe (Bailey & Lopez de Prado): discount the Sharpe for skew/kurtosis AND the
   number of strategy trials we ran to find it.
 - block-bootstrap Sharpe confidence interval (autocorrelation + fat-tail robust).
 - Benjamini-Hochberg FDR across the 8 sleeves + Harvey-Liu-Zhu t>3 hurdle.
 - random-weight placebo (does the invvol weighting beat random, holding the coverage gate?).
numpy/scipy only; no statsmodels dependency (the source study hand-rolled these for the same reason)."""
import numpy as np
import pandas as pd
from scipy import stats

SQ = np.sqrt(252.0)
GAMMA = 0.5772156649015329          # Euler-Mascheroni


def _clean(r):
    return np.asarray(pd.Series(r).dropna(), float)


def psr(returns, sr_benchmark=0.0):
    """Probabilistic Sharpe Ratio: P(true per-period SR > sr_benchmark) given sample skew/kurtosis.
    Returns (prob, sr_per_period, skew, kurtosis_pearson, T)."""
    r = _clean(returns); T = len(r)
    sd = r.std(ddof=1)
    if T < 10 or sd == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), T
    sr = r.mean()/sd
    g3 = float(stats.skew(r)); g4 = float(stats.kurtosis(r, fisher=False))  # Pearson (normal=3)
    denom = np.sqrt(max(1e-12, 1 - g3*sr + (g4-1)/4.0*sr**2))
    z = (sr - sr_benchmark)*np.sqrt(T-1)/denom
    return float(stats.norm.cdf(z)), float(sr), g3, g4, T


def deflated_sharpe(returns, n_trials, trial_sharpes=None):
    """DSR: PSR evaluated at the EXPECTED-MAXIMUM SR a researcher would find across n_trials of
    independent random strategies. trial_sharpes (per-period) sets the cross-trial SR variance;
    falls back to the under-null variance 1/(T-1)."""
    r = _clean(returns); T = len(r)
    if trial_sharpes is not None and len(trial_sharpes) > 2:
        v = float(np.var(np.asarray(trial_sharpes, float), ddof=1))
    else:
        v = 1.0/(T-1)
    N = max(int(n_trials), 2)
    emax = np.sqrt(v) * ((1-GAMMA)*stats.norm.ppf(1 - 1.0/N) + GAMMA*stats.norm.ppf(1 - 1.0/(N*np.e)))
    p, sr, *_ = psr(r, sr_benchmark=emax)
    return dict(dsr=p, sr_benchmark_ann=emax*SQ, sr_ann=sr*SQ, n_trials=N, trial_sd_ann=np.sqrt(v)*SQ)


def block_bootstrap_sharpe(returns, block=20, n=2000, seed=0):
    """Moving-block-bootstrap CI on the ANNUALIZED Sharpe (preserves vol clustering)."""
    r = _clean(returns); T = len(r)
    if T < block*3:
        return dict(lo=float("nan"), hi=float("nan"), median=float("nan"))
    rng = np.random.default_rng(seed); nb = int(np.ceil(T/block)); out = np.empty(n)
    starts_all = rng.integers(0, T-block+1, size=(n, nb))
    for i in range(n):
        samp = np.concatenate([r[s:s+block] for s in starts_all[i]])[:T]
        sd = samp.std(ddof=1)
        out[i] = samp.mean()/sd*SQ if sd > 0 else 0.0
    return dict(lo=float(np.percentile(out, 2.5)), hi=float(np.percentile(out, 97.5)),
                median=float(np.percentile(out, 50)), dist=out)


def sharpe_pvalue(returns):
    """One-sided p-value for H0: true Sharpe <= 0  (via PSR at benchmark 0)."""
    p0, *_ = psr(returns, 0.0)
    return float(1.0 - p0) if p0 == p0 else float("nan")


def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg: returns (keep_mask, critical_p). Largest k with p(k) <= (k/m)q."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    thresh = q*np.arange(1, m+1)/m
    below = p[order] <= thresh
    if not below.any():
        return np.zeros(m, bool), 0.0
    kmax = np.where(below)[0].max() + 1
    keep = np.zeros(m, bool); keep[order[:kmax]] = True
    return keep, float(thresh[kmax-1])


def sharpe_ann(returns):
    r = _clean(returns); sd = r.std(ddof=1)
    return float(r.mean()/sd*SQ) if len(r) > 10 and sd > 0 else float("nan")


def _supF(x, lo, hi):
    """sup-Wald F for a single break in the MEAN of x, over candidate splits [lo,hi). Vectorized."""
    T = len(x); cs = np.cumsum(x); SSx = float((x*x).sum()); tot = float(cs[-1])
    t = np.arange(lo, hi)
    c1 = cs[t-1]
    rss1 = SSx - c1*c1/t - (tot-c1)**2/(T-t)
    rss0 = SSx - tot*tot/T
    Fv = (rss0 - rss1)/(rss1/(T-2))
    k = int(np.argmax(Fv))
    return float(Fv[k]), int(t[k])


def quandt_andrews(returns, index=None, trim=0.15, block=20, n_boot=400, seed=0):
    """Quandt-Andrews sup-Wald test for a single structural break in the mean return.
    p-value via block-bootstrap of the constant-mean null. Returns dict with sup_F, break_loc,
    break_date, p_value, and pre/post annualized Sharpe."""
    r = _clean(returns); T = len(r)
    lo, hi = max(2, int(T*trim)), min(T-2, int(T*(1-trim)))
    F0, bi = _supF(r, lo, hi)
    rng = np.random.default_rng(seed); nb = int(np.ceil(T/block)); cnt = 0
    for _ in range(n_boot):
        starts = rng.integers(0, T-block+1, nb)
        samp = np.concatenate([r[s:s+block] for s in starts])[:T]
        Fb, _ = _supF(samp, lo, hi)
        cnt += (Fb >= F0)
    pre, post = r[:bi], r[bi:]
    sh = lambda a: float(a.mean()/a.std(ddof=1)*SQ) if len(a) > 10 and a.std() > 0 else float("nan")
    bd = None
    if index is not None and bi < len(index):
        bd = str(pd.DatetimeIndex(index)[bi].date())
    return dict(sup_F=F0, break_loc=bi, break_date=bd, p_value=float((cnt+1)/(n_boot+1)),
                pre_sharpe=sh(pre), post_sharpe=sh(post), n_pre=len(pre), n_post=len(post))
