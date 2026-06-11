# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import sleeveA_harness as H
import numpy as np, pandas as pd

SQ = np.sqrt(252.0)
df, fomc = H.load()

print("="*78)
print("DATA SPAN:", df.index[0].date(), "->", df.index[-1].date(), " rows:", len(df))
print("="*78)

# ---------------------------------------------------------------------------
# 1) Component toggles. Keep ALL other params at harness defaults.
# ---------------------------------------------------------------------------
def run(use_trend, use_voltarget, use_fomc):
    return H.sleeve_a(df, fomc, use_trend=use_trend, use_voltarget=use_voltarget, use_fomc=use_fomc)

variants = {
    "FULL (A)":            run(True,  True,  True),
    "no_trend":            run(False, True,  True),
    "no_voltarget":        run(True,  False, True),
    "no_fomc":             run(True,  True,  False),
    "trend_only":          run(True,  False, False),
    "voltarget_only":      run(False, True,  False),
}

# plain buy & hold NQ (no cost, no leverage) -- raw beta benchmark
ret = df["Close"].pct_change()
bh = ret.dropna()

# Build a results table over the COMMON index of Sleeve A FULL so comparisons are apples-to-apples
rA, posA = variants["FULL (A)"]
common = rA.index

rows = []
def add(name, r):
    r = pd.Series(r).reindex(common).dropna()
    m = H.metrics(r)
    rows.append((name, m["sharpe"], m["cagr"], m["maxdd"], m["calmar"], m["vol"], m["n"]))
    return r

series = {}
for name,(r,p) in variants.items():
    series[name] = add(name, r)
series["buy&hold NQ"] = add("buy&hold NQ", bh)

# vol-matched NQ: scale buy&hold NQ so its realized vol over `common` equals Sleeve A's realized vol.
# This is a SINGLE full-sample scalar -- it is NOT a tradeable strategy, it's the correct
# "same-risk passive long" benchmark for the beta-attribution regression (a constant multiplier
# does not change Sharpe, only vol). We report it explicitly as a static benchmark.
rA_c = series["FULL (A)"]
bh_c = bh.reindex(common).dropna()
# align
idx = rA_c.index.intersection(bh_c.index)
rA_c = rA_c.reindex(idx); bh_c = bh_c.reindex(idx)
scale = rA_c.std() / bh_c.std()
volmatched = bh_c * scale
series["vol-matched NQ"] = volmatched
m = H.metrics(volmatched)
rows.append(("vol-matched NQ", m["sharpe"], m["cagr"], m["maxdd"], m["calmar"], m["vol"], m["n"]))

print("\nCOMPONENT / BENCHMARK TABLE (all on Sleeve-A common index)")
print(f"{'variant':<16}{'Sharpe':>8}{'CAGR':>9}{'MaxDD':>9}{'Calmar':>8}{'Vol':>8}{'n':>7}")
for name,sh,cg,dd,cal,vol,n in rows:
    print(f"{name:<16}{sh:>8.3f}{cg*100:>8.2f}%{dd*100:>8.1f}%{cal:>8.2f}{vol*100:>7.1f}%{n:>7}")

# ---------------------------------------------------------------------------
# 2) OLS: Sleeve A daily returns ~ alpha + beta * vol-matched-NQ daily returns
#    (regressing on vol-matched NQ vs raw NQ gives IDENTICAL alpha-t and R^2;
#     beta just rescales. We report on raw NQ for an interpretable beta, AND
#     confirm alpha is invariant. Alpha annualized = alpha_daily * 252.)
# ---------------------------------------------------------------------------
def ols(y, x):
    y = pd.Series(y); x = pd.Series(x)
    j = y.index.intersection(x.index)
    y = y.reindex(j).values; x = x.reindex(j).values
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n,k = X.shape
    dof = n-k
    s2 = (resid@resid)/dof
    cov = s2*np.linalg.inv(X.T@X)
    se = np.sqrt(np.diag(cov))
    tstat = beta/se
    ss_tot = ((y-y.mean())**2).sum()
    ss_res = (resid@resid)
    r2 = 1 - ss_res/ss_tot
    return dict(alpha=beta[0], beta=beta[1], se_alpha=se[0], se_beta=se[1],
                t_alpha=tstat[0], t_beta=tstat[1], r2=r2, n=n,
                ann_alpha=beta[0]*252.0)

print("\n" + "="*78)
print("OLS REGRESSION: Sleeve A (FULL) daily ret ~ alpha + beta * benchmark daily ret")
print("="*78)
for bench_name in ["buy&hold NQ", "vol-matched NQ"]:
    res = ols(series["FULL (A)"], series[bench_name])
    print(f"\n  benchmark = {bench_name}")
    print(f"    beta            = {res['beta']:.4f}   (t={res['t_beta']:.1f})")
    print(f"    alpha (daily)   = {res['alpha']*1e4:.3f} bp/day  (t={res['t_alpha']:.2f})")
    print(f"    alpha (annual)  = {res['ann_alpha']*100:.2f}% / yr")
    print(f"    R^2             = {res['r2']:.3f}")
    print(f"    n               = {res['n']}")
    sig = "SIGNIFICANT (|t|>1.96)" if abs(res['t_alpha'])>1.96 else "NOT significant"
    print(f"    alpha sig?      = {sig}")

# ---------------------------------------------------------------------------
# 3) Sharpe decomposition: how much Sharpe does each removed component cost?
#    Also compare Sleeve A Sharpe vs vol-matched NQ Sharpe (= raw NQ Sharpe).
# ---------------------------------------------------------------------------
print("\n" + "="*78)
print("MARGINAL COMPONENT CONTRIBUTION (Sharpe delta vs FULL)")
print("="*78)
shA = H.metrics(series["FULL (A)"])["sharpe"]
sh_bh = H.metrics(bh.reindex(common).dropna())["sharpe"]
print(f"  Sleeve A FULL Sharpe        = {shA:.3f}")
print(f"  buy&hold NQ Sharpe          = {sh_bh:.3f}  (vol-matched NQ has the SAME Sharpe)")
print(f"  => Sharpe lift over passive  = {shA - sh_bh:+.3f}")
for name in ["no_trend","no_voltarget","no_fomc"]:
    sh = H.metrics(series[name])["sharpe"]
    print(f"  remove {name:<13} Sharpe = {sh:.3f}   delta vs FULL = {sh-shA:+.3f}")

# ---------------------------------------------------------------------------
# 4) Correlation of Sleeve A with NQ (how much is it just long-NQ?)
# ---------------------------------------------------------------------------
j = series["FULL (A)"].index.intersection(bh.index)
corr = np.corrcoef(series["FULL (A)"].reindex(j).values, bh.reindex(j).values)[0,1]
print(f"\n  corr(Sleeve A, NQ daily)    = {corr:.3f}   (R^2 = {corr**2:.3f})")
avg_pos = posA.reindex(common).mean()
frac_invested = (posA.reindex(common) > 0).mean()
print(f"  avg position (exposure)     = {avg_pos:.3f}")
print(f"  frac of days net-long       = {frac_invested:.3f}")
