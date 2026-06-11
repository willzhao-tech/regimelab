# -*- coding: utf-8 -*-
"""Example 51 - INFERENCE on the floor book: PSR / Deflated Sharpe, bootstrap CI, FDR, placebo.
Holds the honest ~1.0 Sharpe to the bar that dissolved the semiconductor 'alpha'. FULL L4 frictions."""
import sys
sys.path.insert(0, r"C:\Users\ASUS\Desktop\claude doc\market study\regimelab\regimelab\regimelab")
import numpy as np, pandas as pd
import bookopt_harness as H, bookopt_floor as F, bookopt_stats as S

book, sleeves, W = F.build()
allidx = pd.DatetimeIndex(sorted(set().union(*[set(sleeves[k].index) for k in sleeves])))
cal = book.reindex(allidx).fillna(0.0)
SQ = S.SQ

print(f"FLOOR BOOK INFERENCE  {book.index.min().date()}..{book.index.max().date()}")
print(f"  active-days Sharpe {S.sharpe_ann(book):.2f}  | calendar Sharpe {S.sharpe_ann(cal):.2f}\n")

# (1) PSR — is Sharpe>0 robust to skew/fat tails? and is it > 0.5? -----------------------
for tag, r in [("active", book), ("calendar", cal)]:
    p0, sr, g3, g4, T = S.psr(r, 0.0)
    p5, *_ = S.psr(r, 0.5/SQ)
    print(f"(1) PSR [{tag:<8}] P(SR>0)={p0:.3f}  P(SR>0.5)={p5:.3f}  (skew {g3:+.2f}, kurt {g4:.1f}, T={T})")

# (2) Deflated Sharpe — survive the number of strategy trials we ran? --------------------
sleeve_sr_daily = [float(sleeves[k].mean()/sleeves[k].std()) for k in sleeves]   # per-period trial dispersion
print("\n(2) DEFLATED SHARPE (calendar): does the Sharpe beat the best-of-N-trials luck bar?")
nulldisp = S.deflated_sharpe(cal, 50)['trial_sd_ann']
empdisp = S.deflated_sharpe(cal, 50, trial_sharpes=sleeve_sr_daily)['trial_sd_ann']
print(f"    HIGHLY sensitive to trial-SR dispersion: null sampling sd {nulldisp:.2f} vs empirical cross-sleeve {empdisp:.2f} (ann)")
print(f"    {'N trials':>9} | {'null-disp  SR*':>14}{'DSR':>7} | {'emp-disp  SR*':>14}{'DSR':>7}")
for N in (2, 5, 10, 50, 250, 1000):
    dn = S.deflated_sharpe(cal, N); de = S.deflated_sharpe(cal, N, trial_sharpes=sleeve_sr_daily)
    print(f"    {N:>9} | {dn['sr_benchmark_ann']:>13.2f}{dn['dsr']:>7.2f} | {de['sr_benchmark_ann']:>13.2f}{de['dsr']:>7.2f}")
print("    READ: the FLOOR is a-priori (selection-free, not argmax-Sharpe) -> charge it FEW trials")
print("    (low-N rows): under null dispersion it survives comfortably. The OPTIMIZED/selected book")
print("    (composed/riskweight, chosen by maximizing) earns the heavy high-N charge -> would NOT")
print("    survive. This IS the quantified case for preferring the floor over the optimized book.")

# (3) Block-bootstrap Sharpe CI ---------------------------------------------------------
for tag, r in [("active", book), ("calendar", cal)]:
    ci = S.block_bootstrap_sharpe(r, block=20, n=3000, seed=11)
    print(f"\n(3) bootstrap 95% CI [{tag:<8}] Sharpe [{ci['lo']:.2f}, {ci['hi']:.2f}]  (median {ci['median']:.2f})")

# (4) BH-FDR across the 8 sleeves -------------------------------------------------------
names = list(sleeves); pv = [S.sharpe_pvalue(sleeves[k]) for k in names]
keep, crit = S.bh_fdr(pv, q=0.05)
t_hlz = [S.sharpe_ann(sleeves[k])/SQ*np.sqrt(len(sleeves[k])) for k in names]   # ~ Sharpe t-stat
print("\n(4) PER-SLEEVE multiple-testing (BH-FDR q=0.05, Harvey-Liu-Zhu t>3 hurdle):")
print(f"    {'sleeve':<7}{'Sharpe':>8}{'t':>7}{'p':>9}{'BH-keep':>9}{'t>3':>6}")
for k, p, t, kp in zip(names, pv, t_hlz, keep):
    print(f"    {k:<7}{S.sharpe_ann(sleeves[k]):>8.2f}{t:>7.1f}{p:>9.3f}{('YES' if kp else 'no'):>9}{('YES' if t>3 else 'no'):>6}")
print(f"    BH critical p = {crit:.4f}; {int(keep.sum())}/{len(names)} sleeves survive FDR; "
      f"{sum(t>3 for t in t_hlz)}/{len(names)} clear t>3")

# (5) Random-weight placebo — does invvol weighting beat random (coverage gate held)? -----
rng = np.random.default_rng(7); Nrep = 1500
real = S.sharpe_ann(cal)
def placebo(use_cov):
    out = np.empty(Nrep)
    for i in range(Nrep):
        Wr = {}
        for k in sleeves:
            w = rng.uniform(0.2, 1.0)                       # random STATIC per-sleeve weight (no return info)
            cov = F.coverage_gate(k).reindex(sleeves[k].index) if use_cov else 1.0
            Wr[k] = (pd.Series(w, index=sleeves[k].index) * cov)
        bk = H.book_of(sleeves, Wr)
        out[i] = S.sharpe_ann(bk.reindex(allidx).fillna(0.0))
    return out
for use_cov, lbl in [(True, "random wts x COVERAGE gate"), (False, "random wts, NO coverage")]:
    dist = placebo(use_cov); pct = float((dist < real).mean()*100)
    print(f"\n(5) PLACEBO [{lbl:<26}] floor {real:.2f} at {pct:.0f}th pct of random "
          f"(median {np.median(dist):.2f}, 5-95% [{np.percentile(dist,5):.2f},{np.percentile(dist,95):.2f}])")
print("    -> high pct vs 'NO coverage' = the COVERAGE gate carries it; modest pct vs 'x coverage'")
print("       = invvol adds little beyond the gate + diversification (honest, expected for a floor).")
