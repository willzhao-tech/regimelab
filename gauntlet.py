# -*- coding: utf-8 -*-
"""THE PROMOTION GAUNTLET — one command, pre-registered thresholds, PASS/FAIL verdict.
The prop-shop research-to-production gate (Jane Street replays weeks of data; we replay decades
and attack the statistics). A strategy may enter paper tracking ONLY if every check passes.

Checks (thresholds fixed BEFORE running — do not tune them to pass):
  G1 causality      : future-perturbation invariance (tests/test_book_causality)
  G2 reproducibility: results.json reproduces from a clean rebuild
  G3 data quality   : no HARD corruption in book inputs
  G4 PSR            : P(true Sharpe > 0) >= 0.99 on the calendar book
  G5 bootstrap CI   : 95% block-bootstrap CI lower bound > 0 (calendar)
  G6 multiple tests : book one-sided p < 0.01 AND >=1 sleeve survives BH-FDR q=0.05
  G7 spread stress  : active Sharpe at 1.5x spreads > 0.30
  G8 regime stability: no significant break (p>=0.05) OR both pre/post Sharpe > 0
  G9 placebo        : floor >= 60th pct of random-weight books (coverage gate held)

Run:  python gauntlet.py     -> gauntlet_report.json + runs.jsonl entry + exit code 0/1"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import bookopt_harness as H, bookopt_floor as F, bookopt_stats as S
from bookopt_runlog import log_run, vintage_digest

OUT = H.OUT
checks = {}

def check(name, ok, detail):
    checks[name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<18} {detail}")

print("PROMOTION GAUNTLET — floor book (invvol x coverage, L4 frictions)\n")

# G1 causality ------------------------------------------------------------------------
try:
    from tests.test_book_causality import test_no_lookahead
    test_no_lookahead()
    check("G1 causality", True, "pre-cutoff P&L invariant to future-data corruption")
except AssertionError as e:
    check("G1 causality", False, str(e)[:120])

# G2 reproducibility -------------------------------------------------------------------
try:
    from tests.test_results_reproducible import test_results_reproduce
    test_results_reproduce()
    check("G2 reproducibility", True, "results.json reproduces from clean rebuild")
except AssertionError as e:
    check("G2 reproducibility", False, str(e)[:160])

# G3 data quality ----------------------------------------------------------------------
from data_quality import audit_ohlcv
hard = []
for _, uf, vf, _ in H.PAIRS:
    for stem in {uf, vf}:
        p = os.path.join(OUT, stem + "_all_history.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
            hard += [i for i in audit_ohlcv(df, stem) if i.startswith("HARD")]
check("G3 data quality", not hard, f"{len(hard)} HARD violations in book inputs")

# build once for the statistical checks --------------------------------------------------
H._DATA.clear()
book, sleeves, W = F.build()
allidx = pd.DatetimeIndex(sorted(set().union(*[set(sleeves[k].index) for k in sleeves])))
cal = book.reindex(allidx).fillna(0.0)

# G4 PSR --------------------------------------------------------------------------------
p0, *_ = S.psr(cal, 0.0)
check("G4 PSR", p0 >= 0.99, f"P(SR>0) = {p0:.4f} (need >= 0.99)")

# G5 bootstrap CI ------------------------------------------------------------------------
ci = S.block_bootstrap_sharpe(cal, block=20, n=3000, seed=11)
check("G5 bootstrap CI", ci["lo"] > 0, f"95% CI [{ci['lo']:.2f}, {ci['hi']:.2f}] (need lo > 0)")

# G6 multiple testing --------------------------------------------------------------------
pv_book = S.sharpe_pvalue(cal)
names = list(sleeves)
keep, _crit = S.bh_fdr([S.sharpe_pvalue(sleeves[k]) for k in names], q=0.05)
check("G6 multiple tests", (pv_book < 0.01) and keep.any(),
      f"book p={pv_book:.4f} (<0.01), {int(keep.sum())}/{len(names)} sleeves survive FDR")

# G7 spread stress -----------------------------------------------------------------------
bk15, _, _ = F.build(mult=1.5)
sh15 = S.sharpe_ann(bk15)
check("G7 spread stress", sh15 > 0.30, f"active Sharpe at 1.5x spreads = {sh15:.2f} (need > 0.30)")

# G8 regime stability ---------------------------------------------------------------------
qa = S.quandt_andrews(cal.values, index=cal.index, n_boot=300, seed=3)
ok8 = (qa["p_value"] >= 0.05) or (qa["pre_sharpe"] > 0 and qa["post_sharpe"] > 0)
check("G8 regime stability", ok8,
      f"break p={qa['p_value']:.2f} at {qa['break_date']}; pre {qa['pre_sharpe']:+.2f} / post {qa['post_sharpe']:+.2f}")

# G9 placebo -------------------------------------------------------------------------------
rng = np.random.default_rng(7)
real = S.sharpe_ann(cal); dist = np.empty(400)
for i in range(400):
    Wr = {k: (pd.Series(rng.uniform(0.2, 1.0), index=sleeves[k].index) *
              F.coverage_gate(k).reindex(sleeves[k].index)) for k in sleeves}
    dist[i] = S.sharpe_ann(H.book_of(sleeves, Wr).reindex(allidx).fillna(0.0))
pct = float((dist < real).mean() * 100)
check("G9 placebo", pct >= 60.0, f"floor at {pct:.0f}th pct of random-weight books (need >= 60)")

# verdict ----------------------------------------------------------------------------------
n_pass = sum(1 for c in checks.values() if c["pass"])
verdict = "PROMOTED" if n_pass == len(checks) else "REJECTED"
print(f"\nVERDICT: {verdict}  ({n_pass}/{len(checks)} checks passed)")

stems = sorted({s for _, uf, vf, _ in H.PAIRS for s in (uf, vf)})
report = {"verdict": verdict, "checks": checks,
          "headline": {"sharpe_active": round(S.sharpe_ann(book), 4),
                       "sharpe_calendar": round(S.sharpe_ann(cal), 4)},
          "vintage_digest": vintage_digest(stems)}
with open(os.path.join(OUT, "gauntlet_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)
log_run("gauntlet", {"verdict": verdict, "passed": n_pass, "total": len(checks),
                     "sharpe_calendar": report["headline"]["sharpe_calendar"]},
        params={"TRAIN": H.TRAIN, "TEST": H.TEST, "K": H.K})
print(f"report -> gauntlet_report.json | logged -> runs.jsonl")
sys.exit(0 if verdict == "PROMOTED" else 1)
