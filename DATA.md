# Getting the data

**No market data ships with this repository.** The CSVs the research consumes are fetched from
public sources under those providers' terms — you must fetch your own copies and may not
redistribute them.

## Setup

```powershell
# 1. point the platform at a data directory of your choice
$env:REGIMELAB_DATA_DIR = "D:\regimelab-data"

# 2. (only if you need an HTTP proxy; default is 127.0.0.1:7897, set "" to disable)
$env:REGIMELAB_PROXY = ""

# 3. fetch everything in the registry (full history on first run; incremental after)
python market_data.py
```

`market_data.py` holds the dataset registry (`DATASETS`): 24 instruments — the 8 equity indices,
their 8 volatility indices, plus NQ/A50/EURUSD/US10Y/WTI/XAU/VIX1D/VIX9D. Sources are
Investing.com (default; the pairIds in the registry were empirically verified against reference
series) and Yahoo Finance (fallback; used where split-adjustment matters). Every fetch passes a
quality gate (corrupt rows refuse to commit) and a revision detector (silent vendor history
rewrites are logged loudly).

Event data: `fetch_fomc.py` (FOMC dates from the Fed site), `fetch_mps.py` (Bauer–Swanson
monetary-policy surprises from the SF Fed), `fetch_cpi.py` (CPI release dates).

## Expected layout

All series land as `<STEM>_all_history.csv` (daily `Date,Open,High,Low,Close,Volume`) inside
`REGIMELAB_DATA_DIR`. Generated artifacts (`results.json`, ledgers, dashboards, the SSRN package)
are written to the same directory — keep it outside the repo.

## Reproducing the headline results

```powershell
python -m pytest tests -q        # 24 tests incl. machine-checked no-look-ahead
python build_results.py          # results.json — every number pinned to code commit + data hashes
python gauntlet.py               # the 9-check promotion gate
```

Numbers will differ slightly from the paper if your fetch date differs (more history) — that is
the point of the vintage pinning: `results.json` records exactly which data produced which numbers.

## Note on historical scripts

`examples/01–42` and the various `p0_/p1_/va_/wf_/ls_` scripts are the **research archive** — the
full, honest trail of what was tried (including everything that failed). They carry the original
machine's absolute paths and are kept as documentation of the process, not as supported entry
points. The supported surface is: `bookopt_*`, `volbook_config.py`, `strategy_lab.py`,
`gauntlet.py`, `build_results.py`, `paper_trade.py`, `market_data.py`, and `examples/43+`.
