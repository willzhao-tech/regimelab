# -*- coding: utf-8 -*-
"""DESK CONFIG — the single, versioned home of every vol-book knob (the Jane Street
'desk-wide, not per-user' principle: configuration is shared state, changed deliberately,
in git). Consumed by bookopt_harness / bookopt_floor / paper_trade. Changing a value here
is a strategy change: re-run the GAUNTLET (python gauntlet.py) before the tracker uses it."""

# ---- instrument / pricing -----------------------------------------------------------
K = 0.82                  # 1-day IV / 30-day vol-index ratio (MEASURED, SPX VIX1D/VIX, ex 39)
DT = 1.0 / 252.0

# ---- universe: (name, underlying_csv_stem, volindex_csv_stem, base_spread) ----------
PAIRS = [("SPX", "SPX", "VIX", .025), ("NQ", "NQ_F", "VXN", .025), ("EEM", "EEM", "VXEEM", .025),
         ("DAX", "DAX", "VDAX", .04), ("SX5E", "SX5E", "VSTOXX", .04), ("N225", "N225", "JNIV", .04),
         ("HSI", "HSI", "VHSI", .04), ("NIFTY", "NSEI", "INDIAVIX", .04)]

# ---- walk-forward -------------------------------------------------------------------
TRAIN, TEST = 1260, 252   # 5y train -> 1y test, params re-selected per window

# ---- signal grids (established; ex 33-34 — do NOT widen without gauntlet re-run) ----
GRID_A = [(a, b, c) for a in (2., 4., 6.) for b in (0., -2.) for c in (0., 1., 2.)]
GRID_B = [(b1, b2) for b1 in (.8, 1., 1.2) for b2 in (1.3, 1.6, 2.)]

# ---- book construction --------------------------------------------------------------
FLOOR_WIN = 252           # trailing window for invvol weight + cost-coverage gate
TARGET_VOL = 0.10         # causal book-level vol target
LEV_CAP = 4.0

# ---- L4 frictions -------------------------------------------------------------------
STRIKE_OFFSET = 0.00125   # discrete-strike payoff offset (alternating sign)
COST_FLOOR = 0.00015      # 1.5bp of notional per in-market day
EEM_ASSIGN_SPREAD = 0.005 # American-style ETF assignment penalty (added to EEM spread)
MARGIN_FRAC = 0.15        # margin posted as fraction of notional (funding drag base)

# ---- paper-trading operations (kill switches; Optiver doctrine) ---------------------
STALE_DAYS = 14           # calendar days a raw feed may lag before HALT
PNL_BOUND = 0.08          # |daily book return| beyond this = anomaly -> HALT
LIVE_MAXDD_HALT = -0.25   # live drawdown beyond backtest maxDD-with-margin -> operational HALT
LEDGER_START = "2024-01-01"
VOLBOOK_INCEPTION = "2026-06-11"


def as_dict():
    return {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith("_")}
