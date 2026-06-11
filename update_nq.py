"""Update the NQ (Nasdaq-100 future) daily OHLCV CSV and the VXN vol index.
Thin wrapper over the shared pipeline in market_data.py. VXN is updated here too
because the vol-arb paper signal (paper_trade.py) needs both, on the same US-close clock.

    python update_nq.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from market_data import update_one, _report

if __name__ == "__main__":
    _report("NQ", update_one("NQ"))
    for extra in ("VXN", "VIX1D", "VIX9D"):     # vol-index inputs for the vol-arb signal/validation
        try:
            _report(extra, update_one(extra))
        except Exception as e:                  # an extra's failure must not block the NQ update
            print(f"[{extra}] update failed ({type(e).__name__}): {e}")
