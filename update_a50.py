"""Update the A50 (FTSE China A50 future) daily OHLCV CSV. Thin wrapper over the
shared pipeline in market_data.py (dataset "A50", source Investing.com #44486).

    python update_a50.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from market_data import update_one, _report

if __name__ == "__main__":
    _report("A50", update_one("A50"))
