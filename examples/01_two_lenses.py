"""
Example 1 — The two lenses and the ranking inversion.

Runs the platform end to end on the bundled legacy panel:
  1. loads data through the data layer,
  2. evaluates a set of strategies in-sample AND in the possibility-weighted
     forward environment,
  3. prints the comparison table showing the ranking divergence.

Run:  python examples/01_two_lenses.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import regimelab as rl
from regimelab.regime import default_model, base_vol_from_panel
from regimelab import evaluation as ev
from regimelab import reporting as rep

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    panel = rl.data.build_panel([rl.data.LegacyJsonSource(os.path.join(HERE, "legacy_panel.json"))])
    core = ["NQ", "A50", "US10Y", "XAU"]
    model = default_model(core, base_vol_from_panel(panel, core), seed=7)

    specs = [
        ("risk_parity", {}),
        ("equal_weight", {}),
        ("fixed", {"weights_map": {"NQ": 0.4, "A50": 0.2, "US10Y": 0.2, "XAU": 0.2}}),
        ("trend", {"lookback": 126}),
        ("fixed", {"weights_map": {"NQ": 0.6, "US10Y": 0.4}}),   # 60/40
        ("fixed", {"weights_map": {"NQ": 1.0}}),                  # all-equity
    ]
    table = ev.compare(specs, panel, model, core, n_paths=600,
                       is_start="2019-01-01", target_vol=0.10, vol_win=63, rebal="M")
    print(rep.format_comparison(table))


if __name__ == "__main__":
    main()
