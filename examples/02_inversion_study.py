"""
Example 2 — The headline experiment: regime concentration vs ranking inversion.

Runs the inversion study and prints the honest, threshold-and-saturate result.
This is the reusable experiment that turns the one-off "the 2019-2026 ranking
inverted" observation into a tested (and appropriately qualified) relationship.

Run:  python examples/02_inversion_study.py
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

    result = ev.inversion_study(panel, model, n_trials=45, is_horizon=7,
                                fwd_paths=150, fwd_horizon=5,
                                target_vol=0.10, vol_win=63, rebal="M", seed=3)
    print(rep.format_inversion(result))


if __name__ == "__main__":
    main()
