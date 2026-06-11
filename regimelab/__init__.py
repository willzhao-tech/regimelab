"""
regimelab
=========

A research platform for regime-conditional evaluation of cross-asset allocation
strategies. See README.md for the architecture.

Layers (each a subpackage):
    regimelab.data         pluggable data ingestion -> Panel
    regimelab.strategies   allocation-rule registry + the vol-targeted engine
    regimelab.regime       regime identification (macro + catalysts) + forward sim
    regimelab.evaluation   in-sample + forward lenses, inference, the inversion study
    regimelab.reporting    regenerate papers / tables / dashboards

Shared types:
    regimelab.panel.Panel  the object that flows between layers
"""

from __future__ import annotations

from .panel import Panel, annualization_factor
from . import data

__version__ = "0.1.0"

# strategies / regime / evaluation / reporting are imported lazily by name to
# keep `import regimelab` light and avoid hard failures while layers are built.
__all__ = ["Panel", "annualization_factor", "data", "__version__"]


def __getattr__(name: str):
    # PEP 562 lazy submodule loading: `regimelab.strategies` etc. resolve on first use.
    if name in {"strategies", "regime", "evaluation", "reporting"}:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
