"""Generalized Method of Moments estimation with missing regressors."""

from missinggmm.estimation import fit
from missinggmm.results import Results

try:
    from missinggmm._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"

__all__ = ["Results", "__version__", "fit"]
