"""Generalized Method of Moments estimation with missing regressors."""

try:
    from missinggmm._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"

__all__ = ["__version__"]
