"""Results container returned by the estimators."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Results:
    """Estimates of a linear regression with one missing regressor."""

    params: np.ndarray
    """Coefficients, ordered like the columns of `X`."""

    bse: np.ndarray
    """Standard errors of `params`."""

    cov: np.ndarray
    """Estimated covariance matrix of `params`."""

    nobs: int
    """Number of observations, including rows with a missing regressor."""

    converged: bool
    """Whether the iterations reached the tolerance."""

    n_iterations: int
    """Number of iterations performed."""

    def summary(self) -> str:
        """Format coefficients, standard errors and z-statistics as a table."""
        lines = [
            f"Observations: {self.nobs}",
            f"Converged: {self.converged} after {self.n_iterations} iterations",
            f"{'':<8}{'coef':>12}{'std err':>12}{'z':>10}",
        ]
        z_statistics = self.params / self.bse
        lines.extend(
            f"{f'x{index}':<8}{coef:>12.4f}{bse:>12.4f}{z_statistic:>10.2f}"
            for index, (coef, bse, z_statistic) in enumerate(
                zip(self.params, self.bse, z_statistics, strict=True)
            )
        )
        return "\n".join(lines)
