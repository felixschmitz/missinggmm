"""Estimation of linear regressions with one missing regressor."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from missinggmm.results import Results

_MAX_ITERATIONS = 100
_TOLERANCE = 1e-8


def fit(
    y: np.ndarray,
    X: np.ndarray,  # noqa: N803
    method: Literal["gmm"] = "gmm",
) -> Results:
    r"""Estimate $y = X \beta + \epsilon$ when one column of `X` has missing values.

    The GMM estimator of Abrevaya and Donald (2017) uses the rows with a missing
    regressor through the moment conditions implied by the projection of the missing
    regressor on the remaining columns of `X`.

    Args:
        y: Outcome of shape `(n,)`, without missing values.
        X: Regressors of shape `(n, k)`. Exactly one column contains `NaN` to mark the
            missing values; all other columns are fully observed and must include a
            constant if the model has an intercept.
        method: Estimator to use. Only `"gmm"` is available.

    Returns:
        Coefficients, standard errors and covariance, ordered like the columns of `X`.

    Raises:
        TypeError: If `y` or `X` is not numeric.
        ValueError: If any of the following holds:

            - `method` is unknown.
            - `y` or `X` contains infinite values.
            - The regressors are collinear in the rows with an observed regressor, or
              the fully observed regressors are collinear in the other rows.
            - `y` is not one-dimensional or `X` is not two-dimensional.
            - `y` and `X` have different numbers of rows.
            - `y` has missing values.
            - `X` does not have missing values in exactly one column.
            - There are fewer incomplete rows than fully observed regressors, or
              fewer complete rows than moment conditions for the complete rows.

    """
    _fail_if_method_is_unknown(method)
    y = _convert_to_float_array(y, "y")
    X = _convert_to_float_array(X, "X")  # noqa: N806
    _fail_if_inputs_are_invalid(y, X)

    is_missing_column = np.isnan(X).any(axis=0)
    missing_column = int(np.argmax(is_missing_column))
    instrument_columns = np.flatnonzero(~is_missing_column)
    data = _GMMData.from_arrays(y, X[:, instrument_columns], X[:, missing_column])
    _fail_if_regressors_are_collinear(data)

    theta_start = _calculate_complete_case_estimates(data)
    weight = np.linalg.inv(_calculate_moment_covariance(data, theta_start))
    theta, converged, n_iterations = _minimize_criterion(data, weight, theta_start)
    covariance = _calculate_sandwich_covariance(data, weight, theta)

    # Reorder (alpha, beta) from [missing regressor, instruments] to the order of `X`.
    position = np.concatenate([[missing_column], instrument_columns])
    params = np.empty(X.shape[1])
    params[position] = theta[: X.shape[1]]
    cov = np.empty((X.shape[1], X.shape[1]))
    cov[np.ix_(position, position)] = covariance[: X.shape[1], : X.shape[1]]

    return Results(
        params=params,
        bse=np.sqrt(np.diag(cov)),
        cov=cov,
        nobs=y.shape[0],
        converged=converged,
        n_iterations=n_iterations,
    )


@dataclass(frozen=True)
class _GMMData:
    """Observations split into complete and incomplete rows."""

    y_complete: np.ndarray
    """Outcome of the rows with an observed regressor."""

    x_complete: np.ndarray
    """Observed values of the otherwise missing regressor."""

    z_complete: np.ndarray
    """Fully observed regressors of the rows with an observed regressor."""

    y_missing: np.ndarray
    """Outcome of the rows with a missing regressor."""

    z_missing: np.ndarray
    """Fully observed regressors of the rows with a missing regressor."""

    @classmethod
    def from_arrays(cls, y: np.ndarray, z: np.ndarray, x: np.ndarray) -> "_GMMData":
        """Split the sample by whether `x` is observed."""
        is_missing = np.isnan(x)
        return cls(
            y_complete=y[~is_missing],
            x_complete=x[~is_missing],
            z_complete=z[~is_missing],
            y_missing=y[is_missing],
            z_missing=z[is_missing],
        )

    @property
    def n_observations(self) -> int:
        """Total number of rows."""
        return self.y_complete.shape[0] + self.y_missing.shape[0]

    @property
    def w_complete(self) -> np.ndarray:
        """Regressors `(x, z)` of the rows with an observed regressor."""
        return np.column_stack([self.x_complete, self.z_complete])


def _calculate_complete_case_estimates(data: _GMMData) -> np.ndarray:
    r"""Estimate $(\alpha, \beta)$ and $\gamma$ by OLS on the complete rows."""
    lambda_start = np.linalg.solve(
        data.w_complete.T @ data.w_complete, data.w_complete.T @ data.y_complete
    )
    gamma_start = np.linalg.solve(
        data.z_complete.T @ data.z_complete, data.z_complete.T @ data.x_complete
    )
    return np.concatenate([lambda_start, gamma_start])


def _minimize_criterion(
    data: _GMMData, weight: np.ndarray, theta_start: np.ndarray
) -> tuple[np.ndarray, bool, int]:
    r"""Minimize the GMM criterion by Gauss-Newton steps with a fixed weight matrix.

    The parameter vector is $\theta = (\alpha, \beta, \gamma)$ with $\alpha$ the
    coefficient of the missing regressor, $\beta$ the coefficients of the observed
    regressors and $\gamma$ the coefficients of their projection onto the missing
    regressor.
    """
    theta = theta_start
    for iteration in range(1, _MAX_ITERATIONS + 1):
        jacobian = _calculate_jacobian(data, theta)
        step = np.linalg.solve(
            jacobian.T @ weight @ jacobian,
            jacobian.T @ weight @ _calculate_mean_moments(data, theta),
        )
        theta = theta + step
        if np.max(np.abs(step)) < _TOLERANCE:
            return theta, True, iteration
    return theta, False, _MAX_ITERATIONS


def _calculate_sandwich_covariance(
    data: _GMMData, weight: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Calculate the covariance of the estimates for the weight matrix used.

    Uses $(G'WG)^{-1} G'W S W G (G'WG)^{-1} / n$ with $S$ the moment covariance at
    `theta`, which stays valid when $W$ is not the inverse of $S$.
    """
    jacobian = _calculate_jacobian(data, theta)
    moment_covariance = _calculate_moment_covariance(data, theta)
    bread = np.linalg.inv(jacobian.T @ weight @ jacobian)
    meat = jacobian.T @ weight @ moment_covariance @ weight @ jacobian
    return bread @ meat @ bread / data.n_observations


def _calculate_mean_moments(data: _GMMData, theta: np.ndarray) -> np.ndarray:
    """Average the moment conditions over all rows, rows without them contribute 0."""
    return (
        _calculate_moment_contributions(data, theta).sum(axis=0) / data.n_observations
    )


def _calculate_moment_covariance(data: _GMMData, theta: np.ndarray) -> np.ndarray:
    r"""Estimate $S = n^{-1} \sum_i g_i g_i'$, which is block diagonal by row group."""
    contributions = _calculate_moment_contributions(data, theta)
    return contributions.T @ contributions / data.n_observations


def _calculate_moment_contributions(data: _GMMData, theta: np.ndarray) -> np.ndarray:
    """Stack the per-row moments; complete rows first, then incomplete rows."""
    n_beta = data.z_complete.shape[1]
    lambda_current = theta[: n_beta + 1]
    gamma_current = theta[n_beta + 1 :]
    delta_current = lambda_current[0] * gamma_current + lambda_current[1:]

    residual_y = data.y_complete - data.w_complete @ lambda_current
    residual_x = data.x_complete - data.z_complete @ gamma_current
    residual_y_missing = data.y_missing - data.z_missing @ delta_current

    n_complete = data.y_complete.shape[0]
    n_missing = data.y_missing.shape[0]
    return np.vstack(
        [
            np.hstack(
                [
                    data.w_complete * residual_y[:, None],
                    data.z_complete * residual_x[:, None],
                    np.zeros((n_complete, n_beta)),
                ]
            ),
            np.hstack(
                [
                    np.zeros((n_missing, n_beta + 1 + n_beta)),
                    data.z_missing * residual_y_missing[:, None],
                ]
            ),
        ]
    )


def _calculate_jacobian(data: _GMMData, theta: np.ndarray) -> np.ndarray:
    """Calculate minus the derivative of the mean moments with respect to `theta`."""
    n_beta = data.z_complete.shape[1]
    lambda_current = theta[: n_beta + 1]
    gamma_current = theta[n_beta + 1 :]
    n_observations = data.n_observations

    cross_w = data.w_complete.T @ data.w_complete / n_observations
    cross_z = data.z_complete.T @ data.z_complete / n_observations
    cross_z_missing = data.z_missing.T @ data.z_missing / n_observations

    return np.vstack(
        [
            np.hstack([cross_w, np.zeros((n_beta + 1, n_beta))]),
            np.hstack([np.zeros((n_beta, n_beta + 1)), cross_z]),
            np.hstack(
                [
                    (cross_z_missing @ gamma_current)[:, None],
                    cross_z_missing,
                    lambda_current[0] * cross_z_missing,
                ]
            ),
        ]
    )


def _convert_to_float_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be numeric, got dtype {array.dtype}.")
    return array.astype(float)


def _fail_if_method_is_unknown(method: str) -> None:
    if method != "gmm":
        raise ValueError(f"Unknown method {method!r}; only 'gmm' is available.")


def _fail_if_inputs_are_invalid(y: np.ndarray, X: np.ndarray) -> None:  # noqa: N803
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if X.ndim != 2:  # noqa: PLR2004
        raise ValueError("X must be two-dimensional.")
    if y.shape[0] != X.shape[0]:
        raise ValueError("y and X must have the same number of rows.")
    if np.isnan(y).any():
        raise ValueError("y must not contain missing values.")
    _fail_if_infinite(y, "y")
    _fail_if_infinite(X, "X")
    n_missing_columns = int(np.isnan(X).any(axis=0).sum())
    if n_missing_columns == 0:
        raise ValueError(
            "X has no missing values; mark the missing regressor with NaN."
        )
    if n_missing_columns > 1:
        raise ValueError("Only one column of X may contain missing values.")
    n_instruments = X.shape[1] - 1
    n_incomplete = int(np.isnan(X).any(axis=1).sum())
    if n_incomplete < n_instruments:
        raise ValueError(
            f"At least {n_instruments} incomplete rows are needed to estimate the "
            f"moment covariance, got {n_incomplete}."
        )
    if X.shape[0] - n_incomplete < 2 * n_instruments + 1:
        raise ValueError(
            f"At least {2 * n_instruments + 1} complete rows are needed to estimate "
            f"the moment covariance, got {X.shape[0] - n_incomplete}."
        )


def _fail_if_infinite(values: np.ndarray, name: str) -> None:
    if np.isinf(values).any():
        raise ValueError(f"{name} must not contain infinite values.")


def _fail_if_regressors_are_collinear(data: _GMMData) -> None:
    if np.linalg.matrix_rank(data.w_complete) < data.w_complete.shape[1]:
        raise ValueError(
            "The regressors are collinear in the rows with an observed regressor."
        )
    if np.linalg.matrix_rank(data.z_missing) < data.z_missing.shape[1]:
        raise ValueError(
            "The fully observed regressors are collinear in the rows with a missing "
            "regressor."
        )
