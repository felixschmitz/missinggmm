"""Baseline estimators for linear regressions with one missing regressor."""

from dataclasses import dataclass

import numpy as np

from missinggmm.estimation import (
    _convert_to_float_array,
    _fail_if_inputs_are_invalid,
    _fail_if_regressors_are_collinear,
    _GMMData,
)
from missinggmm.results import Results


def fit_complete_case(y: np.ndarray, X: np.ndarray) -> Results:  # noqa: N803
    """Estimate by OLS on the rows without a missing regressor.

    Args:
        y: Outcome of shape `(n,)`, without missing values.
        X: Regressors of shape `(n, k)`. Exactly one column contains `NaN` to mark the
            missing values.

    Returns:
        Coefficients, standard errors and covariance, ordered like the columns of `X`.
        `nobs` is the number of rows in the full sample.

    Raises:
        TypeError: If `y` or `X` is not numeric.
        ValueError: If the inputs violate the requirements of `missinggmm.fit`.

    """
    sample = _prepare_sample(y, X)
    is_complete = ~sample.is_missing
    fit = _fit_weighted_least_squares(
        sample.y[is_complete], sample.X[is_complete], np.ones(is_complete.sum())
    )
    return _create_results(fit.params, fit.cov, sample.y.shape[0])


def fit_dummy_variable(y: np.ndarray, X: np.ndarray) -> Results:  # noqa: N803
    """Estimate by OLS with the missing regressor set to zero and a missing indicator.

    The coefficient of the indicator is dropped from the returned estimates.

    Args:
        y: Outcome of shape `(n,)`, without missing values.
        X: Regressors of shape `(n, k)`. Exactly one column contains `NaN` to mark the
            missing values.

    Returns:
        Coefficients, standard errors and covariance, ordered like the columns of `X`.
        `nobs` is the number of rows in the full sample.

    Raises:
        TypeError: If `y` or `X` is not numeric.
        ValueError: If the inputs violate the requirements of `missinggmm.fit`.

    """
    sample = _prepare_sample(y, X)
    design = np.column_stack([np.nan_to_num(sample.X), sample.is_missing.astype(float)])
    fit = _fit_weighted_least_squares(sample.y, design, np.ones(sample.y.shape[0]))
    n_columns = sample.X.shape[1]
    return _create_results(
        fit.params[:n_columns], fit.cov[:n_columns, :n_columns], sample.y.shape[0]
    )


def fit_dagenais(y: np.ndarray, X: np.ndarray) -> Results:  # noqa: N803
    r"""Estimate by FGLS after filling the missing regressor with its projection.

    The missing regressor is replaced by its linear projection on the observed
    regressors, fitted on the complete rows. The filled-in rows have error variance
    $\sigma_\epsilon^2 + \alpha^2 \sigma_v^2$ instead of $\sigma_\epsilon^2$, with
    $\alpha$ the coefficient of the missing regressor and $\sigma_v^2$ the residual
    variance of the projection. The estimates are weighted least squares with the
    inverse of these variances as per-observation weights, where $\alpha$ comes from
    OLS on the filled-in sample and $\sigma_\epsilon^2$ from the complete rows. The
    covariance treats the weights as known.

    Args:
        y: Outcome of shape `(n,)`, without missing values.
        X: Regressors of shape `(n, k)`. Exactly one column contains `NaN` to mark the
            missing values.

    Returns:
        Coefficients, standard errors and covariance, ordered like the columns of `X`.
        `nobs` is the number of rows in the full sample.

    Raises:
        TypeError: If `y` or `X` is not numeric.
        ValueError: If the inputs violate the requirements of `missinggmm.fit`.

    """
    sample = _prepare_sample(y, X)
    is_missing = sample.is_missing
    n_rows = sample.y.shape[0]
    missing_column = sample.missing_column
    z = np.delete(sample.X, missing_column, axis=1)

    projection = _fit_weighted_least_squares(
        sample.X[~is_missing, missing_column],
        z[~is_missing],
        np.ones(n_rows - is_missing.sum()),
    )
    filled = sample.X.copy()
    filled[is_missing, missing_column] = z[is_missing] @ projection.params

    complete_case = _fit_weighted_least_squares(
        sample.y[~is_missing], sample.X[~is_missing], np.ones(n_rows - is_missing.sum())
    )
    first_stage = _fit_weighted_least_squares(sample.y, filled, np.ones(n_rows))

    variance = np.where(
        is_missing,
        complete_case.error_variance
        + first_stage.params[missing_column] ** 2 * projection.error_variance,
        complete_case.error_variance,
    )
    fit = _fit_weighted_least_squares(sample.y, filled, 1 / variance)
    return _create_results(fit.params, fit.cov, n_rows)


@dataclass(frozen=True)
class _Sample:
    """Validated float inputs and the position of the missing regressor."""

    y: np.ndarray
    """Outcome of shape `(n,)`."""

    X: np.ndarray
    """Regressors of shape `(n, k)` with `NaN` in the missing column."""

    missing_column: int
    """Index of the column of `X` with missing values."""

    @property
    def is_missing(self) -> np.ndarray:
        """Whether the regressor is missing, per row."""
        return np.isnan(self.X[:, self.missing_column])


@dataclass(frozen=True)
class _LeastSquaresFit:
    """Weighted least-squares estimates."""

    params: np.ndarray
    """Coefficients."""

    cov: np.ndarray
    """Covariance of `params`, treating the weights as known."""

    error_variance: float
    """Weighted residual sum of squares over the residual degrees of freedom."""


def _prepare_sample(y: np.ndarray, X: np.ndarray) -> _Sample:  # noqa: N803
    y = _convert_to_float_array(y, "y")
    X = _convert_to_float_array(X, "X")  # noqa: N806
    _fail_if_inputs_are_invalid(y, X)
    missing_column = int(np.argmax(np.isnan(X).any(axis=0)))
    instrument_columns = np.flatnonzero(~np.isnan(X).any(axis=0))
    data = _GMMData.from_arrays(y, X[:, instrument_columns], X[:, missing_column])
    _fail_if_regressors_are_collinear(data)
    return _Sample(y=y, X=X, missing_column=missing_column)


def _fit_weighted_least_squares(
    y: np.ndarray, design: np.ndarray, weights: np.ndarray
) -> _LeastSquaresFit:
    """Fit by weighted least squares with one weight per observation."""
    weighted_design = design * weights[:, None]
    bread = np.linalg.inv(design.T @ weighted_design)
    params = bread @ (weighted_design.T @ y)
    residual = y - design @ params
    error_variance = float(
        (weights * residual**2).sum() / (design.shape[0] - design.shape[1])
    )
    return _LeastSquaresFit(
        params=params, cov=error_variance * bread, error_variance=error_variance
    )


def _create_results(params: np.ndarray, cov: np.ndarray, nobs: int) -> Results:
    return Results(
        params=params,
        bse=np.sqrt(np.diag(cov)),
        cov=cov,
        nobs=nobs,
        converged=True,
        n_iterations=0,
    )
