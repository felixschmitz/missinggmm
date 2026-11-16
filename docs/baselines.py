"""Comparison estimators for a regression with one missing regressor.

Each function takes the outcome `y` and the regressors `X`, with `NaN` marking the
missing values of one column, and returns coefficients and standard errors ordered like
the columns of `X`.
"""

import numpy as np

Estimates = tuple[np.ndarray, np.ndarray]


def fit_complete_case(y: np.ndarray, X: np.ndarray) -> Estimates:  # noqa: N803
    """Estimate by OLS on the rows without a missing regressor."""
    is_complete = ~np.isnan(X).any(axis=1)
    return _fit_weighted_least_squares(
        y[is_complete], X[is_complete], np.ones(is_complete.sum())
    )


def fit_dummy_variable(y: np.ndarray, X: np.ndarray) -> Estimates:  # noqa: N803
    """Estimate by OLS with the missing regressor set to zero and a missing indicator.

    The coefficient of the indicator is dropped from the returned estimates.
    """
    is_missing = np.isnan(X).any(axis=1)
    design = np.column_stack([np.nan_to_num(X), is_missing.astype(float)])
    params, bse = _fit_weighted_least_squares(y, design, np.ones(y.shape[0]))
    return params[:-1], bse[:-1]


def fit_dagenais(y: np.ndarray, X: np.ndarray) -> Estimates:  # noqa: N803
    r"""Estimate by FGLS after filling the missing regressor with its projection.

    The filled-in rows have error variance $\sigma_\epsilon^2 + \alpha^2 \sigma_v^2$
    instead of $\sigma_\epsilon^2$, with $\alpha$ the coefficient of the missing
    regressor and $\sigma_v^2$ the residual variance of the projection. The estimates
    are weighted least squares with the inverse of these variances as per-observation
    weights, where $\alpha$ comes from OLS on the filled-in sample and
    $\sigma_\epsilon^2$ from the complete rows. The standard errors treat the weights as
    known.
    """
    is_missing = np.isnan(X).any(axis=1)
    missing_column = int(np.argmax(np.isnan(X).any(axis=0)))
    z = np.delete(X, missing_column, axis=1)

    projection, _, projection_variance = _fit_least_squares(
        X[~is_missing, missing_column], z[~is_missing], np.ones((~is_missing).sum())
    )
    filled = X.copy()
    filled[is_missing, missing_column] = z[is_missing] @ projection

    _, _, error_variance = _fit_least_squares(
        y[~is_missing], X[~is_missing], np.ones((~is_missing).sum())
    )
    first_stage, _, _ = _fit_least_squares(y, filled, np.ones(y.shape[0]))

    variance = np.where(
        is_missing,
        error_variance + first_stage[missing_column] ** 2 * projection_variance,
        error_variance,
    )
    return _fit_weighted_least_squares(y, filled, 1 / variance)


def _fit_weighted_least_squares(
    y: np.ndarray, design: np.ndarray, weights: np.ndarray
) -> Estimates:
    params, cov_unscaled, error_variance = _fit_least_squares(y, design, weights)
    return params, np.sqrt(error_variance * np.diag(cov_unscaled))


def _fit_least_squares(
    y: np.ndarray, design: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return coefficients, `(X'WX)^{-1}` and the weighted residual variance."""
    weighted_design = design * weights[:, None]
    bread = np.linalg.inv(design.T @ weighted_design)
    params = bread @ (weighted_design.T @ y)
    residual = y - design @ params
    error_variance = float(
        (weights * residual**2).sum() / (design.shape[0] - design.shape[1])
    )
    return params, bread, error_variance
