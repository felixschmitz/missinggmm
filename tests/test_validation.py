import numpy as np
import pytest
import statsmodels.api as sm
from test_fit import TRUE_PARAMS, Sample, _simulate

import missinggmm

Estimates = tuple[np.ndarray, np.ndarray]


@pytest.fixture
def large_sample() -> Sample:
    return _simulate(n_observations=20_000, seed=1)


def test_complete_case_baseline_matches_ols_on_complete_rows(
    large_sample: Sample,
) -> None:
    y, X = large_sample
    is_complete = ~np.isnan(X).any(axis=1)
    expected = sm.OLS(y[is_complete], X[is_complete]).fit()
    params, bse = _fit_complete_case(y, X)
    np.testing.assert_allclose(
        np.concatenate([params, bse]),
        np.concatenate([expected.params, expected.bse]),
        atol=1e-10,
    )


def test_dummy_variable_baseline_matches_ols_with_missing_indicator(
    large_sample: Sample,
) -> None:
    y, X = large_sample
    is_missing = np.isnan(X[:, 0])
    design = np.column_stack([np.nan_to_num(X), is_missing.astype(float)])
    expected = sm.OLS(y, design).fit()
    params, bse = _fit_dummy_variable(y, X)
    np.testing.assert_allclose(
        np.concatenate([params, bse]),
        np.concatenate([expected.params[:3], expected.bse[:3]]),
        atol=1e-10,
    )


def test_dagenais_baseline_matches_wls_on_filled_data_with_variance_weights(
    large_sample: Sample,
) -> None:
    y, X = large_sample
    is_missing = np.isnan(X[:, 0])
    z = X[:, 1:]
    projection = sm.OLS(X[~is_missing, 0], z[~is_missing]).fit()
    filled = X.copy()
    filled[is_missing, 0] = z[is_missing] @ projection.params
    first_stage = sm.OLS(y, filled).fit()
    error_variance = sm.OLS(y[~is_missing], X[~is_missing]).fit().scale
    variance = np.where(
        is_missing,
        error_variance + first_stage.params[0] ** 2 * projection.scale,
        error_variance,
    )
    expected = sm.WLS(y, filled, weights=1 / variance).fit()
    params, _ = _fit_dagenais(y, X)
    np.testing.assert_allclose(params, expected.params, atol=1e-8)


@pytest.mark.parametrize("missing_column", [0, 1, 2])
def test_gmm_recovers_true_params_for_any_missing_column(missing_column: int) -> None:
    y, X = _simulate(n_observations=20_000, seed=1, missing_column=missing_column)
    order = {0: [0, 1, 2], 1: [1, 0, 2], 2: [1, 2, 0]}[missing_column]
    np.testing.assert_allclose(
        missinggmm.fit(y, X).params, TRUE_PARAMS[order], atol=0.1
    )


def test_gmm_standard_error_of_missing_regressor_is_below_complete_case(
    large_sample: Sample,
) -> None:
    y, X = large_sample
    _, bse_complete_case = _fit_complete_case(y, X)
    assert missinggmm.fit(y, X).bse[0] < bse_complete_case[0]


def test_gmm_is_unbiased_where_dummy_variable_method_is_biased(
    large_sample: Sample,
) -> None:
    y, X = large_sample
    params_dummy, _ = _fit_dummy_variable(y, X)
    gmm_error = np.abs(missinggmm.fit(y, X).params - TRUE_PARAMS)
    dummy_error = np.abs(params_dummy - TRUE_PARAMS)
    assert gmm_error.max() < 0.1 < dummy_error.max()


def _fit_complete_case(y: np.ndarray, X: np.ndarray) -> Estimates:  # noqa: N803
    """Estimate by OLS on the rows without a missing regressor."""
    is_complete = ~np.isnan(X).any(axis=1)
    return _fit_weighted_least_squares(
        y[is_complete], X[is_complete], np.ones(is_complete.sum())
    )


def _fit_dummy_variable(y: np.ndarray, X: np.ndarray) -> Estimates:  # noqa: N803
    """Estimate by OLS with the missing regressor set to zero and a missing indicator.

    The coefficient of the indicator is dropped from the returned estimates.
    """
    is_missing = np.isnan(X).any(axis=1)
    design = np.column_stack([np.nan_to_num(X), is_missing.astype(float)])
    params, bse = _fit_weighted_least_squares(y, design, np.ones(y.shape[0]))
    return params[:-1], bse[:-1]


def _fit_dagenais(y: np.ndarray, X: np.ndarray) -> Estimates:  # noqa: N803
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
