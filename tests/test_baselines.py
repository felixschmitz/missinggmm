from collections.abc import Callable

import numpy as np
import pytest
import statsmodels.api as sm
from test_fit import TRUE_PARAMS, Sample, _simulate

from missinggmm import baselines
from missinggmm.results import Results

Estimator = Callable[[np.ndarray, np.ndarray], Results]

ESTIMATORS = [
    baselines.fit_complete_case,
    baselines.fit_dummy_variable,
    baselines.fit_dagenais,
]


@pytest.fixture
def sample() -> Sample:
    return _simulate(n_observations=2_000, seed=0)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_baseline_returns_one_parameter_per_column_of_x(
    estimator: Estimator, sample: Sample
) -> None:
    y, X = sample
    assert estimator(y, X).params.shape == (3,)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_baseline_counts_incomplete_rows_in_nobs(
    estimator: Estimator, sample: Sample
) -> None:
    y, X = sample
    assert estimator(y, X).nobs == 2_000


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_baseline_bse_is_root_of_covariance_diagonal(
    estimator: Estimator, sample: Sample
) -> None:
    y, X = sample
    result = estimator(y, X)
    np.testing.assert_allclose(result.bse, np.sqrt(np.diag(result.cov)), atol=1e-14)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_baseline_rejects_x_without_missing_values(
    estimator: Estimator, sample: Sample
) -> None:
    y, X = sample
    with pytest.raises(ValueError, match="X has no missing values"):
        estimator(y, np.nan_to_num(X))


@pytest.mark.parametrize(
    "estimator", [baselines.fit_complete_case, baselines.fit_dagenais]
)
@pytest.mark.parametrize("missing_column", [0, 1, 2])
def test_consistent_baseline_orders_params_like_columns_of_x(
    estimator: Estimator, missing_column: int
) -> None:
    y, X = _simulate(n_observations=20_000, seed=1, missing_column=missing_column)
    order = {0: [0, 1, 2], 1: [1, 0, 2], 2: [1, 2, 0]}[missing_column]
    expected = TRUE_PARAMS[order]
    np.testing.assert_allclose(estimator(y, X).params, expected, atol=0.1)


@pytest.mark.parametrize("missing_column", [0, 1, 2])
def test_dummy_variable_orders_params_like_columns_of_x(missing_column: int) -> None:
    y, X = _simulate(n_observations=2_000, seed=1, missing_column=missing_column)
    is_missing = np.isnan(X).any(axis=1)
    design = np.column_stack([np.nan_to_num(X), is_missing.astype(float)])
    expected = sm.OLS(y, design).fit()
    result = baselines.fit_dummy_variable(y, X)
    np.testing.assert_allclose(result.params, expected.params[:3], atol=1e-10)


def test_complete_case_matches_ols_on_complete_rows(sample: Sample) -> None:
    y, X = sample
    is_complete = ~np.isnan(X).any(axis=1)
    expected = sm.OLS(y[is_complete], X[is_complete]).fit()
    result = baselines.fit_complete_case(y, X)
    np.testing.assert_allclose(result.params, expected.params, atol=1e-10)


def test_complete_case_bse_matches_ols_on_complete_rows(sample: Sample) -> None:
    y, X = sample
    is_complete = ~np.isnan(X).any(axis=1)
    expected = sm.OLS(y[is_complete], X[is_complete]).fit()
    result = baselines.fit_complete_case(y, X)
    np.testing.assert_allclose(result.bse, expected.bse, atol=1e-10)


def test_dummy_variable_matches_ols_with_missing_indicator(sample: Sample) -> None:
    y, X = sample
    is_missing = np.isnan(X[:, 0])
    design = np.column_stack([np.nan_to_num(X), is_missing.astype(float)])
    expected = sm.OLS(y, design).fit()
    result = baselines.fit_dummy_variable(y, X)
    np.testing.assert_allclose(result.params, expected.params[:3], atol=1e-10)


def test_dummy_variable_bse_matches_ols_with_missing_indicator(
    sample: Sample,
) -> None:
    y, X = sample
    is_missing = np.isnan(X[:, 0])
    design = np.column_stack([np.nan_to_num(X), is_missing.astype(float)])
    expected = sm.OLS(y, design).fit()
    result = baselines.fit_dummy_variable(y, X)
    np.testing.assert_allclose(result.bse, expected.bse[:3], atol=1e-10)


def test_dagenais_matches_wls_on_filled_data_with_variance_weights(
    sample: Sample,
) -> None:
    y, X = sample
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
    result = baselines.fit_dagenais(y, X)
    np.testing.assert_allclose(result.params, expected.params, atol=1e-8)
