import numpy as np
import pytest
import statsmodels.api as sm

import missinggmm

Sample = tuple[np.ndarray, np.ndarray]
TRUE_PARAMS = np.array([1.0, 0.5, -1.0])


def _simulate(
    n_observations: int,
    seed: int,
    share_missing: float = 0.3,
    missing_column: int = 0,
) -> Sample:
    """Draw (y, X) with an intercept, one instrument and a regressor that is MCAR."""
    rng = np.random.default_rng(seed=seed)
    instrument = rng.standard_normal(n_observations)
    regressor = 0.5 + 0.8 * instrument + rng.standard_normal(n_observations)
    design = np.column_stack([regressor, np.ones(n_observations), instrument])
    outcome = design @ TRUE_PARAMS + rng.standard_normal(n_observations)
    is_missing = rng.random(n_observations) < share_missing
    design_observed = design.copy()
    design_observed[is_missing, 0] = np.nan
    if missing_column != 0:
        order = [1, 0, 2] if missing_column == 1 else [1, 2, 0]
        design_observed = design_observed[:, order]
    return outcome, design_observed


@pytest.fixture
def sample() -> Sample:
    return _simulate(n_observations=2_000, seed=0)


def test_fit_returns_one_parameter_per_column_of_x(sample: Sample) -> None:
    y, X = sample
    assert missinggmm.fit(y, X).params.shape == (3,)


def test_fit_counts_incomplete_rows_in_nobs(
    sample: Sample,
) -> None:
    y, X = sample
    assert missinggmm.fit(y, X).nobs == 2_000


def test_fit_covariance_is_symmetric(sample: Sample) -> None:
    y, X = sample
    cov = missinggmm.fit(y, X).cov
    np.testing.assert_allclose(cov, cov.T, atol=1e-14)


def test_fit_bse_is_root_of_covariance_diagonal(
    sample: Sample,
) -> None:
    y, X = sample
    result = missinggmm.fit(y, X)
    np.testing.assert_allclose(result.bse, np.sqrt(np.diag(result.cov)), atol=1e-14)


def test_fit_converges_on_well_behaved_data(sample: Sample) -> None:
    y, X = sample
    assert missinggmm.fit(y, X).converged


def test_fit_recovers_true_coefficients_in_large_sample() -> None:
    y, X = _simulate(n_observations=20_000, seed=1)
    np.testing.assert_allclose(missinggmm.fit(y, X).params, TRUE_PARAMS, atol=0.05)


@pytest.mark.parametrize("missing_column", [0, 1, 2])
def test_fit_orders_parameters_like_columns_of_x(missing_column: int) -> None:
    y, X = _simulate(n_observations=20_000, seed=2, missing_column=missing_column)
    order = {0: [0, 1, 2], 1: [1, 0, 2], 2: [1, 2, 0]}[missing_column]
    np.testing.assert_allclose(
        missinggmm.fit(y, X).params, TRUE_PARAMS[order], atol=0.05
    )


def test_fit_with_few_incomplete_rows_is_close_to_statsmodels_ols() -> None:
    y, X = _simulate(n_observations=2_000, seed=3, share_missing=0.0)
    X[:10, 0] = np.nan
    ols = sm.OLS(y[10:], X[10:]).fit()
    np.testing.assert_allclose(missinggmm.fit(y, X).params, ols.params, atol=0.03)


def test_fit_rejects_fewer_incomplete_rows_than_observed_regressors() -> None:
    y, X = _simulate(n_observations=200, seed=4, share_missing=0.0)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="incomplete rows"):
        missinggmm.fit(y, X)


def test_fit_bse_is_not_larger_than_complete_case_ols(
    sample: Sample,
) -> None:
    """Using incomplete rows does not cost precision relative to dropping them."""
    y, X = sample
    complete = ~np.isnan(X).any(axis=1)
    ols = sm.OLS(y[complete], X[complete]).fit(cov_type="HC0")
    assert np.all(missinggmm.fit(y, X).bse <= ols.bse * 1.02)


def test_fit_standard_errors_match_monte_carlo_standard_deviation() -> None:
    """The sandwich standard errors track the sampling spread of the estimates."""
    estimates = []
    reported_bse = []
    for seed in range(300):
        y, X = _simulate(n_observations=500, seed=100 + seed)
        result = missinggmm.fit(y, X)
        estimates.append(result.params)
        reported_bse.append(result.bse)
    np.testing.assert_allclose(
        np.mean(reported_bse, axis=0), np.std(estimates, axis=0, ddof=1), rtol=0.15
    )


def test_fit_is_deterministic(sample: Sample) -> None:
    y, X = sample
    np.testing.assert_array_equal(
        missinggmm.fit(y, X).params, missinggmm.fit(y, X).params
    )


def test_summary_formats_coefficient_row() -> None:
    result = missinggmm.Results(
        params=np.array([1.0, 2.0]),
        bse=np.array([0.5, 1.0]),
        cov=np.diag([0.25, 1.0]),
        nobs=10,
        converged=True,
        n_iterations=3,
    )
    assert "x1      " + "      2.0000" + "      1.0000" + "      2.00" in (
        result.summary().splitlines()
    )


def test_fit_rejects_unknown_method(sample: Sample) -> None:
    y, X = sample
    with pytest.raises(ValueError, match="method"):
        missinggmm.fit(y, X, method="ols")  # ty: ignore[invalid-argument-type]


def test_fit_rejects_fewer_complete_rows_than_moment_conditions() -> None:
    y, X = _simulate(n_observations=20, seed=5, share_missing=0.0)
    X[:17, 0] = np.nan
    with pytest.raises(ValueError, match="At least 5 complete rows"):
        missinggmm.fit(y, X)


def test_fit_rejects_x_without_missing_values(sample: Sample) -> None:
    y, X = sample
    with pytest.raises(ValueError, match="missing"):
        missinggmm.fit(y, np.nan_to_num(X))


def test_fit_rejects_missing_values_in_several_columns(sample: Sample) -> None:
    y, X = sample
    X = X.copy()
    X[0, 2] = np.nan
    with pytest.raises(ValueError, match="one column"):
        missinggmm.fit(y, X)


def test_fit_rejects_missing_values_in_y(sample: Sample) -> None:
    y, X = sample
    y = y.copy()
    y[0] = np.nan
    with pytest.raises(ValueError, match="y"):
        missinggmm.fit(y, X)


def test_fit_rejects_mismatched_number_of_rows(sample: Sample) -> None:
    y, X = sample
    with pytest.raises(ValueError, match="rows"):
        missinggmm.fit(y[:-1], X)


@pytest.mark.parametrize("argument", ["y", "X"])
def test_fit_rejects_non_numeric_input(sample: Sample, argument: str) -> None:
    y, X = sample
    y_input = y.astype(str) if argument == "y" else y
    X_input = X.astype(str) if argument == "X" else X
    with pytest.raises(TypeError, match=f"{argument} must be numeric"):
        missinggmm.fit(y_input, X_input)


@pytest.mark.parametrize("argument", ["y", "X"])
def test_fit_rejects_infinite_values(sample: Sample, argument: str) -> None:
    y, X = sample
    y_input = y.copy()
    X_input = X.copy()
    (y_input if argument == "y" else X_input)[1] = np.inf
    with pytest.raises(ValueError, match=f"{argument} must not contain infinite"):
        missinggmm.fit(y_input, X_input)


def test_fit_rejects_regressors_collinear_among_rows_with_observed_regressor(
    sample: Sample,
) -> None:
    y, X = sample
    X = np.column_stack([X, 2 * X[:, 2]])
    with pytest.raises(ValueError, match="collinear in the rows with an observed"):
        missinggmm.fit(y, X)


def test_fit_rejects_observed_regressors_collinear_among_rows_with_missing_regressor(
    sample: Sample,
) -> None:
    y, X = sample
    X = X.copy()
    X[np.isnan(X[:, 0]), 2] = 1.0
    with pytest.raises(ValueError, match="collinear in the rows with a missing"):
        missinggmm.fit(y, X)
