"""Monte Carlo comparison of GMM, complete-case OLS and the dummy-variable method.

The design follows Section IV of Abrevaya and Donald (2017): heteroskedastic errors,
`x` missing completely at random and all coefficients equal to one, with the share of
missing values and the coefficient of `z` in the projection of `x` varied.

Run with `pixi run python docs/efficiency_simulation.py`.
"""

import numpy as np

import missinggmm

N_OBSERVATIONS = 400
N_REPLICATIONS = 1_000
TRUE_PARAMS = np.array([1.0, 1.0, 1.0])  # alpha, beta_1 (constant), beta_2


def simulate_mean_squared_errors(
    share_missing: float, gamma_2: float, seed: int
) -> dict[str, np.ndarray]:
    """Return the mean squared error of (alpha, beta_1, beta_2) for each estimator."""
    rng = np.random.default_rng(seed=seed)
    squared_errors: dict[str, list[np.ndarray]] = {
        "complete case": [],
        "dummy variable": [],
        "GMM": [],
    }
    for _ in range(N_REPLICATIONS):
        y, X = _draw_sample(rng, share_missing, gamma_2)  # noqa: N806
        estimates = {
            "complete case": _fit_complete_case(y, X),
            "dummy variable": _fit_dummy_variable(y, X),
            "GMM": missinggmm.fit(y, X).params,
        }
        for name, params in estimates.items():
            squared_errors[name].append((params - TRUE_PARAMS) ** 2)
    return {name: np.mean(errors, axis=0) for name, errors in squared_errors.items()}


def _draw_sample(
    rng: np.random.Generator, share_missing: float, gamma_2: float
) -> tuple[np.ndarray, np.ndarray]:
    z_2 = rng.standard_normal(N_OBSERVATIONS)
    x = 1 + gamma_2 * z_2 + np.sqrt(1 + z_2**2) * rng.standard_normal(N_OBSERVATIONS)
    y = x + 1 + z_2 + np.sqrt(1 + x**2 + z_2**2) * rng.standard_normal(N_OBSERVATIONS)
    x_observed = np.where(rng.random(N_OBSERVATIONS) < share_missing, np.nan, x)
    return y, np.column_stack([x_observed, np.ones(N_OBSERVATIONS), z_2])


def _fit_complete_case(y: np.ndarray, X: np.ndarray) -> np.ndarray:  # noqa: N803
    is_complete = ~np.isnan(X[:, 0])
    return np.linalg.lstsq(X[is_complete], y[is_complete], rcond=None)[0]


def _fit_dummy_variable(y: np.ndarray, X: np.ndarray) -> np.ndarray:  # noqa: N803
    is_missing = np.isnan(X[:, 0])
    design = np.column_stack([np.where(is_missing, 0.0, X[:, 0]), X[:, 1:], is_missing])
    return np.linalg.lstsq(design, y, rcond=None)[0][:3]


if __name__ == "__main__":
    print(f"n = {N_OBSERVATIONS}, replications = {N_REPLICATIONS}")  # noqa: T201
    for share_missing, gamma_2 in [(0.5, 1.0), (0.2, 1.0), (0.8, 1.0), (0.5, 0.0)]:
        results = simulate_mean_squared_errors(share_missing, gamma_2, seed=0)
        print(f"\nshare missing = {share_missing}, gamma_2 = {gamma_2}")  # noqa: T201
        for name, mse in results.items():
            print(f"{name:<16}" + "".join(f"{value:>9.4f}" for value in mse))  # noqa: T201
