# missinggmm

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

Generalized Method of Moments (GMM) estimation of linear regressions with a missing
regressor, following Abrevaya and Donald (2017).

> [!WARNING]
> Pre-alpha: the API is not yet stable.

## Overview

`missinggmm` estimates linear regression models when one regressor is missing for part
of the sample. Instead of dropping incomplete observations or imputing values, the
estimator uses the incomplete cases through additional moment conditions, which can
increase efficiency over the complete-case estimator. See
[docs/efficiency.md](docs/efficiency.md) for the argument.

Features:

- GMM estimator of Abrevaya and Donald (2017) with robust standard errors.
- Input validation with descriptive errors.
- Typed, with `numpy` as the only runtime dependency.

The test suite validates the estimator against complete-case OLS, the dummy-variable
method, and the weighted FGLS estimator of Dagenais (1973).

## Installation

The package is not yet released on PyPI. To install it from source:

```sh
pip install git+https://github.com/felixschmitz/missinggmm
```

## Usage

Mark the missing values of the incomplete regressor with `NaN`. Exactly one column of
`X` may contain missing values; include a constant column if the model has an intercept.

```python
import numpy as np
import missinggmm

rng = np.random.default_rng(seed=0)
n = 2_000
constant = np.ones(n)
z = rng.normal(size=n)
x = 0.5 * z + rng.normal(size=n)
y = 1.0 + 2.0 * x - 1.0 * z + rng.normal(size=n)

x_observed = np.where(rng.uniform(size=n) < 0.7, x, np.nan)
X = np.column_stack([constant, x_observed, z])

results = missinggmm.fit(y, X)
print(results.summary())
```

`fit` returns a `missinggmm.Results` object with the coefficients (`params`), standard
errors (`bse`), covariance matrix (`cov`), the number of observations (`nobs`), and
convergence information (`converged`, `n_iterations`). Coefficients are ordered like the
columns of `X`.

## Development

The project uses [pixi](https://pixi.sh) for environments and
[prek](https://github.com/j178/prek) for linting and formatting.

```sh
git clone --recurse-submodules https://github.com/felixschmitz/missinggmm
cd missinggmm
pixi install
pixi run tests
pixi run prek run --all-files
```

## Citation

If you use this package, please cite the original paper. `CITATION.cff` contains
citation metadata for the software.

## References

Abrevaya, J., & Donald, S. G. (2017). A GMM approach for dealing with missing data on
regressors. *The Review of Economics and Statistics*, 99(4), 657-662.

Dagenais, M. G. (1973). The use of incomplete observations in multiple regression
analysis: A generalized least squares approach. *Journal of Econometrics*, 1(4),
317-328.

## License

MIT. See [LICENSE](LICENSE).
