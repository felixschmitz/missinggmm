# missinggmm

Generalized Method of Moments (GMM) estimation with missing regressors, following
Abrevaya and Donald (2017).

> [!WARNING]
> Pre-alpha: this package is under active development and the API is not yet stable.

## Overview

`missinggmm` estimates linear regression models when one regressor is missing for part
of the sample. Instead of dropping incomplete observations or imputing values, the
estimator uses the incomplete cases through additional moment conditions, which can
increase efficiency over the complete-case estimator.

The following estimators are planned:

- The GMM estimator of Abrevaya and Donald (2017).
- Baselines for comparison: complete-case OLS, the dummy-variable method, and the
  weighted FGLS estimator of Dagenais (1973).

## Installation

Not yet released on PyPI. For development, see below.

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

## References

Abrevaya, J., & Donald, S. G. (2017). A GMM approach for dealing with missing data on
regressors. *The Review of Economics and Statistics*, 99(4), 657-662.

Dagenais, M. G. (1973). The use of incomplete observations in multiple regression
analysis: A generalized least squares approach. *Journal of Econometrics*, 1(4),
317-328.

## License

MIT. See [LICENSE](LICENSE).
