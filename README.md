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

## Status

The package is pre-alpha. The GMM estimator is implemented as `missinggmm.fit`, which
returns a `missinggmm.Results` object with input validation. The baseline estimators are
not implemented yet. Progress is tracked in the roadmap issues:

- [#1](https://github.com/felixschmitz/missinggmm/issues/1): Define public API and scope
  (done).
- [#2](https://github.com/felixschmitz/missinggmm/issues/2): Port GMM estimator (done).
- [#3](https://github.com/felixschmitz/missinggmm/issues/3): Add input validation
  (done).
- [#4](https://github.com/felixschmitz/missinggmm/issues/4): Port baseline estimators
  (open): complete-case OLS, the dummy-variable method, and the weighted FGLS estimator
  of Dagenais (1973).
- [#5](https://github.com/felixschmitz/missinggmm/issues/5): Release 0.1.0 (open).

Documentation of the efficiency gains over the baselines will be linked here once it
exists.

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
