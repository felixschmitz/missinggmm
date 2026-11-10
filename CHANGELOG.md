# Changelog

## Unreleased

- Initial project scaffold.
- Require the Claude co-author trailer on agent-authored commits and PR descriptions,
  enforced by a `PreToolUse` hook.
- Add `missinggmm.fit(y, X, method="gmm")` and the `Results` container. The GMM
  estimator of Abrevaya and Donald (2017) handles one regressor with missing values
  (marked by `NaN`) and reports sandwich standard errors.
- Validate `fit()` inputs: non-numeric arrays raise `TypeError`; infinite values and
  collinear regressors raise `ValueError`.
- Align the package description with the repository description.
- Add `docs/efficiency.md`, arguing the efficiency case for the GMM estimator, and
  `docs/efficiency_simulation.py`, a Monte Carlo comparison with complete-case OLS and
  the dummy-variable method.
- Validate the GMM estimator in `tests/test_validation.py` against complete-case OLS, the
  dummy-variable method and the FGLS estimator of Dagenais, implemented as test helpers
  rather than package code.
- Add a tag-triggered release workflow that publishes to TestPyPI, then PyPI, via trusted
  publishing.
