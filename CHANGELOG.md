# Changelog

## [Unreleased]

- Extend Dependabot to `pre-commit` hook revisions and the `.ai-instructions` submodule.

## 0.1.0

- Initial project scaffold.
- Require the Claude co-author trailer on agent-authored commits and PR descriptions,
  enforced by a `PreToolUse` hook.
- Add `missinggmm.fit(y, X, method="gmm")` and the `Results` container. The GMM
  estimator of Abrevaya and Donald (2017) handles one regressor with missing values
  (marked by `NaN`) and reports sandwich standard errors.
- Validate `fit()` inputs: non-numeric arrays raise `TypeError`; infinite values and
  collinear regressors raise `ValueError`.
- Align the package description with the repository description.
- Add the "Why GMM" documentation page, arguing the efficiency case for the GMM
  estimator.
- Validate the GMM estimator in `tests/test_validation.py` against complete-case OLS, the
  dummy-variable method and the FGLS estimator of Dagenais, implemented as test helpers
  rather than package code.
- Rewrite the README for package users: replace the status roadmap with installation,
  a usage example and citation sections.
- Add a tag-triggered release workflow that publishes to TestPyPI, then PyPI, via trusted
  publishing. Tags containing `rc` or `dev` publish to TestPyPI only.
- Declare the license as the SPDX expression `MIT` instead of embedding the license text
  in the package metadata.
- Add a Quarto documentation site in `docs/` (`pixi run -e docs docs`) with a Monte Carlo
  comparison of GMM against complete-case OLS, the dummy-variable method and Dagenais'
  FGLS.
