# Changelog

## [Unreleased]

- Compute the data cross products of the GMM Jacobian once instead of at every Gauss-Newton
  iteration.
- Bump pre-commit hooks, GitHub Actions and the `.ai-instructions` submodule; ignore
  ruff `CPY001` (no copyright headers).
- Extend Dependabot to `pre-commit` hook revisions and the `.ai-instructions` submodule.
- Report the J statistic of the over-identifying restrictions and its degrees of
  freedom in `Results` (`j_statistic`, `j_df`) and in `Results.summary()`.
- Report the GMM covariance as `inv(G' S⁻¹ G) / n` with the weight matrix `S⁻¹`
  re-estimated at the final estimates, as in the reference GAUSS code; `Results.bse` and
  `Results.cov` change slightly in finite samples.

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
- Add `docs/efficiency.md`, arguing the efficiency case for the GMM estimator, and
  `docs/efficiency_simulation.py`, a Monte Carlo comparison with complete-case OLS and
  the dummy-variable method.
- Validate the GMM estimator in `tests/test_validation.py` against complete-case OLS, the
  dummy-variable method and the FGLS estimator of Dagenais, implemented as test helpers
  rather than package code.
- Rewrite the README for package users: replace the status roadmap with installation,
  a usage example and citation sections.
- Add a tag-triggered release workflow that publishes to TestPyPI, then PyPI, via trusted
  publishing. Tags containing `rc` or `dev` publish to TestPyPI only.
- Declare the license as the SPDX expression `MIT` instead of embedding the license text
  in the package metadata.
