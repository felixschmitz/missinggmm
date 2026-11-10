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
