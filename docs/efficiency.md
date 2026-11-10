# Why GMM: the efficiency case

This note explains why the GMM estimator of Abrevaya and Donald (2017) is preferable to
the baselines it is compared with: complete-case OLS, the dummy-variable method and
Dagenais' FGLS. Equation, assumption and proposition numbers refer to the published
paper (Abrevaya and Donald, 2017, *Review of Economics and Statistics* 99(4): 657–662).

## Setting

The model is $y_i = \alpha_0 x_i + z_i'\beta_0 + \varepsilon_i$, where $x_i$ is a scalar
regressor that is missing for some rows ($m_i = 1$) and $z_i$ collects the regressors
that are always observed, including a constant. The missing regressor is linked to the
observed ones by the linear projection $x_i = z_i'\gamma_0 + \xi_i$ (equation 3).
Combining the two gives, for rows with a missing $x_i$,

$$y_i = z_i'(\gamma_0 \alpha_0 + \beta_0) + \varepsilon_i + \xi_i \alpha_0$$

(equation 4). Missingness may depend on $z_i$ in arbitrary ways but not on $y_i$ or on
$x_i$ (Assumption 1: $E(m_i z_i \varepsilon_i) = E(m_i z_i \xi_i) = E(m_i x_i \varepsilon_i) =
0$).

## 1. Incomplete rows add moment conditions

The moment function (equation 6) stacks three blocks, each multiplied by $z_i$ or
$w_i = (x_i, z_i')'$:

- $(1 - m_i)\, w_i (y_i - \alpha x_i - z_i'\beta)$: the OLS conditions in complete rows.
- $(1 - m_i)\, z_i (x_i - z_i'\gamma)$: the imputation regression in complete rows.
- $m_i\, z_i (y_i - z_i'(\gamma\alpha + \beta))$: the regression of $y$ on $z$ in
  incomplete rows, which is where the incomplete rows enter.

There are $3K + 1$ moment conditions for the $2K + 1$ parameters $(\alpha, \beta,
\gamma)$, leaving $K$ overidentifying restrictions. The just-identified estimator that
uses the first block and either the second or the third block (not both) equals the
complete-case estimator. Hence the complete-case estimator is GMM with a subset of the
moment conditions, and the optimally weighted GMM estimator using all of them is at
least as efficient (Section III.A). The extra conditions are valid only because of the
restriction in Assumption 1(b), that the imputation regression is the same in complete
and incomplete rows. That restriction is the source of the efficiency gain.

The gain is not confined to the coefficient of the missing regressor. Proposition 2
shows that the difference in asymptotic variances is positive semi-definite, for both
$\alpha_0$ and $\beta_0$:

- For $\alpha_0$ the gain is $\sigma_{\xi c}^{-2} \Lambda_{\varepsilon\eta\xi c}' D
  \Lambda_{\varepsilon\eta\xi c}$, which is zero if and only if $\Lambda_{\varepsilon\eta\xi
  c} = 0$.
- For $\beta_0$ the gain is $B' D B$, which is zero if and only if $B = \Omega_{\varepsilon
  \eta c} \Gamma_c^{-1} - \Lambda_{\varepsilon\eta\xi c} \sigma_{\xi c}^{-2} \gamma_0'$
  is zero.

($D$ is positive definite and defined in the paper's technical appendix.) Earlier
imputation estimators were known to gain only for $\beta_0$; the paper notes that the
cross-moment $\Omega_{12}$, which an earlier version incorrectly set to zero, is what
allows a gain for $\alpha_0$ (footnote 4).

## 2. When the gain is large or small

Proposition 2 gives exact conditions under which there is *no* gain, and the paper's
Monte Carlo study illustrates the rest. The paper does not state a closed-form rule
linking the gain to the share of missing data or to the predictive power of $z$, so the
following is read from the conditions above and from the simulations in this
repository.

- **No missing data in the relevant moment:** with no incomplete rows the third block
  is empty and GMM reduces to OLS. With few incomplete rows there is little extra
  information.
- **$\Lambda_{\varepsilon\eta\xi c} = 0$:** no gain for $\alpha_0$. The paper reports
  this holds when the conditional variance of $\varepsilon$ depends only on $x$, in which
  case no method improves on complete-case estimation of $\alpha_0$ (Section IV).
- **$B = 0$:** no gain for $\beta_0$.
- **Heteroskedasticity:** the paper reports a more pronounced gain for $\alpha_0$ when
  the conditional variances are more heteroskedastic (Section IV).
- **Predictive power of $z$:** the simulation below varies the slope $\gamma_2$ of $z$ in
  the projection of $x$. In this design the gains for $\alpha$ and $\beta_1$ are of
  similar size for $\gamma_2 = 0$ and $\gamma_2 = 1$, so the simulation gives no
  evidence that a weaker projection removes them.

## 3. A specification test

The $K$ overidentifying restrictions allow a standard test (equation 10): under
Assumption 1, $n\, \bar g(\hat\alpha, \hat\beta, \hat\gamma)' \hat\Omega^{-1} \bar
g(\hat\alpha, \hat\beta, \hat\gamma) \to_d \chi^2(K)$. The paper recommends that
researchers always report its p-value. A rejection signals that the imputation
regression differs between complete and incomplete rows, or that another part of
Assumption 1 fails. The complete-case estimator offers no such check, because it is
just-identified. Reporting this statistic from `Results` is tracked separately.

## 4. The dummy-variable method is generally inconsistent

The dummy-variable method sets missing $x_i$ to zero and adds the indicator $m_i$ to the
regression. The correct model for the data is (equation 13)

$$y_i = (1 - m_i) x_i \alpha_0 + z_i'\beta_0 + m_i \gamma_{10} \alpha_0 + m_i z_{2i}'
\gamma_{20} \alpha_0 + \varepsilon_i + m_i \xi_i \alpha_0,$$

and the dummy-variable regression omits the terms $m_i z_{2i}' \gamma_{20}$. Proposition 3
(after Jones, 1996) states that the OLS estimators of $(\alpha_0, \beta_0)$ are biased and
inconsistent unless $\alpha_0 = 0$ or $\gamma_{20} = 0$, that is, unless $x$ is irrelevant or
orthogonal to the other regressors. This holds even when data are missing completely at
random. When it is consistent, it still need not be more efficient than complete-case
OLS: the residual variance differs between missing and non-missing rows, so the
restricted estimator has no guaranteed advantage (Section III.B).

## 5. Dagenais FGLS

FGLS (Dagenais, 1973) is asymptotically equivalent to a GMM estimator whose weight
matrix is optimal only under homoskedasticity and $\Omega_{12} = 0$. It gains over
complete-case OLS for $\beta_0$ but not for $\alpha_0$ and is relatively inefficient
compared to GMM otherwise (Section III.C). The baseline estimators are not yet
implemented in this package, so the simulation below omits FGLS; the paper's Table 1
reports it.

## Simulation

`docs/efficiency_simulation.py` reproduces the design of Section IV of the paper with
$K = 2$, $n = 400$, 1,000 replications, all coefficients equal to one and heteroskedastic
errors $\sigma_\varepsilon = \sqrt{1 + x^2 + z_2^2}$, $\sigma_\xi = \sqrt{1 + z_2^2}$. It
varies the share of missing values and $\gamma_2$. Entries are mean squared errors of
$(\alpha, \beta_1, \beta_2)$; run `pixi run python docs/efficiency_simulation.py` to
reproduce them.

| Share missing | $\gamma_2$ | Estimator      | $\alpha$ | $\beta_1$ | $\beta_2$ |
| ------------- | ---------- | -------------- | -------- | --------- | --------- |
| 0.5           | 1          | Complete case  | 0.0343   | 0.0459    | 0.0617    |
|               |            | Dummy variable | 0.0732   | 0.0870    | 0.3901    |
|               |            | GMM            | 0.0326   | 0.0380    | 0.0482    |
| 0.2           | 1          | Complete case  | 0.0221   | 0.0309    | 0.0382    |
|               |            | Dummy variable | 0.0296   | 0.0384    | 0.1046    |
|               |            | GMM            | 0.0222   | 0.0284    | 0.0348    |
| 0.8           | 1          | Complete case  | 0.0854   | 0.1212    | 0.1521    |
|               |            | Dummy variable | 0.1626   | 0.2025    | 0.7685    |
|               |            | GMM            | 0.0770   | 0.0892    | 0.1129    |
| 0.5           | 0          | Complete case  | 0.0315   | 0.0374    | 0.0446    |
|               |            | Dummy variable | 0.0317   | 0.0374    | 0.0272    |
|               |            | GMM            | 0.0297   | 0.0321    | 0.0290    |

What the numbers show:

- **First block** (50% missing, $\gamma_2 = 1$) is the paper's design. The results are
  close to the paper's Table 1: GMM has the lowest MSE for every parameter, complete case is
  next and the dummy-variable method is the worst, especially for $\beta_2$ (bias
  from omitted $m_i z_{2i}$).
- **The gain from GMM is modest for $\alpha$** (about 5% at 50% missing, about 10% at 80%)
  and larger for $\beta_1$ and $\beta_2$ (roughly 15–25%).
- **Less missing data, smaller gain:** at 20% missing, GMM and complete case are
  indistinguishable for $\alpha$ and differ by under 10% for the other coefficients.
- **More missing data, larger gain:** at 80% missing the MSE reductions relative to
  complete case are about 10% ($\alpha$) and 25% ($\beta_1$, $\beta_2$).
- **Dummy variable method:** with $\gamma_2 = 1$ it has larger MSE than complete case for
  every parameter. With $\gamma_2 = 0$ it is consistent. Its MSE is then about equal
  to complete case for $\alpha$ and $\beta_1$ and lower for $\beta_2$, which matches the
  paper's remark that its variance is smaller for $\beta_2$. Here GMM is still lower for
  $\alpha$ and $\beta_1$.

Caveats: MSE differences of a few percent are within Monte Carlo noise at 1,000
replications; only one seed is run; and the missingness is MCAR, which satisfies
Assumption 1 trivially. The simulation says nothing about performance when Assumption 1
fails (for example when missingness depends on $y$), where the paper's estimator is not
consistent.

## Reference

Abrevaya, J. and Donald, S. G. (2017). A GMM approach for dealing with missing data on
regressors. *The Review of Economics and Statistics*, 99(4), 657–662.
doi:10.1162/REST_a_00645.
