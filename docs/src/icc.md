```@meta
CurrentModule = MixedModelsExtras
DocTestSetup = quote
    using MixedModelsExtras
end
```

# Intraclass Correlation Coefficients

"Intraclass correlation coefficient" is not the name of one statistic. It is the name of a
family of statistics that share a form -- a ratio of a variance component to a total
variance -- but that answer different questions, carry different names in different
literatures, and are frequently confused for one another. Liljequist, Elfving and Skavberg
Roaldsen ([2019](https://doi.org/10.1371/journal.pone.0219854)) open their treatment by
observing that the ICC "is not a single, well defined statistical parameter", and much of
the confusion in applied work comes from reporting a number without saying which one it is.

This page lays out the coefficients that [`icc`](@ref) computes, what each one means, and
how to choose between them.

## Three families

The coefficients fall into three groups, which differ in what data they are computed from
and in what question they answer.

| Family | Question | Computed from | Also called |
|:---|:---|:---|:---|
| Variance partitioning | What share of the variance sits at this level? | a fitted `MixedModel` | VPC, repeatability |
| Inter-rater reliability | How well do raters agree? | a table of ratings | ICC(1,1)…ICC(3,k), ICC(A,1)… |
| Group-mean reliability | How reliable are the group means? | a fitted `MixedModel` | ICC(2) in the Bliese sense |

They are related -- all three reduce to the same ratio of variance components in the
simplest case -- but they are not interchangeable, and the same symbol "ICC(2)" refers to
two entirely different quantities in the second and third rows.

## Variance partitioning

Given a fitted mixed model, the ICC is the variance attributable to a grouping factor
divided by the total variance:

```math
\mathrm{ICC} = \frac{\sigma^2_{\alpha}}{\sigma^2_{\alpha} + \sigma^2_{\varepsilon}}
```

This is what [`icc`](@ref) returns for a `MixedModel`. In the multilevel modelling
literature it is called the **variance partition coefficient** (VPC; Goldstein, Browne and
Rasbash, 2002) and in the ecological literature **repeatability** (Nakagawa and Schielzeth,
2010). It answers: of the variability in the response, how much is explained by the
grouping or nesting structure?

```julia
using MixedModels, MixedModelsExtras, MixedModelsDatasets

model = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)),
            MixedModelsDatasets.dataset(:sleepstudy))
icc(model, :subj)
```

With several grouping factors, `icc` aggregates over the groups you name and puts every
grouping factor in the denominator, so `icc(model, :subj)` and `icc(model, :item)` sum to
`icc(model)`.

### Adjusted and conditional

The variance of the fixed effects can be either excluded from or included in the
denominator, and the two choices have different meanings:

- **Adjusted ICC** (`conditional=false`, the default) partitions variance among the random
  terms alone. It describes the correlation between two observations from the same group,
  *after* adjusting for the covariates -- which is the quantity the classical ICC
  definition refers to, and the one you want when the covariates are experimental
  manipulations you are controlling for.
- **Conditional ICC** (`conditional=true`), also called the *unadjusted* ICC, adds
  ``\mathrm{var}(X\beta)`` to the denominator. It gives the grouping factor's share of the
  *total* variance actually present in the data. It is always the smaller of the two when
  the fixed effects explain anything at all.

The terminology follows Nakagawa, Johnson and Schielzeth (2017) and the R `performance`
package. If a model has no fixed effects beyond the intercept, the two coincide.

### Random slopes

A model with random slopes has no single random-effects variance: the variance contributed
by the grouping factor depends on where you are in covariate space. Johnson
([2014](https://doi.org/10.1111/2041-210X.12225)) resolves this by averaging the
random-effects variance over the observations,

```math
\sigma^2_{\alpha} = \frac{1}{n}\sum_{i=1}^{n} z_i' \Sigma z_i
```

where ``z_i`` is the ``i``th row of the random-effects model matrix. This is the default
(`slopes=:mean`). It reduces exactly to the intercept variance for a random-intercept
model.

Because the average depends on the model matrix, it also depends on the scaling and
centring of your covariates: uncentred predictors will inflate it. `slopes=:diagonal` sums
the random-effect variances instead, ignoring both the covariances and the model matrix.
That was the behaviour of MixedModelsExtras before version 3.0 and is retained for
reproducing older results, but it does not correspond to a variance the model actually
implies for any observation.

### Generalized linear mixed models

A GLMM has no residual variance in the usual sense, so the denominator's
``\sigma^2_{\varepsilon}`` has to be supplied by some convention. This is a real modelling
choice, not an implementation detail, and different conventions give materially different
numbers. Nakagawa et al. (2017) catalogue the options; `method` selects among them.

For **binomial** families:

- `:theoretical` (default) uses the variance of the latent distribution that the link
  implies: ``\pi^2/3`` for logit, ``1`` for probit, ``\pi^2/6`` for complementary log-log.
  The ICC is then a statement about a latent continuous variable underlying the binary
  response.
- `:observation_level` uses the variance of the response distribution,
  ``1/(\bar{p}(1-\bar{p}))`` for logit. Since ``1/(\bar{p}(1-\bar{p})) \geq 4 > \pi^2/3``,
  this always gives a smaller ICC.

For **Poisson** models with a log link, all three approximations are functions of the
expected count ``\lambda = \exp(\bar\eta + \sigma^2/2)``:

| `method` | ``\sigma^2_{\varepsilon}`` |
|:---|:---|
| `:delta` | ``1/\lambda`` |
| `:lognormal` (default) | ``\ln(1 + 1/\lambda)`` |
| `:trigamma` | ``\psi_1(\lambda)`` |

!!! note "Changed in version 3.0"
    Earlier versions used a residual variance of exactly `1.0` for Poisson models, which is
    not one of the standard estimators. The default is now `:lognormal`, so Poisson ICCs
    will differ from those computed with version 2.

!!! note "Deviation from the reference implementations"
    Nakagawa et al. express ``\lambda`` and ``\bar{p}`` in terms of the intercept of a
    refitted intercept-only *null model*. MixedModels.jl does not retain the data needed to
    refit one, so the mean of the fixed-effects linear predictor ``\overline{X\beta}`` is
    used instead. The two agree exactly when the intercept is the only fixed effect, and
    closely when the covariates are centred. Expect small differences from R's
    `performance` and `MuMIn` for models with uncentred covariates.

`method=:simulation` sidesteps the choice entirely. Following Goldstein et al. (2002), it
simulates random effects from the fitted model, transforms them to the response scale, and
partitions the variance there, so the result is a variance partition of the observed data
rather than of a latent variable. It accepts `nsim` and `rng`. For binary data it is
typically noticeably smaller than the latent-scale ICC, and the gap between them is a good
reminder that the latent-scale number is not a statement about the observed responses.

### REML and ML

An ICC computed from a fitted model inherits that model's estimation criterion. On balanced
data, REML variance components reproduce the classical ANOVA estimators; maximum likelihood
does not correct for the estimation of the fixed effects and gives different --- generally
smaller --- variance components. MixedModels.jl fits with ML by default, so refit with
`REML=true` if you want the ICC to match a classical ANOVA-based calculation.

Note that ML bias does *not* translate into a predictable direction for the ICC itself: it
is a ratio, and both numerator and denominator shift. GLMMs have no REML.

## Inter-rater reliability

This is the ICC of the reliability literature: ``n`` targets are each rated by ``k`` raters,
and the question is how much of the variability in the ratings reflects real differences
between targets rather than disagreement between raters. Pass a long-format table:

```julia
icc(ratings; target=:subject, rater=:judge, score=:rating)
```

With no further arguments this returns all six coefficients as an
[`InterraterICCTable`](@ref); naming a `model`, `type` or `unit` returns the single
[`InterraterICC`](@ref) you asked for.

### The naming problem

Shrout and Fleiss (1979) defined six coefficients and named them by a pair of numbers.
McGraw and Wong (1996) extended the framework to ten and introduced a more transparent
notation in which the important distinction -- absolute agreement versus consistency --
appears in the name. Both notations remain in wide use, so `icc` reports both:

| McGraw–Wong | Shrout–Fleiss | Model | Definition |
|:---|:---|:---|:---|
| ICC(1) | ICC(1,1) | one-way random | ``\sigma^2_r/(\sigma^2_r+\sigma^2_w)`` |
| ICC(A,1) | ICC(2,1) | two-way random | ``\sigma^2_r/(\sigma^2_r+\sigma^2_c+\sigma^2_v)`` |
| ICC(C,1) | ICC(3,1) | two-way mixed | ``\sigma^2_r/(\sigma^2_r+\sigma^2_v)`` |
| ICC(k) | ICC(1,k) | one-way random | as above, for the mean of ``k`` |
| ICC(A,k) | ICC(2,k) | two-way random | as above, for the mean of ``k`` |
| ICC(C,k) | ICC(3,k) | two-way mixed | as above, for the mean of ``k`` |

Here ``\sigma^2_r`` is the variance between targets, ``\sigma^2_c`` the variance due to
systematic differences between raters, and ``\sigma^2_v`` the residual noise, in the
notation of Liljequist et al. (2019), whose measurement model is

```math
x_{ij} = \mu + r_i + c_j + v_{ij}.
```

Three decisions pick out a coefficient, and they map onto the keyword arguments:

- `model`: `:oneway` if each target is rated by a *different* set of raters, so that rater
  effects cannot be separated from noise; `:twoway` if every rater rates every target.
- `type`: `:agreement` if you care whether raters give the *same* value; `:consistency` if
  you only care whether they *rank* targets the same way. Consistency excludes
  ``\sigma^2_c`` from the denominator, so it is the larger of the two whenever raters differ
  systematically.
- `unit`: `:single` if a single rating will be used in practice; `:average` if the mean of
  ``k`` ratings will be. Average-measure coefficients are always the larger.

McGraw and Wong's remaining four forms distinguish random from fixed raters. Liljequist et
al. show that this distinction does not change the formulas for a given data matrix -- only
the population the inference generalizes to -- so `icc` computes six distinct values.

### The Liljequist workflow

Rather than committing to a model in advance, Liljequist et al. (2019) recommend computing
all of the coefficients and letting the data show what is going on. That is why `icc`
returns all six by default. Their procedure:

1. Compare ICC(1), ICC(A,1) and ICC(C,1). If the three are close, there is no appreciable
   rater bias. If ICC(C,1) is much larger than ICC(A,1), there is.
2. Confirm with the F test ``\mathrm{MSC}/\mathrm{MSE}``, reported as `biastest`.
3. If there is no bias, report any of them -- they agree. If there is, report **both**
   ICC(A,1) and ICC(C,1), because they carry complementary information: how well raters
   agree in absolute terms, and how well they agree about the ordering.
4. Report the variance components ``\hat\sigma^2_r``, ``\hat\sigma^2_c``,
   ``\hat\sigma^2_v`` alongside the coefficient. They are in the `varcomp` field.

The key warning is about ICC(1): when rater bias is present, it estimates *neither*
population coefficient, because a one-way model folds bias and noise together. It is not a
conservative choice.

### Negative values

An ICC estimate can come out negative, when the between-target mean square falls below the
within-target mean square. This is not a meaningful negative correlation; it is an unlucky
sample, and it happens most easily with few targets. `icc` reports such values as computed
rather than truncating them at zero, since truncation hides the fact that the data carry
essentially no between-target signal. Note that the `estimator=:lmm` backend constrains
variances to be non-negative and so cannot produce them, returning a boundary fit at
``\hat\sigma^2 = 0`` instead.

### Estimators and intervals

`estimator=:anova` (the default) computes the coefficients from ANOVA mean squares and
reports the exact F-based confidence intervals of Shrout and Fleiss (1979) and McGraw and
Wong (1996). It requires a complete, balanced design.

`estimator=:lmm` fits a mixed model with crossed random intercepts for target and rater and
derives the mean squares from its variance components. This handles unbalanced and
incomplete data, at the cost of the exact intervals -- use
`MixedModels.parametricbootstrap` for an interval instead. With `reml=true` (the default)
it reproduces the ANOVA values on balanced data.

### Interpreting the magnitude

Two sets of rule-of-thumb benchmarks are in common use: Cicchetti (1994) and Koo and Li
(2016), the latter suggesting values below 0.5 indicate poor reliability, 0.5–0.75
moderate, 0.75–0.9 good and above 0.9 excellent. Treat these as very rough. They ignore the
width of the confidence interval, which for the small samples typical of reliability studies
is often wide enough to span several categories, and what counts as adequate depends
entirely on the use to which the measurement will be put.

## Group-mean reliability

In organizational and multilevel survey research, following Bliese (2000), "ICC(1)" means
the variance-partitioning ICC and "ICC(2)" means something else entirely: the reliability of
the *group means*, obtained by stepping ICC(1) up with the Spearman–Brown formula,

```math
\mathrm{ICC(2)} = \frac{k \cdot \mathrm{ICC(1)}}{1 + (k-1)\,\mathrm{ICC(1)}}
```

where ``k`` is the group size. It answers whether group means can be reliably told apart --
the relevant question when deciding whether individual responses can be aggregated to the
group level. Because ``k`` enters directly, large groups produce reliable means even when
ICC(1) is small.

```julia
icc(model, :subj; groupmean=true)
```

`k` defaults to the harmonic mean of the group sizes, which is the appropriate summary for
unequal groups because reliability is a function of ``1/k``; pass `k` to override it.

Note that this ICC(2) is mathematically ICC(1,k) from the reliability table above, and has
nothing to do with Shrout and Fleiss's ICC(2,1). This is the single most common source of
confusion in the ICC literature, and a good reason to write out which coefficient you mean
rather than relying on the numeral.

## Bootstrap intervals

Any of the model-based ICCs can be given a bootstrap interval by passing a
`MixedModelBootstrap`:

```julia
boot = parametricbootstrap(rng, 1000, model)
ci = confint(icc(boot, :subj))
```

Because `MixedModelBootstrap` stores only variances and correlations, a random-slope model
(or `conditional=true`, which needs the fixed-effects model matrix) additionally requires
the original model as the `model` keyword argument. For a GLMM, the family must be given,
since the bootstrap does not record it.

## References

- Bartko, J. J. (1966). The intraclass correlation coefficient as a measure of reliability.
  *Psychological Reports*, 19, 3–11.
- Bliese, P. D. (2000). Within-group agreement, non-independence, and reliability. In K. J.
  Klein & S. W. Kozlowski (Eds.), *Multilevel Theory, Research, and Methods in
  Organizations* (pp. 349–381). Jossey-Bass.
- Browne, W. J., Subramanian, S. V., Jones, K., & Goldstein, H. (2005). Variance
  partitioning in multilevel logistic models that exhibit overdispersion. *Journal of the
  Royal Statistical Society A*, 168, 599–613.
- Cicchetti, D. V. (1994). Guidelines, criteria, and rules of thumb for evaluating normed
  and standardized assessment instruments in psychology. *Psychological Assessment*, 6,
  284–290.
- Fisher, R. A. (1925). *Statistical Methods for Research Workers*. Oliver & Boyd.
- Goldstein, H., Browne, W., & Rasbash, J. (2002). Partitioning variation in multilevel
  models. *Understanding Statistics*, 1, 223–231.
- Johnson, P. C. D. (2014). Extension of Nakagawa & Schielzeth's ``R^2_{GLMM}`` to random
  slopes models. *Methods in Ecology and Evolution*, 5, 944–946.
  [doi:10.1111/2041-210X.12225](https://doi.org/10.1111/2041-210X.12225)
- Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass
  correlation coefficients for reliability research. *Journal of Chiropractic Medicine*, 15,
  155–163.
- Liljequist, D., Elfving, B., & Skavberg Roaldsen, K. (2019). Intraclass correlation -- a
  discussion and demonstration of basic features. *PLoS ONE*, 14, e0219854.
  [doi:10.1371/journal.pone.0219854](https://doi.org/10.1371/journal.pone.0219854)
- Lüdecke, D., Ben-Shachar, M. S., Patil, I., Waggoner, P., & Makowski, D. (2021).
  performance: An R package for assessment, comparison and testing of statistical models.
  *Journal of Open Source Software*, 6, 3139.
- McGraw, K. O., & Wong, S. P. (1996). Forming inferences about some intraclass correlation
  coefficients. *Psychological Methods*, 1, 30–46.
- Nakagawa, S., Johnson, P. C. D., & Schielzeth, H. (2017). The coefficient of determination
  ``R^2`` and intra-class correlation coefficient from generalized linear mixed-effects
  models revisited and expanded. *Journal of the Royal Society Interface*, 14, 20170213.
  [doi:10.1098/rsif.2017.0213](https://doi.org/10.1098/rsif.2017.0213)
- Nakagawa, S., & Schielzeth, H. (2010). Repeatability for Gaussian and non-Gaussian data.
  *Biological Reviews*, 85, 935–956.
- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in assessing rater
  reliability. *Psychological Bulletin*, 86, 420–428.
- Stoffel, M. A., Nakagawa, S., & Schielzeth, H. (2017). rptR: repeatability estimation and
  variance decomposition by generalized linear mixed-effects models. *Methods in Ecology and
  Evolution*, 8, 1639–1644.
