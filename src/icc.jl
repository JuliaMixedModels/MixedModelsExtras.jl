const SymbolCollection = Union{Symbol,Tuple{Symbol,Vararg{Symbol}},AbstractVector{Symbol}}

"""
    IccBootstrap{T} <: AbstractVector{T}

Thin wrapper around the per-iteration ICC values computed by [`icc`](@ref) from a
`MixedModelBootstrap`. Behaves like the underlying vector for all `AbstractArray`
purposes (indexing, iteration, `sort`, `quantile`, etc.), but is a distinct type so
that `confint` can be specialized for it without pirating `MixedModelBootstrap`.
"""
struct IccBootstrap{T} <: AbstractVector{T}
    values::Vector{T}
end

Base.size(x::IccBootstrap) = size(x.values)
Base.getindex(x::IccBootstrap, i::Int) = getindex(x.values, i)
Base.setindex!(x::IccBootstrap, v, i::Int) = setindex!(x.values, v, i)
Base.IndexStyle(::Type{<:IccBootstrap}) = IndexLinear()
Base.similar(x::IccBootstrap, ::Type{S}, dims::Dims) where {S} = similar(x.values, S, dims)

function Base.show(io::IO, ::MIME"text/plain", x::IccBootstrap)
    println(io, "$(length(x))-element IccBootstrap{$(eltype(x))}")
    isempty(x) && return nothing
    lower, upper = shortestcovint(x, 0.95)
    print(io, "  median ", _fmt(median(x)), ", 95% CI (", _fmt(lower), ", ",
          _fmt(upper), ")")
    return nothing
end

####
#### random-effects variance
####

# for a LMM, λ is relative to the residual standard deviation; for a GLMM without
# a dispersion parameter, λ is already on the absolute scale
_re_scale(model::LinearMixedModel) = varest(model)
_re_scale(::GeneralizedLinearMixedModel) = 1.0

"""
    _recov(re::AbstractReMat, scale)

The (unscaled) random-effects covariance matrix ``Σ = scale ⋅ λλ'`` for a single
grouping factor.
"""
_recov(re, scale) = scale * (Matrix(re.λ) * Matrix(re.λ)')

"""
    _mean_re_variance(re, scale, slopes)

The random-effects variance for one grouping factor.

For a random-intercept term this is simply the intercept variance, and both methods
agree. For a model with random slopes there is no single random-effects variance --
the variance contributed by the grouping factor depends on the covariate values -- and
the two methods differ:

- `slopes=:mean` (the default) uses Johnson's (2014) extension, averaging the quadratic
  form ``z_i' Σ z_i`` over the ``n`` observations, where ``z_i`` is the ``i``th row of
  the random-effects model matrix.
- `slopes=:diagonal` sums the variances of the random-effects terms, ignoring both the
  covariances and the model matrix. This was the behaviour of MixedModelsExtras before
  version 3.0 and is retained for reproducing older results.
"""
function _mean_re_variance(re, scale, slopes::Symbol=:mean)
    Σ = _recov(re, scale)
    slopes === :diagonal && return tr(Σ)
    slopes === :mean ||
        throw(ArgumentError("`slopes` must be either :mean or :diagonal, got " *
                            "$(repr(slopes))."))
    return mean(z -> dot(z, Σ, z), eachcol(re.z))
end

function _reterm(model::MixedModel, group::Symbol)
    idx = findfirst(re -> fname(re) === group, model.reterms)
    isnothing(idx) &&
        throw(ArgumentError("$(repr(group)) is not a grouping variable in this model. " *
                            "Available grouping variables: $(collect(fnames(model)))."))
    return model.reterms[idx]
end

function _group_var(model::MixedModel, group::Symbol; slopes::Symbol=:mean)
    return _mean_re_variance(_reterm(model, group), _re_scale(model), slopes)
end

function _group_var(model::MixedModel, groups::SymbolCollection; slopes::Symbol=:mean)
    scale = _re_scale(model)
    return sum(g -> _mean_re_variance(_reterm(model, g), scale, slopes), groups)
end

function _group_var(model::MixedModel; slopes::Symbol=:mean)
    return _group_var(model, fnames(model); slopes)
end

# --- bootstrap versions -------------------------------------------------------
#
# The bootstrap stores only the standard deviations and correlations of the random
# effects, so the Johnson (2014) average requires the random-effects model matrices
# from the original model. Without a model we can still handle the random-intercept
# case exactly, because there the average is just the intercept variance.

function _group_var(tbl, group::Symbol)
    d = _split_by_iter(tbl)
    return map(sort!(collect(keys(d)))) do key
        return sum(abs2(row.σ) for row in Tables.rows(d[key]) if row.group == group;
                   init=0.0)
    end
end

function _group_var(tbl)
    groups = unique((row.group for row in Tables.rows(tbl)))
    return _group_var(tbl, groups)
end

function _group_var(tbl, groups::SymbolCollection)
    return sum(_group_var.(Ref(tbl), groups))
end

function _split_by_iter(tbl)
    d = Dict{Int,Vector}()
    for row in Tables.rows(tbl)
        vv = get!(Vector{Any}, d, row.iter)
        push!(vv, row)
    end
    return d
end

"""
    _boot_group_var(boot, model, groups)

Per-iteration Johnson (2014) random-effects variance, reconstructing each iteration's
``λ`` from the stored ``θ`` and combining it with the random-effects model matrices of
`model`.
"""
function _boot_group_var(boot::MixedModelBootstrap, model::MixedModel,
                         groups::Union{Symbol,SymbolCollection}; slopes::Symbol=:mean)
    groups = groups isa Symbol ? (groups,) : groups
    scale = _boot_re_scale(boot, model)
    reterms = [_reterm(model, g) for g in groups]
    return map(eachindex(boot.fits)) do i
        MixedModels.setθ!(boot, i)
        return sum(zip(reterms, _boot_lambdas(boot, model, groups))) do (re, λ)
            Σ = scale[i] * (Matrix(λ) * Matrix(λ)')
            slopes === :diagonal && return tr(Σ)
            return mean(z -> dot(z, Σ, z), eachcol(re.z))
        end
    end
end

function _boot_lambdas(boot::MixedModelBootstrap, model::MixedModel,
                       groups::SymbolCollection)
    all = fnames(model)
    return [boot.λ[findfirst(==(g), all)] for g in groups]
end

_boot_re_scale(boot::MixedModelBootstrap, ::LinearMixedModel) = abs2.(boot.σ)
function _boot_re_scale(boot::MixedModelBootstrap, ::GeneralizedLinearMixedModel)
    return ones(length(boot.fits))
end

_has_random_slopes(model::MixedModel) = any(re -> size(re.λ, 1) > 1, model.reterms)

####
#### observation-level (residual) variance
####

"""
    _residual_variance(model, method)

The observation-level (residual) variance on the link scale.

For a `LinearMixedModel` this is just the residual variance. For a
`GeneralizedLinearMixedModel` there is no residual variance as such, and one of the
approximations catalogued by Nakagawa, Johnson and Schielzeth (2017) is used instead.
"""
function _residual_variance(model::LinearMixedModel, method::Symbol)
    method in (:auto, :theoretical) ||
        throw(ArgumentError("`method=$(repr(method))` is only meaningful for " *
                            "generalized linear mixed models; a linear mixed model has " *
                            "an actual residual variance."))
    return varest(model)
end

function _residual_variance(model::GeneralizedLinearMixedModel, method::Symbol)
    dispersion_parameter(model) &&
        throw(ArgumentError("GLMMs with dispersion parameters are not currently supported."))
    d = model.resp.d
    method = _resolve_method(d, method)
    return _residual_variance(d, Link(model.resp), method, model)
end

_resolve_method(d, method::Symbol) = method === :auto ? _default_method(d) : method
_default_method(::Union{Binomial,Bernoulli}) = :theoretical
_default_method(::Poisson) = :lognormal
function _default_method(d)
    return throw(ArgumentError("Family $(typeof(d)) currently unsupported, please file an issue."))
end

const _BINOMIAL_METHODS = (:theoretical, :observation_level)
const _POISSON_METHODS = (:delta, :lognormal, :trigamma)

function _residual_variance(d::Union{Binomial,Bernoulli}, link, method::Symbol, model)
    method in _BINOMIAL_METHODS ||
        throw(ArgumentError("`method=$(repr(method))` is not defined for a $(typeof(d)) " *
                            "model; use one of $(_BINOMIAL_METHODS)."))
    if method === :theoretical
        link isa LogitLink && return π^2 / 3
        link isa ProbitLink && return 1.0
        link isa CloglogLink && return π^2 / 6
        throw(ArgumentError("The theoretical observation-level variance is not defined " *
                            "for $(typeof(link)); use `method=:observation_level`."))
    end
    p = _mean_probability(model)
    link isa LogitLink && return 1 / (p * (1 - p))
    link isa ProbitLink &&
        return 2π * p * (1 - p) * abs2(exp(abs2(quantile(Normal(), p) / sqrt(2))))
    link isa CloglogLink && return p / abs2(log1p(-p)) / (1 - p)
    return throw(ArgumentError("`method=:observation_level` is not defined for $(typeof(link))."))
end

function _residual_variance(d::Poisson, link, method::Symbol, model)
    method in _POISSON_METHODS ||
        throw(ArgumentError("`method=$(repr(method))` is not defined for a Poisson " *
                            "model; use one of $(_POISSON_METHODS)."))
    link isa LogLink ||
        throw(ArgumentError("Only the log link is supported for Poisson models, got " *
                            "$(typeof(link))."))
    λ = _mean_rate(model)
    method === :delta && return 1 / λ
    method === :lognormal && return log1p(1 / λ)
    return trigamma(λ)
end

function _residual_variance(d, link, method::Symbol, model)
    return throw(ArgumentError("Family $(typeof(d)) currently unsupported, please file an issue."))
end

"""
    _mean_linear_predictor(model)

The average of the fixed-effects linear predictor, ``mean(Xβ)``.

Nakagawa et al. (2017) express the observation-level variance in terms of the intercept
of a corresponding intercept-only ("null") model. MixedModels.jl does not retain the
data needed to refit such a model, so the mean of the fixed-effects linear predictor is
used instead. The two agree exactly when the only fixed effect is the intercept, and
closely when the covariates are centered.
"""
_mean_linear_predictor(model::MixedModel) = mean(model.X * model.β)

"""
    _mean_rate(model)

The expected count ``λ = exp(η̄ + σ²/2)`` for a Poisson model, where ``σ²`` is the total
random-effects variance. The correction term accounts for Jensen's inequality when
back-transforming from the log scale.
"""
function _mean_rate(model::MixedModel)
    return exp(_mean_linear_predictor(model) + _group_var(model) / 2)
end

"""
    _mean_probability(model)

The expected probability for a binomial model, using the third-order approximation to
the mean of a logit-normal distribution used by Nakagawa et al. (2017).
"""
function _mean_probability(model::MixedModel)
    η = _mean_linear_predictor(model)
    σ² = _group_var(model)
    return logistic(η - 0.5 * σ² * tanh(η * (1 + 2 * exp(-0.5 * σ²)) / 6))
end

logistic(x) = inv(1 + exp(-x))

####
#### the ICC itself
####

"""
    icc(model::MixedModel, [groups]; kwargs...)
    icc(boot::MixedModelBootstrap, [family], [groups]; kwargs...)
    icc(table; target, rater, score, kwargs...)

Compute an intraclass correlation coefficient (ICC).

The name "ICC" covers several distinct coefficients; see the
[ICC documentation](@ref "Intraclass Correlation Coefficients") for the taxonomy and for
guidance on choosing between them. Two broad families are supported here, distinguished
by their first argument.

# Variance partitioning from a fitted model

Given a `MixedModel`, the ICC is the variance attributable to the `groups` divided by
the total variance. This quantity is also known as the *variance partition coefficient*
(VPC) and, in the ecological literature, as *repeatability*. It can be read as the
proportion of variance explainable by the grouping/nesting structure.

A single group can be given as a `Symbol`, e.g. `:subj`, and several as a collection,
e.g. `[:subj, :item]`. If no `groups` are given, all grouping variables are used. The
result aggregates across the groups given; for a per-group ICC, call `icc` once per
group.

- `conditional=false`: whether to include the variance of the fixed effects, `var(Xβ)`,
  in the denominator. The default, `conditional=false`, is often called the *adjusted*
  ICC and describes variance partitioning among the random terms alone;
  `conditional=true` gives the *conditional* (or *unadjusted*) ICC, the share of the
  total variance in the data attributable to the grouping factor.
- `method=:auto`: how to obtain the observation-level variance for a
  `GeneralizedLinearMixedModel`, which has no residual variance in the usual sense. For
  binomial models, `:theoretical` (the default) uses the variance of the latent
  distribution -- ``π²/3`` for the logit link, ``1`` for probit, ``π²/6`` for
  complementary log-log -- and `:observation_level` uses the variance of the response
  distribution instead. For Poisson models, `:delta`, `:lognormal` (the default) and
  `:trigamma` are the three approximations of Nakagawa et al. (2017). `:simulation`
  computes the variance partition coefficient on the response scale by simulating from
  the fitted model, following Goldstein et al. (2002); it accepts `nsim` and `rng`.
- `slopes=:mean`: how to summarize a grouping factor whose random-effects term has more
  than one column. A model with random slopes has no single random-effects variance --
  the variance contributed by the grouping factor depends on the covariate values -- so
  `:mean` uses the extension of Johnson (2014), averaging the random-effects variance
  over the observations. `:diagonal` instead sums the variances of the random-effects
  terms, ignoring the covariances and the model matrix; this was the behaviour before
  version 3.0 and is retained for reproducing older results. The two agree exactly for
  random intercepts.
- `groupmean=false`: if `true`, return the reliability of the *group means* rather than
  the ICC itself, applying the Spearman-Brown formula. This is the coefficient called
  ICC(2) in the organizational literature following Bliese (2000), as distinct from
  ICC(1), the ICC proper. The number of observations per group `k` defaults to the
  harmonic mean of the group sizes and can be overridden.

Linear mixed models fit with `REML=true` reproduce the classical ANOVA variance
components on balanced data. MixedModels.jl fits with maximum likelihood by default,
which biases the individual variance components downwards; because the ICC is a ratio,
though, this does not translate into a predictable direction for the ICC itself. Refit
with `REML=true` to match a classical ANOVA-based calculation.

# Bootstrap

When a `MixedModelBootstrap` is passed, a vector of ICC values, one per bootstrap
iteration, is returned; `confint` computes an interval from it. Because
`MixedModelBootstrap` does not store the model family, the family must be given for
generalized linear mixed models, e.g. `Bernoulli()` or `Poisson()`.

The bootstrap stores only variances and correlations, so a model with random slopes
(or `conditional=true`, which needs the fixed-effects model matrix) additionally
requires the original model, passed as the `model` keyword argument.

# Inter-rater reliability from a table

Given any Tables.jl-compatible table of ratings in long format, together with the
`target`, `rater` and `score` column names, `icc` computes the inter-rater reliability
coefficients of Shrout and Fleiss (1979) and McGraw and Wong (1996). See
[`InterraterICC`](@ref) and the
[ICC documentation](@ref "Intraclass Correlation Coefficients").

# Examples

```julia
icc(model)                            # all grouping variables
icc(model, :subj)                     # one grouping variable
icc(model, :subj; conditional=true)   # include fixed-effects variance
icc(model, :subj; groupmean=true)     # Bliese's ICC(2)
icc(glmm, :subj; method=:simulation)  # response-scale VPC

icc(ratings; target=:subj, rater=:judge, score=:rating)
```

# References

The [ICC documentation](@ref "Intraclass Correlation Coefficients") lists the full
bibliography.
"""
function icc(model::MixedModel, groups::Union{Symbol,SymbolCollection};
             conditional::Bool=false, method::Symbol=:auto, slopes::Symbol=:mean,
             nsim::Integer=10_000, rng::AbstractRNG=Random.default_rng(),
             groupmean::Bool=false, k=nothing)
    value = if method === :simulation
        _simulation_vpc(model, groups; conditional, nsim, rng, slopes)
    else
        σ²res = _residual_variance(model, method)
        σ²α = _group_var(model, groups; slopes)
        σ² = σ²res + _group_var(model; slopes) + (conditional ? _fixef_var(model) : 0.0)
        σ²α / σ²
    end
    groupmean || return value
    return _spearman_brown(value, something(k, _group_size(model, groups)))
end

function icc(model::MixedModel; kwargs...)
    return icc(model, fnames(model); kwargs...)
end

_fixef_var(model::MixedModel) = var(model.X * model.β)

"""
    _spearman_brown(ρ, k)

Step up a single-measurement reliability to the reliability of the mean of `k`
measurements: ``kρ / (1 + (k-1)ρ)``.
"""
_spearman_brown(ρ, k) = k * ρ / (1 + (k - 1) * ρ)

"""
    _group_size(model, groups)

The harmonic mean of the number of observations per level of a grouping factor, used as
the effective group size when stepping up to a group-mean reliability. The harmonic mean
is the appropriate summary for unequal group sizes because reliability is a function of
`1/k`.
"""
function _group_size(model::MixedModel, group::Symbol)
    refs = _reterm(model, group).refs
    counts = zeros(Int, maximum(refs))
    for r in refs
        counts[r] += 1
    end
    return length(counts) / sum(inv, counts)
end

function _group_size(model::MixedModel, groups::SymbolCollection)
    length(groups) == 1 && return _group_size(model, only(groups))
    return throw(ArgumentError("A group-mean reliability is defined for a single grouping " *
                               "variable, but $(length(groups)) were given: " *
                               "$(collect(groups)). Call `icc` separately for each, or pass " *
                               "`k` explicitly."))
end

####
#### simulation-based VPC (Goldstein, Browne & Rasbash 2002)
####

"""
    _simulation_vpc(model, groups; conditional, nsim, rng)

The variance partition coefficient on the response scale, estimated by simulation.

For a generalized linear mixed model the link-scale variance partition depends on an
arbitrary choice of observation-level variance. Goldstein et al. (2002) avoid this by
simulating random effects from the fitted model, transforming to the response scale, and
partitioning the variance there.

The random effects of the grouping factors of interest are drawn in an outer loop and
those of the remaining grouping factors in an inner loop, so that the numerator is the
variance of the response mean conditional on the groups of interest, marginalizing over
the others.
"""
function _simulation_vpc(model::MixedModel, groups; conditional::Bool=false,
                         nsim::Integer=10_000, rng::AbstractRNG=Random.default_rng(),
                         slopes::Symbol=:mean)
    model isa GeneralizedLinearMixedModel ||
        throw(ArgumentError("`method=:simulation` is only meaningful for generalized " *
                            "linear mixed models; for a linear mixed model the " *
                            "variance partition is exact."))
    groups = groups isa Symbol ? (groups,) : groups
    scale = _re_scale(model)

    # total variance of each grouping factor, averaged over observations
    target = [_mean_re_variance(_reterm(model, g), scale, slopes) for g in groups]
    other = [_mean_re_variance(re, scale, slopes)
             for re in model.reterms if !(fname(re) in groups)]

    σ²target = sum(target; init=0.0)
    σ²other = sum(other; init=0.0)
    η̄ = _mean_linear_predictor(model)
    linkfun = Link(model.resp)
    d = model.resp.d

    ninner = iszero(σ²other) ? 1 : max(1, isqrt(Int(nsim)))
    nouter = Int(nsim)

    inner = iszero(σ²other) ? zeros(ninner) : sqrt(σ²other) .* randn(rng, ninner)

    condmeans = Vector{Float64}(undef, nouter)
    condvars = Vector{Float64}(undef, nouter)
    allmeans = Float64[]
    sizehint!(allmeans, nouter * ninner)

    for s in 1:nouter
        u = sqrt(σ²target) * randn(rng)
        m = 0.0
        v = 0.0
        for t in 1:ninner
            μ = linkinv(linkfun, η̄ + u + inner[t])
            m += μ
            v += _conditional_variance(d, μ)
            push!(allmeans, μ)
        end
        condmeans[s] = m / ninner
        condvars[s] = v / ninner
    end

    σ²between = var(condmeans)
    σ²within = mean(condvars) + var(allmeans) - σ²between
    # var(allmeans) is the total variance of the conditional mean across *all* random
    # effects; subtracting the between-group part leaves the contribution of the other
    # grouping factors, which belongs in the denominator but not the numerator.

    denominator = σ²between + σ²within
    conditional && (denominator += _response_scale_fixef_var(model))
    return σ²between / denominator
end

_conditional_variance(::Union{Bernoulli,Binomial}, μ) = μ * (1 - μ)
_conditional_variance(::Poisson, μ) = μ
function _conditional_variance(d, μ)
    return throw(ArgumentError("Family $(typeof(d)) currently unsupported, please file an issue."))
end

function _response_scale_fixef_var(model::MixedModel)
    return var(linkinv.(Ref(Link(model.resp)), model.X * model.β))
end

####
#### bootstrap methods
####

# TODO: upstream
# fnames(boot::MixedModelBootstrap) = propertynames(boot.fcnames)
function icc(boot::MixedModelBootstrap; kwargs...)
    return icc(boot, propertynames(boot.fcnames); kwargs...)
end
function icc(boot::MixedModelBootstrap, family; kwargs...)
    return icc(boot, family, propertynames(boot.fcnames); kwargs...)
end

function icc(boot::MixedModelBootstrap, groups::Union{Symbol,SymbolCollection};
             model=nothing, conditional::Bool=false, slopes::Symbol=:mean,
             groupmean::Bool=false, k=nothing)
    all(ismissing, boot.σ) &&
        throw(ArgumentError("Bootstrapping GLMM requires specifying the family."))
    return _icc_bootstrap(boot, groups, abs2.(boot.σ), model, conditional, slopes,
                          groupmean, k)
end

function icc(boot::MixedModelBootstrap, family, groups::Union{Symbol,SymbolCollection};
             model=nothing, method::Symbol=:auto, conditional::Bool=false,
             slopes::Symbol=:mean, groupmean::Bool=false, k=nothing)
    method = _resolve_method(family, method)
    σ²res = if isnothing(model)
        _residual_variance(family, _boot_link(family, method), method, nothing)
    else
        _residual_variance(model, method)
    end
    return _icc_bootstrap(boot, groups, σ²res, model, conditional, slopes, groupmean, k)
end

# Only the family is known, so only the link-free (theoretical) variances are available.
function _boot_link(family, method::Symbol)
    method in _POISSON_METHODS &&
        throw(ArgumentError("`method=$(repr(method))` needs the fitted model to compute " *
                            "the expected rate; pass it as the `model` keyword argument."))
    method === :observation_level &&
        throw(ArgumentError("`method=:observation_level` needs the fitted model to " *
                            "compute the expected probability; pass it as the `model` " *
                            "keyword argument."))
    return LogitLink()
end

function _icc_bootstrap(boot, groups, σ²res, model, conditional::Bool, slopes::Symbol,
                        groupmean::Bool, k)
    # `slopes=:diagonal` sums the random-effect variances, which is exactly what the
    # bootstrap stores, so it needs no model even for a random-slope fit
    if !isnothing(model) && _has_random_slopes(model) && slopes === :mean
        σ²α = _boot_group_var(boot, model, groups; slopes)
        σ² = σ²res .+ _boot_group_var(boot, model, fnames(model); slopes)
    else
        if isnothing(model) && slopes === :mean && _boot_has_random_slopes(boot)
            throw(ArgumentError("This bootstrap is from a model with random slopes, for " *
                                "which the Johnson (2014) random-effects variance " *
                                "depends on the random-effects model matrix. Pass the " *
                                "original model as the `model` keyword argument, or use " *
                                "`slopes=:diagonal` for the sum of the random-effect " *
                                "variances."))
        end
        σ²α = _group_var(boot.σs, groups)
        σ² = σ²res .+ _group_var(boot.σs)
    end
    if conditional
        isnothing(model) &&
            throw(ArgumentError("`conditional=true` needs the fixed-effects model " *
                                "matrix; pass the original model as the `model` keyword " *
                                "argument."))
        σ² = σ² .+ _boot_fixef_var(boot, model)
    end
    values = σ²α ./ σ²
    groupmean || return IccBootstrap(values)
    isnothing(model) && isnothing(k) &&
        throw(ArgumentError("`groupmean=true` needs the group sizes; pass either the " *
                            "original model as `model` or the group size as `k`."))
    kk = something(k, _group_size(model, groups))
    return IccBootstrap(_spearman_brown.(values, kk))
end

_boot_has_random_slopes(boot) = any(length(v) > 1 for v in boot.fcnames)

function _boot_fixef_var(boot::MixedModelBootstrap, model::MixedModel)
    coefs = Dict{Int,Vector{Float64}}()
    order = Symbol.(coefnames(model))
    for row in Tables.rows(boot.β)
        v = get!(() -> zeros(length(order)), coefs, row.iter)
        v[findfirst(==(row.coefname), order)] = row.β
    end
    return [var(model.X * coefs[i]) for i in sort!(collect(keys(coefs)))]
end

####
#### tabular (inter-rater) methods
####

function icc(tbl; target::Symbol, rater::Symbol, score::Symbol,
             model=nothing, type=nothing, unit=nothing,
             estimator::Symbol=:anova, reml::Bool=true, level::Real=0.95)
    Tables.istable(tbl) ||
        throw(ArgumentError("Expected a Tables.jl-compatible table, got $(typeof(tbl))."))
    estimator in (:anova, :lmm) ||
        throw(ArgumentError("`estimator` must be either :anova or :lmm, got " *
                            "$(repr(estimator))."))
    0 < level < 1 ||
        throw(ArgumentError("`level` must lie strictly between 0 and 1, got $level."))

    ms, n, k = if estimator === :lmm
        _meansquares_lmm(tbl, target, rater, score; reml)
    else
        Y = _rating_matrix(tbl, target, rater, score)
        _meansquares(Y), size(Y, 1), size(Y, 2)
    end

    # with none of model/type/unit specified, return all six coefficients, the workflow
    # recommended by Liljequist et al. (2019)
    if isnothing(model) && isnothing(type) && isnothing(unit)
        coefs = [_interrater_icc(ms, n, k, m, t, u, level, estimator)
                 for (m, t, u) in ((:oneway, :agreement, :single),
                                   (:twoway, :agreement, :single),
                                   (:twoway, :consistency, :single),
                                   (:oneway, :agreement, :average),
                                   (:twoway, :agreement, :average),
                                   (:twoway, :consistency, :average))]
        return InterraterICCTable(coefs)
    end

    model = something(model, :twoway)
    type = something(type, :agreement)
    unit = something(unit, :single)
    model in (:oneway, :twoway) ||
        throw(ArgumentError("`model` must be either :oneway or :twoway, got " *
                            "$(repr(model))."))
    type in (:agreement, :consistency) ||
        throw(ArgumentError("`type` must be either :agreement or :consistency, got " *
                            "$(repr(type))."))
    unit in (:single, :average) ||
        throw(ArgumentError("`unit` must be either :single or :average, got " *
                            "$(repr(unit))."))
    return _interrater_icc(ms, n, k, model, type, unit, level, estimator)
end

####
#### confidence intervals
####

"""
    confint(icc_boot::IccBootstrap; level::Real=0.95, method=:shortest)

Compute a bootstrap confidence interval for an [`IccBootstrap`](@ref), i.e. the
result of calling [`icc`](@ref) on a `MixedModelBootstrap`.

The keyword argument `level` is the confidence level (0.95 by default). The keyword
argument `method` determines whether the `:shortest`, i.e. highest density, interval
is used (the default) or the `:equaltail`, i.e. quantile-based, interval is used --
matching the behavior of `confint(::MixedModelBootstrap)` from MixedModels.jl.
"""
function StatsBase.confint(icc_boot::IccBootstrap; level::Real=0.95, method=:shortest)
    method in (:shortest, :equaltail) ||
        throw(ArgumentError("`method` must be either :shortest or :equaltail."))
    method === :shortest && return shortestcovint(icc_boot, level)
    tails = ((1 - level) / 2, (1 + level) / 2)
    return Tuple(quantile(icc_boot, tails))
end
