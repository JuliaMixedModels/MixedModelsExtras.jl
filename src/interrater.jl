####
#### Inter-rater reliability ICCs (Shrout & Fleiss 1979; McGraw & Wong 1996)
####
#### These are computed from a "targets × raters" design rather than from a
#### fitted mixed model, although a linear mixed model can be used as the
#### estimator of the underlying variance components.
####

"""
    InterraterICC{T<:AbstractFloat}

A single inter-rater reliability intraclass correlation coefficient, as computed
by [`icc`](@ref) from a table of ratings.

In addition to the point estimate and its confidence interval, this object carries
the quantities that Liljequist, Elfving and Skavberg Roaldsen (2019) recommend
reporting alongside an ICC: the estimated variance components and a test for the
presence of systematic bias between raters.

# Fields
- `name`: the McGraw & Wong (1996) name, e.g. `"ICC(A,1)"`.
- `alias`: the corresponding Shrout & Fleiss (1979) name, e.g. `"ICC(2,1)"`.
- `estimate`: the point estimate.
- `lower`, `upper`, `level`: the confidence interval and its level. `NaN` when the
  estimator does not provide an exact interval (see `confint`).
- `F`, `df1`, `df2`, `p`: the F test of the null hypothesis that the ICC is zero.
- `varcomp`: a `NamedTuple` `(; subject, rater, residual)` of the estimated variance
  components ``σ̂²_r``, ``σ̂²_c`` and ``σ̂²_v``.
- `meansquares`: a `NamedTuple` `(; subject, rater, residual, within)` of the mean
  squares (`MSR`, `MSC`, `MSE`, `MSW`; in the notation of Liljequist et al., `MSBS`,
  `MSBM`, `MSE` and `MSWS`).
- `biastest`: a `NamedTuple` `(; F, df1, df2, p)` for the F test `MSC / MSE` of the
  null hypothesis that there is no systematic difference between raters.
- `n`, `k`: the number of targets and of raters.
- `estimator`: `:anova` or `:lmm`.

See also [`icc`](@ref), [`InterraterICCTable`](@ref).
"""
struct InterraterICC{T<:AbstractFloat}
    name::String
    alias::String
    estimate::T
    lower::T
    upper::T
    level::Float64
    F::T
    df1::T
    df2::T
    p::T
    varcomp::NamedTuple{(:subject, :rater, :residual),NTuple{3,T}}
    meansquares::NamedTuple{(:subject, :rater, :residual, :within),NTuple{4,T}}
    biastest::NamedTuple{(:F, :df1, :df2, :p),NTuple{4,T}}
    n::Int
    k::Int
    estimator::Symbol
end

"""
    InterraterICCTable{T}

The full set of six inter-rater ICCs computed from the same table of ratings, as
returned by [`icc`](@ref) when no `model`, `type` or `unit` is specified.

Computing all six and comparing them is the workflow recommended by Liljequist,
Elfving and Skavberg Roaldsen (2019): rather than committing to one ANOVA model in
advance, fit all of them and let the data show whether systematic rater bias is
present. Iterating over this object yields the individual [`InterraterICC`](@ref)
values, and they can also be accessed by name:

```julia
result = icc(ratings; target=:subj, rater=:judge, score=:y)
result.ICCA1     # absolute agreement, single rater
result[:ICC11]   # Shrout & Fleiss aliases work too
```
"""
struct InterraterICCTable{T<:AbstractFloat}
    coefficients::Vector{InterraterICC{T}}
end

Base.length(t::InterraterICCTable) = length(t.coefficients)
Base.iterate(t::InterraterICCTable, s...) = iterate(t.coefficients, s...)
Base.eltype(::Type{InterraterICCTable{T}}) where {T} = InterraterICC{T}

# canonical key => (McGraw-Wong name, Shrout-Fleiss name)
const _ICC_NAMES = (ICC1=("ICC(1)", "ICC(1,1)"),
                    ICCA1=("ICC(A,1)", "ICC(2,1)"),
                    ICCC1=("ICC(C,1)", "ICC(3,1)"),
                    ICC1k=("ICC(k)", "ICC(1,k)"),
                    ICCAk=("ICC(A,k)", "ICC(2,k)"),
                    ICCCk=("ICC(C,k)", "ICC(3,k)"))

# Shrout & Fleiss spellings accepted as lookup keys
const _ICC_ALIASES = (ICC11=:ICC1, ICC21=:ICCA1, ICC31=:ICCC1,
                      ICC2k=:ICCAk, ICC3k=:ICCCk)

function Base.getindex(t::InterraterICCTable, key::Symbol)
    canonical = get(_ICC_ALIASES, key, key)
    idx = findfirst(==(canonical), keys(_ICC_NAMES))
    isnothing(idx) &&
        throw(KeyError(key))
    return t.coefficients[idx]
end

Base.getindex(t::InterraterICCTable, i::Integer) = t.coefficients[i]
function Base.propertynames(::InterraterICCTable)
    return (keys(_ICC_NAMES)..., keys(_ICC_ALIASES)...,
            :coefficients)
end

function Base.getproperty(t::InterraterICCTable, key::Symbol)
    key === :coefficients && return getfield(t, :coefficients)
    return t[key]
end

####
#### mean squares
####

"""
    _rating_matrix(tbl, target, rater, score)

Reshape a long-format table into an `n × k` matrix of ratings (targets in rows,
raters in columns). Throws an informative error if the design is not complete.
"""
function _rating_matrix(tbl, target::Symbol, rater::Symbol, score::Symbol)
    cols = Tables.columns(tbl)
    for col in (target, rater, score)
        Tables.columnindex(cols, col) == 0 &&
            throw(ArgumentError("Column $(repr(col)) not found in the table. " *
                                "Available columns: $(collect(Tables.columnnames(cols)))."))
    end
    tvals = Tables.getcolumn(cols, target)
    rvals = Tables.getcolumn(cols, rater)
    yvals = Tables.getcolumn(cols, score)

    tlevels = unique(tvals)
    rlevels = unique(rvals)
    n = length(tlevels)
    k = length(rlevels)
    n > 1 || throw(ArgumentError("At least two targets are required, got $n."))
    k > 1 || throw(ArgumentError("At least two raters are required, got $k."))

    tindex = Dict(lev => i for (i, lev) in enumerate(tlevels))
    rindex = Dict(lev => j for (j, lev) in enumerate(rlevels))

    T = float(eltype(yvals))
    Y = Matrix{T}(undef, n, k)
    filled = falses(n, k)
    for (t, r, y) in zip(tvals, rvals, yvals)
        ismissing(y) && continue
        i, j = tindex[t], rindex[r]
        filled[i, j] &&
            throw(ArgumentError("Duplicate rating for target $(repr(t)) by rater " *
                                "$(repr(r)). Each target-rater combination must occur " *
                                "at most once."))
        Y[i, j] = y
        filled[i, j] = true
    end
    all(filled) ||
        throw(ArgumentError("The ANOVA estimator requires a complete, balanced " *
                            "targets × raters design, but $(count(!, filled)) of the " *
                            "$(n * k) target-rater combinations are missing. Use " *
                            "`estimator=:lmm` to handle unbalanced data."))
    return Y
end

"""
    _meansquares(Y::AbstractMatrix)

Two-way ANOVA mean squares for an `n × k` matrix of ratings. Returns
`(; subject, rater, residual, within)`, i.e. `MSR`, `MSC`, `MSE` and `MSW` in the
notation of Shrout & Fleiss (1979) (`MSBS`, `MSBM`, `MSE` and `MSWS` in that of
Liljequist et al. 2019).
"""
function _meansquares(Y::AbstractMatrix{T}) where {T}
    n, k = size(Y)
    grand = mean(Y)
    rowmeans = vec(mean(Y; dims=2))
    colmeans = vec(mean(Y; dims=1))

    ssr = k * sum(abs2, rowmeans .- grand)
    ssc = n * sum(abs2, colmeans .- grand)
    sse = zero(T)
    ssw = zero(T)
    for j in axes(Y, 2), i in axes(Y, 1)
        sse += abs2(Y[i, j] - rowmeans[i] - colmeans[j] + grand)
        ssw += abs2(Y[i, j] - rowmeans[i])
    end

    return (; subject=ssr / (n - 1),
            rater=ssc / (k - 1),
            residual=sse / ((n - 1) * (k - 1)),
            within=ssw / (n * (k - 1)))
end

"""
    _meansquares_lmm(tbl, target, rater, score; reml=true)

Estimate the mean squares implied by a linear mixed model with crossed random
intercepts for target and rater. Unlike [`_meansquares`](@ref), this works for
unbalanced and incomplete designs, and for a balanced design with an interior
optimum it reproduces the ANOVA mean squares when `reml=true`, up to the tolerance
of the optimizer.
"""
function _meansquares_lmm(tbl, target::Symbol, rater::Symbol, score::Symbol;
                          reml::Bool=true)
    cols = Tables.columns(tbl)
    for col in (target, rater, score)
        Tables.columnindex(cols, col) == 0 &&
            throw(ArgumentError("Column $(repr(col)) not found in the table. " *
                                "Available columns: $(collect(Tables.columnnames(cols)))."))
    end
    n = length(unique(Tables.getcolumn(cols, target)))
    k = length(unique(Tables.getcolumn(cols, rater)))

    form = FormulaTerm(Term(score),
                       (ConstantTerm(1),
                        RandomEffectsTerm(ConstantTerm(1), Term(target)),
                        RandomEffectsTerm(ConstantTerm(1), Term(rater))))
    model = fit(MixedModel, form, tbl; REML=reml, progress=false)

    vc = VarCorr(model)
    σ²r = abs2(only(vc.σρ[target].σ))
    σ²c = abs2(only(vc.σρ[rater].σ))
    σ²v = varest(model)

    # invert the expected mean squares to stay on the ANOVA scale, so that the
    # same ICC and variance-component formulas apply to both estimators
    return (; subject=k * σ²r + σ²v,
            rater=n * σ²c + σ²v,
            residual=σ²v,
            within=σ²c + σ²v), n, k
end

####
#### the coefficients themselves
####

function _interrater_icc(ms, n::Int, k::Int, model::Symbol, type::Symbol, unit::Symbol,
                         level::Real, estimator::Symbol)
    msr, msc, mse, msw = ms.subject, ms.rater, ms.residual, ms.within

    # variance components, following Liljequist et al. (2019, eqs. 11 and 26).
    # the one-way model cannot separate rater bias from noise, so σ²c is folded
    # into the residual there.
    varcomp = if model === :oneway
        (; subject=(msr - msw) / k, rater=zero(msr), residual=msw)
    else
        (; subject=(msr - mse) / k, rater=(msc - mse) / n, residual=mse)
    end

    single = unit === :single
    estimate, Fstat, df1, df2 = if model === :oneway
        est = single ? (msr - msw) / (msr + (k - 1) * msw) : (msr - msw) / msr
        (est, msr / msw, float(n - 1), float(n * (k - 1)))
    elseif type === :consistency
        est = single ? (msr - mse) / (msr + (k - 1) * mse) : (msr - mse) / msr
        (est, msr / mse, float(n - 1), float((n - 1) * (k - 1)))
    else # :agreement
        est = single ? (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n) :
              (msr - mse) / (msr + (msc - mse) / n)
        (est, msr / mse, float(n - 1), float((n - 1) * (k - 1)))
    end

    p = ccdf(FDist(df1, df2), Fstat)

    lower, upper = if estimator === :anova
        _interrater_interval(ms, n, k, model, type, unit, estimate, Fstat, df1, df2, level)
    else
        (oftype(estimate, NaN), oftype(estimate, NaN))
    end

    # Liljequist et al.'s test for the presence of systematic rater bias
    fbias, dfb1, dfb2 = msc / mse, float(k - 1), float((n - 1) * (k - 1))
    biastest = (; F=fbias, df1=dfb1, df2=dfb2, p=ccdf(FDist(dfb1, dfb2), fbias))

    key = _icc_key(model, type, unit)
    name, alias = _ICC_NAMES[key]

    return InterraterICC(name, alias, estimate, lower, upper, Float64(level),
                         Fstat, df1, df2, p,
                         map(x -> oftype(estimate, x), varcomp),
                         map(x -> oftype(estimate, x), ms),
                         map(x -> oftype(estimate, x), biastest),
                         n, k, estimator)
end

function _icc_key(model::Symbol, type::Symbol, unit::Symbol)
    single = unit === :single
    model === :oneway && return single ? :ICC1 : :ICC1k
    type === :consistency && return single ? :ICCC1 : :ICCCk
    return single ? :ICCA1 : :ICCAk
end

"""
    _interrater_interval(...)

Exact confidence limits for the ANOVA estimators, following Shrout & Fleiss (1979)
and McGraw & Wong (1996). The one-way and consistency forms invert the usual F
statistic directly; the absolute-agreement form requires a Satterthwaite-style
approximate degrees of freedom.
"""
function _interrater_interval(ms, n::Int, k::Int, model::Symbol, type::Symbol,
                              unit::Symbol, estimate, Fstat, df1, df2, level::Real)
    α = 1 - level
    msr, msc, mse = ms.subject, ms.rater, ms.residual
    single = unit === :single

    if model === :oneway || type === :consistency
        fl = Fstat / quantile(FDist(df1, df2), 1 - α / 2)
        fu = Fstat * quantile(FDist(df2, df1), 1 - α / 2)
        single && return ((fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1))
        return (1 - 1 / fl, 1 - 1 / fu)
    end

    # absolute agreement: McGraw & Wong (1996), Table 7. The interval is always
    # derived for the single-measurement form and then stepped up with the
    # Spearman-Brown formula for the average-measurement form.
    r = single ? estimate : estimate / (k - (k - 1) * estimate)
    fj = msc / mse
    num = (n - 1) * (k - 1) * abs2(k * r * fj + n * (1 + (k - 1) * r) - k * r)
    den = (n - 1) * k^2 * abs2(r) * abs2(fj) + abs2(n * (1 + (k - 1) * r) - k * r)
    ν = num / den

    fu = quantile(FDist(df1, ν), 1 - α / 2)
    fl = quantile(FDist(ν, df1), 1 - α / 2)

    lower = n * (msr - fu * mse) /
            (fu * (k * msc + (k * n - k - n) * mse) + n * msr)
    upper = n * (fl * msr - mse) /
            (k * msc + (k * n - k - n) * mse + n * fl * msr)

    single && return (lower, upper)
    return (k * lower / (1 + (k - 1) * lower), k * upper / (1 + (k - 1) * upper))
end

####
#### show methods
####

function Base.show(io::IO, ::MIME"text/plain", x::InterraterICC)
    println(io, "$(x.name) [$(x.alias)]: ", _fmt(x.estimate))
    if !isnan(x.lower)
        println(io, "  $(round(Int, 100 * x.level))% CI: (", _fmt(x.lower), ", ",
                _fmt(x.upper), ")")
    end
    println(io, "  F($(_fmt(x.df1)), $(_fmt(x.df2))) = ", _fmt(x.F), ", p = ", _fmt(x.p))
    println(io, "  n = $(x.n) targets, k = $(x.k) raters, estimator = $(repr(x.estimator))")
    vc = x.varcomp
    println(io, "  variance components: subject = ", _fmt(vc.subject),
            ", rater = ", _fmt(vc.rater), ", residual = ", _fmt(vc.residual))
    b = x.biastest
    print(io, "  rater bias: F($(_fmt(b.df1)), $(_fmt(b.df2))) = ", _fmt(b.F),
          ", p = ", _fmt(b.p))
    return nothing
end

Base.show(io::IO, x::InterraterICC) = show(io, MIME("text/plain"), x)

function Base.show(io::IO, ::MIME"text/plain", t::InterraterICCTable)
    first_coef = first(t.coefficients)
    println(io, "Intraclass correlation coefficients ",
            "(n = $(first_coef.n) targets, k = $(first_coef.k) raters)")
    pct = round(Int, 100 * first_coef.level)
    println(io, rpad("", 10), rpad("Shrout-Fleiss", 15), rpad("estimate", 12),
            "$(pct)% CI")
    for c in t.coefficients
        ci = isnan(c.lower) ? "" : "(" * _fmt(c.lower) * ", " * _fmt(c.upper) * ")"
        println(io, rpad(c.name, 10), rpad(c.alias, 15), rpad(_fmt(c.estimate), 12), ci)
    end
    b = first_coef.biastest
    print(io, "rater bias: F($(_fmt(b.df1)), $(_fmt(b.df2))) = ", _fmt(b.F),
          ", p = ", _fmt(b.p))
    return nothing
end

Base.show(io::IO, t::InterraterICCTable) = show(io, MIME("text/plain"), t)

_fmt(x::Integer) = string(x)
_fmt(x::AbstractFloat) = isnan(x) ? "NaN" : string(round(x; digits=4))

"""
    confint(x::InterraterICC; level=nothing)

The confidence interval for an inter-rater [`icc`](@ref), as a `(lower, upper)` tuple.

The interval is computed when the ICC itself is computed, because the exact intervals
of Shrout & Fleiss (1979) and McGraw & Wong (1996) depend on the ANOVA mean squares.
Passing `level` therefore only checks that the requested level matches the one that
was used; to obtain a different level, recompute the ICC with `level` set there.

Exact intervals are available only for `estimator=:anova`. For `estimator=:lmm`,
use `MixedModels.parametricbootstrap` on the underlying model instead.
"""
function StatsBase.confint(x::InterraterICC; level=nothing)
    if isnan(x.lower)
        throw(ArgumentError("Exact confidence intervals are only available for " *
                            "`estimator=:anova`; this ICC was computed with " *
                            "`estimator=$(repr(x.estimator))`. Use " *
                            "`MixedModels.parametricbootstrap` for an interval instead."))
    end
    if !isnothing(level) && level != x.level
        throw(ArgumentError("This ICC was computed at level $(x.level); recompute it " *
                            "with `icc(...; level=$(level))` to obtain that interval."))
    end
    return (x.lower, x.upper)
end
