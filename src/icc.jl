const SymbolCollection = Union{Symbol,Tuple{Symbol,Vararg{Symbol}},AbstractVector{Symbol}}

function _group_var(vc::VarCorr, group::Symbol)
    return sum(abs2, getproperty(getproperty(vc.σρ, group), :σ))
end

function _group_var(vc::VarCorr)
    groups = collect(propertynames(vc.σρ))
    return _group_var(vc, groups)
end

# TODO rework this so that _split_by_iter is only called once
function _group_var(vc, groups::SymbolCollection)
    return sum(_group_var.(Ref(vc), groups))
end

# for MixedModelBootstrap
# σtbl is boot.σs
function _group_var(tbl, group::Symbol)
    vals = map(values(_split_by_iter(tbl))) do iter
        return sum(Tables.rows(iter)) do row
            return row.group == group ? abs2(row.σ) : 0
        end
    end
    return vals
end

function _group_var(tbl)
    groups = unique((row.group for row in Tables.rows(tbl)))
    return _group_var(tbl, groups)
end

function _split_by_iter(tbl)
    d = Dict{Int,Vector}()
    for row in Tables.rows(tbl)
        vv = get!(d, row.iter, Vector{Any}())
        push!(vv, row)
    end
    return d
end

"""
    icc(model::MixedModel, [groups])
    icc(boot::MixedModelBootstrap, [family], [groups])

Compute the intra-class correlation coefficient (ICC) for a mixed model.

The ICC is defined as the variance attributable to the `groups` divided by
the total variance from all groups and the observation-level (residual) variance.
In other words, the ICC can be interpreted as the proportion of the variance explainable
by the grouping/nesting structure.

A single `group` can be specified as a Symbol, e.g. `:subj` or a number of groups
can be specified as an array: `[:subj, :item]`. If no `groups` are specified, then
all grouping variables are used.

When a `MixedModelBootstrap` is passed, a vector of ICC values for each
bootstrap iteration is returned. Because `MixedModelBootstrap` does not
store the associated model family for generalized linear mixed models,
the family must be specified (e.g., `Bernoulli()`, `Poisson()`). A shortest
coverage interval can be computed with `MixedModels.shortestcovint`.

!!! note
    The value returned here is sometimes called the "adjusted ICC" and does not take
    the variance of the fixed effects into account (the "conditional ICC").

!!! note
    The result returned aggregates across groups. If you require the ICC for
    each group separately, then you must call `icc` separately for each group.
"""
function icc(model::GeneralizedLinearMixedModel,
             groups::Union{Symbol,SymbolCollection})
    dispersion_parameter(model) &&
        throw(ArgumentError("GLMMs with dispersion parameters are not currently supported."))

    σ²res = _residual_variance(model.resp.d)

    return _icc(VarCorr(model), groups, σ²res)
end

_residual_variance(::Union{Binomial,Bernoulli}) = π^2 / 3
_residual_variance(::Poisson) = 1.0
function _residual_variance(d::Any)
    throw(ArgumentError("Family $(typeof(d)) currently unsupported, please file an issue."))
end

function icc(model::LinearMixedModel,
             groups::Union{Symbol,SymbolCollection})
    return _icc(VarCorr(model), groups, varest(model))
end

icc(model::MixedModel) = icc(model, fnames(model))

function _icc(vc::VarCorr, groups::Union{Symbol,SymbolCollection}, σ²res)
    σ²_α = _group_var(vc, groups) # random effect(s) in numerator
    # TODO: don't recompute this for groups already in the above
    σ² = σ²res + _group_var(vc)
    return σ²_α / σ²
end

# TODO: upstream
# fnames(boot::MixedModelBootstrap) = propertynames(boot.fcnames)
icc(boot::MixedModelBootstrap) = icc(boot, propertynames(boot.fcnames))
icc(boot::MixedModelBootstrap, family) = icc(boot, family, propertynames(boot.fcnames))

function icc(boot::MixedModelBootstrap,
             groups::Union{Symbol,SymbolCollection})
    all(ismissing, boot.σ) &&
        throw(ArgumentError("Bootstrapping GLMM requires specifying the family."))
    return _icc(boot.σs, groups, abs2.(boot.σ))
end

function icc(boot::MixedModelBootstrap, family,
             groups::Union{Symbol,SymbolCollection})
    σ²res = _residual_variance(family)
    return _icc(boot.σs, groups, σ²res)
end

function _icc(tbl, groups::Union{Symbol,SymbolCollection}, σ²res)
    σ²_α = _group_var(tbl, groups) # random effect(s) in numerator
    # TODO: don't recompute this for groups already in the above
    σ² = σ²res .+ _group_var(tbl)
    return σ²_α ./ σ²
end
