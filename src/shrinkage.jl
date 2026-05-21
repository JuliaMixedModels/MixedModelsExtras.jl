"""
    _ranef(m::MixedModel, θref; uscale::Bool=false)

Compute the conditional modes at θref.

!!! warning
    This function is **not** thread safe because it temporarily mutates
    the passed model before restoring its original form.
"""
function _ranef(m::LinearMixedModel, θref; uscale::Bool=false)
    vv = try
        ranef(updateL!(setθ!(m, θref)); uscale)
    catch e
        @error "Failed to compute unshrunken values with the following exception:"
        rethrow(e)
    finally
        updateL!(setθ!(m, m.optsum.final)) # restore parameter estimates and update m
    end
    return vv
end

function _ranef(m::GeneralizedLinearMixedModel, θref; uscale::Bool=false)
    fast = length(m.θ) == length(m.optsum.final)
    setpar! = fast ? MixedModels.setθ! : MixedModels.setβθ!
    vv = try
        ranef(pirls!(setpar!(m, θref), fast, false); uscale) # not verbose
    catch e
        @error "Failed to compute unshrunken values with the following exception:"
        rethrow(e)
    finally
        pirls!(setpar!(m, m.optsum.final), fast, false) # restore parameter estimates and update m
    end

    return vv
end

"""
    shrinkagetables(m::MixedModel{T},
                    θref::AbstractVector{T}=(isa(m, LinearMixedModel) ? 1e4 : 1) .*
                                            m.optsum.initial;
                    uscale=false) where {T}

Returns a NamedTuple of Tables.jl-tables of the change from OLS estimates
to BLUPs from the mixed model.

Each entry in the named tuple corresponds to a single grouping term.

!!! warning
    This function is **not** thread safe because it temporarily mutates
    the passed model before restoring its original form.
"""
function shrinkagetables(m::MixedModel{T},
                         θref::AbstractVector{T}=(isa(m, LinearMixedModel) ? 1e4 : 1) .*
                                                 m.optsum.initial;
                         uscale::Bool=false) where {T}

    # BLUPs θref - same at estimated θ
    re = _ranef(m, θref; uscale) .- ranef(m; uscale)
    return NamedTuple{fnames(m)}((map(MixedModels.retbl, re, m.reterms)...,))
end

"""
    shrinkagenorm(m::MixedModel{T},
                  θref::AbstractVector{T}=(isa(m, LinearMixedModel) ? 1e4 : 1) .*
                                          m.optsum.initial;
                  uscale=false, p=2)

Returns a NamedTuple of Tables.jl-tables norm of the change from OLS estimates (across all relevant coefficients)
to BLUPs from the mixed model.

`p` corresponds to the ``L_p`` norms, i.e. ``p=2`` is the Euclidean metric.

Each entry in the named tuple corresponds to a single grouping term.

!!! warning
    This function is **not** thread safe because it temporarily mutates
    the passed model before restoring its original form.
"""
function shrinkagenorm(m::MixedModel{T},
                       θref::AbstractVector{T}=(isa(m, LinearMixedModel) ? 1e4 : 1) .*
                                               m.optsum.initial;
                       uscale::Bool=false, p::Real=2) where {T}
    reest = ranef(m; uscale)
    reref = _ranef(m, θref; uscale)

    sh = map(zip(reref, reest, m.reterms)) do (ref, est, trm)
        shrinkage = norm.((view(ref, :, j) .- view(est, :, j) for j in axes(est, 2)), p)
        return merge(NamedTuple{(MixedModels.fname(trm),)}((trm.levels,)),
                     (; shrinkage))
    end
    return NamedTuple{fnames(m)}(sh)
end
