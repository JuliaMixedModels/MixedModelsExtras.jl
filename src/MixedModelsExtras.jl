module MixedModelsExtras

using Distributions
using LinearAlgebra
using MixedModels
using Random
using SpecialFunctions
using Statistics
using StatsBase
using StatsModels
using Tables

using MixedModels: replicate, fname
using GLM: linkinv, Link, LogitLink, ProbitLink, CloglogLink, LogLink
using StatsModels: termnames, vif, gvif
export termnames, gvif, vif

StatsModels.termnames(::RandomEffectsTerm) = String[]

include("interrater.jl")
export InterraterICC, InterraterICCTable

include("icc.jl")
export icc, confint, IccBootstrap

include("r2.jl")
export r², r2, adjr², adjr2

include("remef.jl")
export partial_fitted

include("shrinkage.jl")
export shrinkagenorm, shrinkagetables

include("bootstrap.jl")
export bootstrap_lrt

include("tables.jl")
export ictable

end
