using Aqua
using DataFrames
using Distributions
using GLM
using LinearAlgebra
using MixedModels
using MixedModelsExtras
using SpecialFunctions
using Suppressor
using StableRNGs
using Statistics
using StatsBase
using Tables
using Test

using GLM: linkinv, Link
using MixedModels: likelihoodratiotest
using MixedModelsDatasets: dataset
using MixedModelsExtras: _ranef
using RDatasets: dataset as rdataset

const progress = Base.isinteractive()

macro suppress_in_ci(ex)
    if Base.isinteractive()
        return esc(ex)
    else
        return :(@suppress $(esc(ex)))
    end
end
