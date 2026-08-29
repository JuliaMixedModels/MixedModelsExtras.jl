using Documenter
using MixedModelsExtras
using StatsAPI
using StatsModels

# `vif` and `gvif` are re-exported from StatsAPI and `termnames` from StatsModels, so
# the API page renders docstrings that live outside this package. Pkg installs those
# packages without a `.git` directory, so Documenter cannot work out where their source
# lives and fails with a `MissingRemoteError`. Naming the remote together with a commit
# lets it build the source links without consulting git.
function upstream(mod, org, name)
    return pkgdir(mod) => (Remotes.GitHub(org, name), "v$(pkgversion(mod))")
end

makedocs(;
         repo=Remotes.GitHub("JuliaMixedModels", "MixedModelsExtras.jl"),
         remotes=Dict(upstream(StatsAPI, "JuliaStats", "StatsAPI.jl"),
                      upstream(StatsModels, "JuliaStats", "StatsModels.jl")),
         # the default branch is named explicitly because Documenter otherwise probes
         # the remote and falls back to "master" when it cannot reach it
         format=Documenter.HTML(; edit_link="main"),
         sitename="MixedModelsExtras",
         doctest=true,
         checkdocs=:exports,
         warnonly=[:cross_references],
         pages=["index.md", "icc.md", "api.md"])

deploydocs(; repo="github.com/JuliaMixedModels/MixedModelsExtras.jl.git",
           devbranch="main", push_preview=true)
