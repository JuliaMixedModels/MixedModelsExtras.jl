using Documenter
using MixedModelsExtras

makedocs(;
         repo=Remotes.GitHub("JuliaMixedModels", "MixedModelsExtras.jl"),
         sitename="MixedModelsExtras",
         doctest=true,
         checkdocs=:exports,
         warnonly=[:cross_references],
         pages=["index.md", "api.md"])

deploydocs(; repo="github.com/JuliaMixedModels/MixedModelsExtras.jl.git", push_preview=true)
