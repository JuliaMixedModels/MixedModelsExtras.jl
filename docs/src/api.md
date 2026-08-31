```@meta
CurrentModule = MixedModelsExtras
CollapsedDocStrings = true
DocTestSetup = quote
    using MixedModelsExtras
end
DocTestFilters = [r"([a-z]*) => \1", r"getfield\(.*##[0-9]+#[0-9]+"]
```

# MixedModelsExtras.jl API

## Coefficient of Determination

```@docs
r2(::LinearMixedModel)
```

```@docs
adjr2(::LinearMixedModel)
```


## Intraclass Correlation Coefficient

See [Intraclass Correlation Coefficients](@ref) for the taxonomy of coefficients, guidance
on choosing between them, and references.

```@docs
icc
```

```@docs
InterraterICC
```

```@docs
InterraterICCTable
```

```@docs
IccBootstrap
```

## Bootstrapped Likelihood Ratio Test

```@docs
bootstrap_lrt
```

## Variance Inflation Factor

```@docs
vif
```

```@docs
termnames(::MixedModel)
```

```@docs
gvif
```

## "Partial" Effects

```@docs
partial_fitted
```

## Shrinkage Metrics

```@docs
shrinkagenorm
```

```@docs
shrinkagetables
```

## Tables

```@docs
ictable
```
