```@meta
CurrentModule = MixedModelsExtras
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


## Intra-Class Correlation Coefficient

```@docs
icc
```

## Variance Inflation Factor

```@docs
vif(::MixedModel)
```

```@docs
termnames(::MixedModel)
```

```@docs
gvif(::MixedModel)
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
