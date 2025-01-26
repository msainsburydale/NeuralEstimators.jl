# Approximate distributions

When constructing a [`PosteriorEstimator`](@ref), one must choose an approximate distribution $q(\boldsymbol{\theta}; \boldsymbol{\kappa})$ for the posterior distribution $p(\boldsymbol{\theta} \mid \boldsymbol{Z})$, implemented as a subtype of [ApproximateDistribution](@ref). 

## Distributions 

```@docs
ApproximateDistribution

GaussianDistribution

NormalisingFlow
```

## Methods

```@docs
numdistributionalparams
```

## Building blocks

```@docs
AffineCouplingBlock
```
