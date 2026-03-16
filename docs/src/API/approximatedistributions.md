```@meta
CollapsedDocStrings = true
```

# Approximate distributions

When constructing a [`PosteriorEstimator`](@ref), one must choose a family of distributions $q(\boldsymbol{\theta}; \boldsymbol{\kappa})$, parameterized by $\boldsymbol{\kappa} \in \mathcal{K}$, used to approximate the posterior distribution. These families of distributions are implemented as subtypes of the abstract supertype [ApproximateDistribution](@ref).

## Distributions 

```@docs
ApproximateDistribution

GaussianMixture

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
