```@meta
CollapsedDocStrings = true
```

# Approximate distributions

When constructing a [`PosteriorEstimator`](@ref), one must specify a parametric family of probability distributions used to approximate the posterior distribution. These families of distributions are implemented as subtypes of the abstract type [ApproximateDistribution](@ref).

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
CouplingLayer

AffineCouplingBlock

ActNorm

Permutation
```
