```@meta
CollapsedDocStrings = true
```

# Estimators

The package provides several classes of neural estimator, organised within a type hierarchy rooted at the abstract supertype [`NeuralEstimator`](@ref).

```@docs
NeuralEstimator
```

## Posterior estimators

[`PosteriorEstimator`](@ref) approximates the posterior distribution, using a flexible, parametric family of distributions (see [Approximate distributions](@ref)).

```@docs
PosteriorEstimator
```

## Ratio estimators

[`RatioEstimator`](@ref) approximates the likelihood-to-evidence ratio, enabling both frequentist and Bayesian inference through various downstream algorithms.

```@docs
RatioEstimator
```

## Bayes estimators

Neural Bayes estimators are implemented as subtypes of [`BayesEstimator`](@ref). The general-purpose [`PointEstimator`](@ref) supports user-defined loss functions (see [Loss functions](@ref)). The types [`IntervalEstimator`](@ref) and its generalisation [`QuantileEstimator`](@ref) are designed for posterior quantile estimation based on user-specified probability levels, automatically configuring the quantile loss and enforcing non-crossing constraints.

```@docs
BayesEstimator

PointEstimator

IntervalEstimator

QuantileEstimator
```

## Ensembles

[`Ensemble`](@ref) combines multiple estimators, aggregating their individual estimates to improve accuracy.

```@docs
Ensemble
```

## Helper functions

The following helper functions operate on an estimator to inspect its components or apply parts of it to data. For the main inference functions used post-training, see [Inference with observed data](@ref).

```@docs
summarynetwork

setsummarynetwork

summarystatistics
```

## Lux.jl convenience wrapper

Both [Flux.jl](https://fluxml.ai/Flux.jl/stable/) and [Lux.jl](https://lux.csail.mit.edu/stable/) are supported. These frameworks differ in a key way: Flux networks store their trainable parameters and states inside the network object, while Lux networks store them externally as explicit, separate objects. 

For convenience, [`LuxEstimator`](@ref) bundles a Lux-based estimator together with its parameters and states for a unified, backend-agnostic API.

```@docs
LuxEstimator
```