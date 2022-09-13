# Core functions

This page documents the functions that are central to the workflow of `NeuralEstimators`. Its organisation reflects the order in which these functions appear in a standard implementation; that is, from storing parameters sampled from the prior distribution, to uncertainty quantification of the final estimates via bootstrapping.


## Storing parameters

Parameters sampled from the prior distribution ``\Omega(\cdot)`` may be stored i) as a $p \times K$ matrix, where $p$ is the number of parameters in the model and $K$ is the number of parameter vectors sampled from the prior distribution, or ii) in a user-defined subtype of the abstract type [`ParameterConfigurations`](@ref). See [Storing expensive intermediate objects for data simulation](@ref) for further discussion.   

```@docs
ParameterConfigurations

subsetparameters
```

## Simulating data

`NeuralEstimators` facilitates neural estimation for arbitrary statistical models by having the user implicitly define their model either by providing simulated data, or by defining a function for data simulation. If the latter option is chosen, the user must provide a method `simulate(parameters, m)`, which returns simulated data from a set of `parameters`, with `m` the sample size of these simulated data.

Irrespective of their source, the simulated data must be stored as a subtype of `AbstractVector{AbstractArray}`, where each array stores `m` independent replicates simulated from one parameter vector in `parameters`, and these replicates must stored in the final dimension of each array.

```@docs
simulate

simulate(parameters, m::Integer, J::Integer)
```

## Representations for neural estimators

Although the user is free to construct their neural estimator however they see fit, `NeuralEstimators` provides several useful representations described below. Note that if the user wishes to use an alternative representation, for compatibility with `NeuralEstimators`, simply ensure that the estimator processes data stored as subtypes of `AbstractVector{AbstractArray}`, as discussed in [`DeepSet`](@ref) (see also its source code).

### Deep Set

```@docs
DeepSet
```

### Piecewise estimators

```@docs
PiecewiseEstimator
```


## Training

```@docs
train

train(θ̂, P)

train(θ̂, θ_train::P, θ_val::P) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

train(θ̂, θ_train::P, θ_val::P, Z_train::T, Z_val::T) where {T, P <: Union{AbstractMatrix, ParameterConfigurations}}

train(θ̂, θ_train::P, θ_val::P, Z_train::T, Z_val::T, M::Vector{I}) where {T, P <: Union{AbstractMatrix, ParameterConfigurations}, I <: Integer}
```

## Assessing a neural estimator

```@docs
assess

Assessment

merge(::Assessment)
```

## Bootstrapping

Note that all bootstrapping functions are currently implemented for a single parameter configuration only.

```@docs
parametricbootstrap

nonparametricbootstrap
```
