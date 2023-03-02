# Core functions

This page documents the functions that are central to the workflow of `NeuralEstimators`. Its organisation reflects the order in which these functions appear in a standard implementation; that is, from sampling parameters from the prior distribution, to uncertainty quantification of the final estimates via bootstrapping.


## Sampling parameters

Parameters sampled from the prior distribution ``\Omega(\cdot)`` are stored as a $p \times K$ matrix, where $p$ is the number of parameters in the model and $K$ is the number of parameter vectors sampled from the prior distribution.

It can sometimes be helpful to wrap the parameter matrix in a user-defined type that also stores expensive intermediate objects needed for data simulated (e.g., Cholesky factors). In this case, the user-defined type should be a subtype of the abstract type [`ParameterConfigurations`](@ref), whose only requirement is a field `θ` that stores the matrix of parameters. See [Storing expensive intermediate objects for data simulation](@ref) for further discussion.   

```@docs
ParameterConfigurations
```

## Simulating data

`NeuralEstimators` facilitates neural estimation for arbitrary statistical models by having the user implicitly define the model via simulated data. The user may provide simulated data directly, or provide a function that simulates data from the model (by overloading the generic function `simulate`).

The data should be stored as a `Vector{A}`, where each element of the vector is associated with one parameter configuration, and where `A` depends on the representation of the neural estimator. For example, if the neural estimator is a [`DeepSet`](@ref) object, the data should be stored as a `Vector{Array}`, where each array may store independent replicates in its final dimension. Similarly, if the neural estimator is a [`GNNEstimator`](@ref), the data should be stored as a `Vector{GNNGraph}`, where each graph may store independent replicates in sub-graphs.

```@docs
simulate

simulate(parameters, m, J::Integer)
```

## Neural-estimator representations

Although the user is free to construct their neural estimator however they see fit, `NeuralEstimators` provides several useful representations described below.


```@docs
DeepSet

DeepSetExpert

PiecewiseEstimator

GNNEstimator

Compress
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

risk

coverage

plotrisk

plotdistribution
```

## Bootstrapping

```@docs
bootstrap

confidenceinterval
```
