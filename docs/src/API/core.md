# Core

This page documents the functions that are central to the workflow of `NeuralEstimators`. Its organisation reflects the order in which these functions appear in a standard implementation; that is, from sampling parameters from the prior distribution, to uncertainty quantification of the final estimates via bootstrapping.


## Sampling parameters

Parameters sampled from the prior distribution ``\Omega(\cdot)`` are stored as a $p \times K$ matrix, where $p$ is the number of parameters in the model and $K$ is the number of parameter vectors sampled from the prior distribution.

It can sometimes be helpful to wrap the parameter matrix in a user-defined type that also stores expensive intermediate objects needed for data simulated (e.g., Cholesky factors). In this case, the user-defined type should be a subtype of the abstract type [`ParameterConfigurations`](@ref), whose only requirement is a field `θ` that stores the matrix of parameters. See [Storing expensive intermediate objects for data simulation](@ref) for further discussion.   

```@docs
ParameterConfigurations
```

## Simulating data

`NeuralEstimators` facilitates neural estimation for arbitrary statistical models by having the user implicitly define the model via simulated data. The user may provide simulated data directly, or provide a function that simulates data from the model.

The data should be stored as a `Vector{A}`, where each element of the vector is associated with one parameter configuration, and where `A` depends on the architecture of the neural estimator.

## Types of estimators

See also [Architectures and activations functions](@ref) that are often used
when constructing neural estimators, and the convenience constructor [`initialise_estimator`](@ref). 

```@docs
NeuralEstimator

PointEstimator

IntervalEstimator

IntervalEstimatorCompactPrior

PointIntervalEstimator

QuantileEstimator

PiecewiseEstimator
```

## Training

The function [`train`](@ref) is used to train a single neural estimator, while the wrapper function [`trainx`](@ref) is useful for training multiple neural estimators over a range of sample sizes, making using of the technique known as pre-training.

```@docs
train

trainx
```


## Assessing a neural estimator

```@docs
assess

Assessment

risk
```

## Bootstrapping

```@docs
bootstrap

interval
```
