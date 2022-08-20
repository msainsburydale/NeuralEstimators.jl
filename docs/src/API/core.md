# Core functions

This page documents the functions that are central to the workflow of `NeuralEstimators`. Its organisation reflects the order in which these functions appear in a standard implementation; that is, from storing parameters sampled from the prior distribution, to uncertainty quantification of the final estimates via bootstrapping.


## Storing parameters

```@docs
ParameterConfigurations

subsetparameters
```

## Data simulation

```@docs
simulate
```

## Neural estimator representations

### Deep Set (vanilla)

```@docs
DeepSet

DeepSet(ψ, ϕ; aggregation::String)
```

### Deep Set (with expert summary statistics)

```@docs
DeepSetExpert

DeepSetExpert(deepset::DeepSet, ϕ, S)

DeepSetExpert(ψ, ϕ, S; aggregation::String)

samplesize
```

### Piecewise neural estimators

```@docs
DeepSetPiecewise
```


## Training

There are two training methods. For both methods, the validation parameters and validation data are held fixed so that the validation risk is interpretable. There are a number of practical considerations to keep in mind: In particular, see [Balancing time and memory complexity during training](@ref).

```@docs
train
```

## Assessment

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
