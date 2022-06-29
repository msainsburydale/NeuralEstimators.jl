# Core functions

## Parameters

```@docs
ParameterConfigurations

subsetparameters
```

## Simulation

```@docs
simulate
```

## Deep Set representation

### Vanilla Deep Set

```@docs
DeepSet

DeepSet(ψ, ϕ; aggregation::String)
```

### Deep Set with expert summary statistics

```@docs
DeepSetExpert

DeepSetExpert(deepset::DeepSet, ϕ, S)

DeepSetExpert(ψ, ϕ, S; aggregation::String)

samplesize
```

### Piecewise Deep Set neural estimators

```@docs
DeepSetPiecewise
```


## Training

There are two training methods. For both methods, the validation parameters and validation data are held fixed so that the validation risk is interpretable. There are a number of practical considerations to keep in mind: In particular, see [Balancing time and memory complexity](@ref).

```@docs
train
```

## Estimation

```@docs
estimate

Estimates

merge(::Estimates)
```

## Bootstrapping

Note that all bootstrapping functions are currently implemented for a single parameter configuration only.

```@docs
parametricbootstrap

nonparametricbootstrap
```
