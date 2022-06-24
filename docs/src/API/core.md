# Core functions

## Deep Set representation

```@docs
DeepSet

DeepSet(ψ, ϕ; aggregation::String)
```

## Simulation

```@docs
simulate
```

## Training

There are two training methods. For both methods, the validation parameters and validation data are
held fixed so that the validation risk is interpretable. There are a number of practical considerations to keep in mind: In particular, see [Balancing time and memory complexity](@ref).

```@docs
train
```

## Estimation

```@docs
estimate
```

## Bootstrapping

Note that all bootstrapping functions are currently implemented for a single parameter configuration only.

```@docs
parametricbootstrap

nonparametricbootstrap
```
