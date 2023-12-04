# Architectures and activations functions

## Index

```@index
Order = [:type, :function]
Pages   = ["architectures.md"]
```

## Architectures

Although the user is free to construct their neural estimator however they see fit (i.e., using arbitrary `Flux` code), `NeuralEstimators` provides several useful architectures described below that are specifically relevant to neural estimation. See also the convenience constructor [`initialise_estimator`](@ref).  

```@docs
DeepSet

DeepSetExpert

GNN
```

## Layers

```@docs
WeightedGraphConv

UniversalPool
```

## Output activation functions

These layers can be used at the end of an architecture to ensure that the
neural estimator provides valid parameters.

```@docs
Compress

CholeskyCovariance

CovarianceMatrix

CorrelationMatrix

SplitApply
```
