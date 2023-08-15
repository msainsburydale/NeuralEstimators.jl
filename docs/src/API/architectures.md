# Architectures and activations functions

## Index

```@index
Order = [:type, :function]
Pages   = ["architectures.md"]
```

## Architectures

Although the user is free to construct their neural estimator however they see fit, `NeuralEstimators` provides several useful architectures described below.

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
