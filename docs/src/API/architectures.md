# Architectures

```@index
Order = [:type, :function]
Pages   = ["architectures.md"]
```

Although the user is free to construct their neural estimator however they see fit (i.e., using arbitrary `Flux` code), `NeuralEstimators` provides several useful architectures described below that are specifically relevant to neural estimation. See also the convenience constructor [`initialise_estimator`](@ref).  

```@docs
DeepSet

GNN
```

## Layers

```@docs
DensePositive

WeightedGraphConv

UniversalPool
```
