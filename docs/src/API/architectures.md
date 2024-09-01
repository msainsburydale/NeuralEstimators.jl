# Architectures

## Modules

The following high-level modules are often used when constructing a neural-network architecture. In particular, the [`DeepSet`](@ref) is the building block for most classes of [Estimators](@ref) in the package.

```@docs
DeepSet

GNNSummary
```

# User-defined summary statistics

```@index
Order = [:type, :function]
Pages   = ["summarystatistics.md"]
```

The following functions correspond to summary statistics that are often useful
as user-defined summary statistics in [`DeepSet`](@ref) objects.

```@docs
samplesize

samplecorrelation

samplecovariance

NeighbourhoodVariogram
```

## Layers

In addition to the [built-in layers](https://fluxml.ai/Flux.jl/stable/reference/models/layers/) provided by Flux, the following layers may be used when constructing a neural-network architecture.

```@docs
DensePositive

PowerDifference

ResidualBlock

SpatialGraphConv
```


# Output activation functions

```@index
Order = [:type, :function]
Pages   = ["activationfunctions.md"]
```

In addition to the [standard activation functions](https://fluxml.ai/Flux.jl/stable/models/activation/) provided by Flux, the following structs can be used at the end of an architecture to act as output activation functions that ensure valid estimates for certain models. **NB:** Although we refer to the following objects as "activation functions", they should be treated as layers that are included in the final stage of a Flux `Chain()`.

```@docs
Compress

CorrelationMatrix

CovarianceMatrix
```
