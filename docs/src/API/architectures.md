# Architectures

```@index
Order = [:type, :function]
Pages   = ["architectures.md"]
```

## Modules

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

```@docs
SpatialGraphConv

DensePositive

PowerDifference
```


# Output activation functions

```@index
Order = [:type, :function]
Pages   = ["activationfunctions.md"]
```

In addition to the standard activation functions provided by [Flux](https://fluxml.ai/Flux.jl/stable/models/activation/), the following layers can be used at the end of an architecture, to act as output activation functions that ensure valid estimates for certain models. **NB:** Although we refer to the following objects as "activation functions", they should be treated as layers that are included in the final stage of a Flux `Chain()`. 

```@docs
Compress

CorrelationMatrix

CovarianceMatrix
```

