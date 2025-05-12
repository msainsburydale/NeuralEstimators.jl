# Architectures

As discussed in the [Overview](@ref), any [Flux](https://fluxml.ai/Flux.jl/stable/) model can be used to construct a neural network when using the package. In addition to the standard Flux layers and architectures, the following components can be useful.

## Modules

The structures listed below are often useful when constructing neural estimators. In particular, [`DeepSet`](@ref) provides a convenient wrapper for embedding standard neural networks (e.g., MLPs, CNNs, GNNs) into a framework suited to making inference with an arbitrary number of independent replicates. 

```@docs
DeepSet

GNNSummary

MLP
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

logsamplesize

invsqrtsamplesize

samplecorrelation

samplecovariance

NeighbourhoodVariogram
```

## Layers

In addition to the [built-in layers](https://fluxml.ai/Flux.jl/stable/reference/models/layers/) provided by Flux, the following layers may be used when building a neural-network architecture.

```@docs
DensePositive

PowerDifference

ResidualBlock

SpatialGraphConv
```


## Output layers

```@index
Order = [:type, :function]
Pages   = ["activationfunctions.md"]
```

In addition to the [standard activation functions](https://fluxml.ai/Flux.jl/stable/models/activation/) provided by Flux (e.g., `relu`, `softplus`), the following layers can be used at the end of an architecture to ensure valid estimates for certain models. Note that the Flux layer `Parallel` can be useful for applying several different parameter constraints, as shown in the [Univariate data](@ref) example.

!!! note "Layers vs. activation functions"
    Although we may conceptualise the following types as "output activation functions", they should be treated as separate layers included in the final stage of a Flux `Chain()`. In particular, they cannot be used as the activation function of a `Dense` layer. 

```@docs
Compress

CorrelationMatrix

CovarianceMatrix
```
