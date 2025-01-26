# Architectures

In principle, any [`Flux`](https://fluxml.ai/Flux.jl/stable/) model can be used to construct the neural network. To integrate it into the workflow, one need only define a method that transforms $K$-dimensional vectors of data sets into matrices with $K$ columns, where the number of rows corresponds to the dimensionality of the output spaces listed in the [Overview](@ref). The type [`DeepSet`](@ref) serves as a convenient wrapper for embedding standard neural networks (e.g., MLPs, CNNs, GNNs) in a framework for making inference with an arbitrary number of independent replicates, and it comes with pre-defined methods for handling the transformations from a $K$-dimensional vector of data to a matrix output. 

## Modules

The following high-level modules are often used when constructing a neural-network architecture. In particular, the [`DeepSet`](@ref) is a frequently-used building block for most classes of [Estimators](@ref) in the package.

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

In addition to the [standard activation functions](https://fluxml.ai/Flux.jl/stable/models/activation/) provided by Flux, the following layers can be used at the end of an architecture to act as "output activation functions" that ensure valid estimates for certain models. Note that, although we may conceptualise the following structs as "output activation functions", they should be treated as separate layers included in the final stage of a Flux `Chain()`. In particular, they cannot be used as the activation function of a `Dense` layer. 

```@docs
Compress

CorrelationMatrix

CovarianceMatrix
```
