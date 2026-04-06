```@meta
CollapsedDocStrings = true
```

# Neural-network building blocks

Any [Flux.jl](https://fluxml.ai/Flux.jl/stable/) or [Lux.jl](https://lux.csail.mit.edu/stable/) model can be used to construct a neural network when using the package. In addition to the standard layers and architectures provided with these deep-learning packages ([Flux](https://fluxml.ai/Flux.jl/stable/reference/models/layers/)/[Lux](https://lux.csail.mit.edu/stable/api/Lux/layers)), the following components can be useful.

## Modules

The structures listed below are often useful when constructing neural estimators. In particular, [`DeepSet`](@ref) provides a convenient wrapper for embedding standard neural networks (e.g., MLPs, CNNs, GNNs) into a framework suited to making inference with an arbitrary number of replicates. 

```@docs
DeepSet

GNNSummary

MLP
```

## User-defined summary statistics

```@index
Order = [:type, :function]
Pages   = ["summarystatistics.md"]
```

The following functions correspond to summary statistics that are often useful
as user-defined summaries in [`DeepSet`](@ref) objects.

```@docs
samplesize

logsamplesize

invsqrtsamplesize

samplecorrelation

samplecovariance

NeighbourhoodVariogram
```

## Layers

In addition to the built-in layers provided by [Flux](https://fluxml.ai/Flux.jl/stable/reference/models/layers/) and [Lux](https://lux.csail.mit.edu/stable/api/Lux/layers), the following layers may be used when building a neural-network architecture.

```@docs
ResidualBlock

SpatialGraphConv
```


## Output layers

```@index
Order = [:type, :function]
Pages   = ["activationfunctions.md"]
```

In addition to the standard activation functions provided by [NNlib.jl](https://fluxml.ai/Flux.jl/stable/reference/models/activation/#man-activations) (e.g., `relu`, `gelu`, `softplus`), the following layers can be used at the end of an architecture to ensure valid point estimates for certain models. Note that `Parallel` ([Flux](https://fluxml.ai/Flux.jl/stable/reference/models/layers/#Flux.Parallel)/[Lux](https://lux.csail.mit.edu/stable/api/Lux/layers)) can be useful for applying several parameter constraints.

!!! note "Layers vs. activation functions"
    Although we may conceptualise the following types as "output activation functions", they should be treated as separate layers included in the final stage of a `Chain`. In particular, they cannot be used as the activation function of a `Dense` layer. 

```@docs
Compress

CorrelationMatrix

CovarianceMatrix
```

## Miscellaneous

```@docs
IndicatorWeights

KernelWeights

PowerDifference
```