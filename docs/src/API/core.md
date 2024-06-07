# Core

This page documents the classes and functions that are central to the workflow of `NeuralEstimators`. Its organisation reflects the order in which these classes and functions appear in a standard implementation; that is, from sampling parameters from the prior distribution, to using a neural Bayes estimator to make inference with observed data sets.

## Sampling parameters

Parameters sampled from the prior distribution are stored as a $p \times K$ matrix, where $p$ is the number of parameters in the statistical model and $K$ is the number of parameter vectors sampled from the prior distribution.

It can sometimes be helpful to wrap the parameter matrix in a user-defined type that also stores expensive intermediate objects needed for data simulated (e.g., Cholesky factors). In this case, the user-defined type should be a subtype of the abstract type [`ParameterConfigurations`](@ref), whose only requirement is a field `θ` that stores the matrix of parameters. See [Storing expensive intermediate objects for data simulation](@ref) for further discussion.   

```@docs
ParameterConfigurations
```

## Simulating data

`NeuralEstimators` facilitates neural estimation for arbitrary statistical models by having the user implicitly define their model via simulated data, either as fixed instances or via a function that simulates data from the statistical model.

The data are always stored as a `Vector{A}`, where each element of the vector corresponds to a data set of $m$ independent replicates associated with one parameter vector (note that $m$ is arbitrary), and where the type `A` depends on the multivariate structure of the data:

- For univariate and unstructured multivariate data, `A` is a $d \times m$ matrix where $d$ is the dimension each replicate (e.g., $d=1$ for univariate data).
- For data collected over a regular grid, `A` is a ($N + 2$)-dimensional array, where $N$ is the dimension of the grid (e.g., $N = 1$ for time series, $N = 2$ for two-dimensional spatial grids, etc.). The first $N$ dimensions of the array correspond to the dimensions of the grid; the penultimate dimension stores the so-called "channels" (this dimension is singleton for univariate processes, two for bivariate processes, and so on); and the final dimension stores the independent replicates. For example, to store 50 independent replicates of a bivariate spatial process measured over a 10x15 grid, one would construct an array of dimension 10x15x2x50.
- For spatial data collected over irregular spatial locations, `A` is a [`GNNGraph`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/api/gnngraph/#GraphNeuralNetworks.GNNGraphs.GNNGraph) with independent replicates (possibly with differing spatial locations) stored as subgraphs using the function [`batch`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/api/gnngraph/#MLUtils.batch-Tuple{AbstractVector{%3C:GNNGraph}}).

## Estimators

Several classes of neural estimators are available in the package.

The simplest class is [`PointEstimator`](@ref), used for constructing arbitrary mappings from the sample space to the parameter space. When constructing a generic point estimator, the user defines the loss function and therefore the Bayes estimator that will be targeted.

Several classes cater for the estimation of marginal posterior quantiles, based on the quantile loss function (see [`quantileloss()`](@ref)); in particular, see [`IntervalEstimator`](@ref) and [`QuantileEstimatorDiscrete`](@ref) for estimating marginal posterior quantiles for a fixed set of probability levels, and [`QuantileEstimatorContinuous`](@ref) for estimating marginal posterior quantiles with the probability level as an input to the neural network.

In addition to point estimation, the package also provides the class [`RatioEstimator`](@ref) for approximating the so-called likelihood-to-evidence ratio. The binary classification problem at the heart of this approach proceeds based on the binary cross-entropy loss.

Users are free to choose the neural-network architecture of these estimators as they see fit (subject to some class-specific requirements), but the package also provides the convenience constructor [`initialise_estimator()`](@ref).

```@docs
NeuralEstimator

PointEstimator

IntervalEstimator

QuantileEstimatorDiscrete

QuantileEstimatorContinuous

RatioEstimator

PiecewiseEstimator
```

## Training

The function [`train`](@ref) is used to train a single neural estimator, while the wrapper function [`trainx`](@ref) is useful for training multiple neural estimators over a range of sample sizes, making using of the technique known as pre-training.

```@docs
train

trainx
```


## Assessment/calibration

```@docs
assess

Assessment

diagnostics

risk

bias

rmse

coverage
```

## Inference with observed data



### Inference using point estimators

Inference with a neural Bayes (point) estimator proceeds simply by applying the estimator `θ̂` to the observed data `Z` (possibly containing multiple data sets) in a call of the form `θ̂(Z)`. To leverage a GPU, simply move the estimator and the data to the GPU using [`gpu()`](https://fluxml.ai/Flux.jl/stable/models/functors/#Flux.gpu-Tuple{Any}); see also [`estimateinbatches()`](@ref) to apply the estimator over batches of data, which can alleviate memory issues when working with a large number of data sets.

Uncertainty quantification often proceeds through the bootstrap distribution, which is essentially available "for free" when bootstrap data sets can be quickly generated; this is facilitated by [`bootstrap()`](@ref) and [`interval()`](@ref). Alternatively, one may approximate a set of low and high marginal posterior quantiles using a specially constructed neural Bayes estimator, which can then be used to construct credible intervals: see [`IntervalEstimator`](@ref), [`QuantileEstimatorDiscrete`](@ref), and [`QuantileEstimatorContinuous`](@ref).  

```@docs
bootstrap

interval
```

### Inference using likelihood and likelihood-to-evidence-ratio estimators

```@docs
mlestimate

mapestimate

sampleposterior
```
