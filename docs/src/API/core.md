# Core

This page documents the classes and functions that are central to the workflow of `NeuralEstimators`. Its organisation reflects the order in which these classes and functions appear in a standard implementation: from sampling parameters from the prior distribution, to making inference with observed data.

## Sampling parameters

Parameters sampled from the prior distribution are stored as a $d \times K$ matrix, where $d$ is the dimension of the parameter vector to make inference on and $K$ is the number of sampled parameter vectors. 

It can sometimes be helpful to wrap the parameter matrix in a user-defined type that also stores expensive intermediate objects needed for data simulated (e.g., Cholesky factors). The user-defined type should be a subtype of [`ParameterConfigurations`](@ref), whose only requirement is a field `θ` that stores the matrix of parameters. See [Storing expensive intermediate objects for data simulation](@ref) for further discussion.   

```@docs
ParameterConfigurations
```

## Simulating data

The package accommodates any model for which simulation is feasible by allowing users to define their model implicitly through simulated data.

The data are stored as a `Vector{A}`, where each element of the vector is associated with one parameter vector, and the subtype `A` depends on the multivariate structure of the data. Common formats include:

* **Unstructured data**: `A` is typically an $n \times m$ matrix, where:
    * ``n`` is the dimension of each replicate (e.g., $n=1$ for univariate data, $n=2$ for bivariate data).  
    * ``m`` is the number of independent replicates in each data set ($m$ is allowed to vary between data sets). 
* __Data collected over a regular grid__: `A` is typically an ($N + 2$)-dimensional array, where: 
    * The first $N$ dimensions correspond to the dimensions of the grid (e.g., $N = 1$ for time series, $N = 2$ for two-dimensional spatial grids). 
    * The penultimate dimension stores the so-called "channels" (e.g., singleton for univariate processes, two for bivariate processes). 
    * The final dimension stores the $m$ independent replicates. 
* **Spatial data collected over irregular locations**: `A` is typically a [`GNNGraph`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/api/gnngraph/#GraphNeuralNetworks.GNNGraphs.GNNGraph), where independent replicates (possibly with differing spatial locations) are stored as subgraphs. See the helper function [`spatialgraph()`](@ref) for constructing these graphs from matrices of spatial locations and data. 

While the formats above cover many applications, the package is flexible: the data structure simply needs to align with the chosen neural-network architecture. 

## Estimators

The package provides several classes of neural estimators, organised within a type hierarchy. At the top-level of the hierarchy is [`NeuralEstimator`](@ref), an abstract supertype for all neural estimators in the package. 

Neural Bayes estimators are implemented as subtypes of the abstract supertype [`BayesEstimator`](@ref). The simple type [`PointEstimator`](@ref) is used for constructing neural Bayes estimators under general, user-defined loss functions. Several specialised types cater for the estimation of posterior quantiles based on the quantile loss function: see [`IntervalEstimator`](@ref) and its generalisation [`QuantileEstimator`](@ref) for estimating posterior quantiles for a fixed set of probability levels. 

The type [`PosteriorEstimator`](@ref) can be used to approximate the posterior distribution, and [`RatioEstimator`](@ref) can be used to approximate the likelihood-to-evidence ratio.

Several types serve as wrappers around the aforementioned estimators, enhancing their functionality. [`PiecewiseEstimator`](@ref) applies different estimators based on the sample size of the data (see the discussion on [Variable sample sizes](@ref)). Finally, [`Ensemble`](@ref) combines multiple estimators, aggregating their individual estimates to improve accuracy.


```@docs
NeuralEstimator

BayesEstimator

PointEstimator

PosteriorEstimator

RatioEstimator

IntervalEstimator

QuantileEstimator

PiecewiseEstimator

Ensemble
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

risk

bias

rmse

coverage
```

## Inference with observed data

The following functions facilitate the use of a trained neural estimator with observed data. 

```@docs
estimate

bootstrap

interval

sampleposterior

posteriormean 

posteriormedian

posteriorquantile

posteriormode
```
