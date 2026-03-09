# Core

This page documents the classes and functions that are central to the workflow of `NeuralEstimators`. Its organisation reflects the order in which these classes and functions appear in a standard implementation: from sampling parameters from the prior distribution, to making inference with observed data.

## Sampling parameters

Parameters sampled from the prior distribution are stored as a $d \times K$ matrix, where $d$ is the dimension of the parameter vector to make inference on and $K$ is the number of sampled parameter vectors. 

It can sometimes be helpful to wrap the parameter matrix in a user-defined type that also stores expensive intermediate objects needed for data simulated (e.g., Cholesky factors). The user-defined type should be a subtype of [`AbstractParameterSet`](@ref), whose only requirement is a field `θ` that stores the matrix of parameters. See [Storing expensive intermediate objects for data simulation](@ref) for further discussion.   

```@docs
AbstractParameterSet
```

## Simulating data

The package accommodates any model for which simulation is feasible by allowing users to define their model implicitly through simulated data. 

Simulated data sets are stored as mini-batches in a format amenable to the chosen neural-network architecture. For example, when constructing an estimator from data collected over a grid, one may use a generic CNN, with each data set stored in the final dimension of a four-dimensional array. When performing inference from replicated data, a [`DeepSet`](@ref) architecture may be used, where simulated data sets are stored in a vector, and conditionally independent replicates are stored as mini-batches within each element of the vector.

Expert summary statistics can be incorporated by wrapping the simulated data in a [`DataSet`](@ref) object, which couples the raw data with a matrix of precomputed summary statistics.

```@docs
DataSet
```

## Estimators

The package provides several classes of neural estimators, organised within a type hierarchy. At the top-level of the hierarchy is [`NeuralEstimator`](@ref), an abstract supertype for all neural estimators in the package. 

Neural Bayes estimators are implemented as subtypes of the abstract supertype [`BayesEstimator`](@ref). The simple type [`PointEstimator`](@ref) is used for constructing neural Bayes estimators under general, user-defined loss functions. Several specialised types cater for the estimation of posterior quantiles based on the quantile loss function: see [`IntervalEstimator`](@ref) and its generalisation [`QuantileEstimator`](@ref) for estimating posterior quantiles for a fixed set of probability levels. 

The type [`PosteriorEstimator`](@ref) can be used to approximate the posterior distribution, and [`RatioEstimator`](@ref) can be used to approximate the likelihood-to-evidence ratio.

Finally, [`Ensemble`](@ref) combines multiple estimators, aggregating their individual estimates to improve accuracy.


```@docs
NeuralEstimator

BayesEstimator

PointEstimator

PosteriorEstimator

RatioEstimator

IntervalEstimator

QuantileEstimator

Ensemble
```

### Helper functions

The following helper functions operate on an estimator to inspect its components or apply parts of it to data. For the main inference functions used post-training, see [Inference with observed data](@ref).

```@docs
summarynetwork

setsummarynetwork

summarystatistics

logdensity
```


## Training

The function [`train()`](@ref) is used to train a neural estimator.

After training, the risk history and optimiser state can be accessed and inspected using [`loadrisk()`](@ref), [`plotrisk()`](@ref), and [`loadoptimiser()`](@ref).

```@docs
train

loadrisk

plotrisk

loadoptimiser
```


## Assessment/calibration

```@docs
assess

Assessment

plot(assessment::Assessment)

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

logratio
```
