module NeuralEstimators

using Base: @propagate_inbounds, @kwdef
using Base.GC: gc
import Base: merge
import Base: size
using BSON: @save, load
using ChainRulesCore: @non_differentiable, @ignore_derivatives
using CUDA
using CSV
using DataFrames
using Distances
using Distributions
using Distributions: Bernoulli, Product
import Distributions: cdf, logpdf, quantile, minimum, maximum, insupport, var, skewness
using Flux
using Flux: ofeltype, params, DataLoader, update!, glorot_uniform
using Functors: @functor
using GraphNeuralNetworks
using GraphNeuralNetworks: check_num_nodes
using GaussianRandomFields
using Graphs
using LinearAlgebra
using NamedArrays
using Random: randexp, shuffle
using RecursiveArrayTools: VectorOfArray, convert
using SparseArrays
using SpecialFunctions: besselk, gamma, loggamma
using Statistics: mean, median, sum
using StatsBase
using Zygote

export kpowerloss, intervalscore, quantileloss
include("loss.jl")

export ParameterConfigurations, subsetparameters
include("Parameters.jl")

export DeepSet, DeepSetExpert, Compress, SplitApply
export CholeskyCovariance, CovarianceMatrix, CorrelationMatrix
export vectotril, vectotriu
include("Architectures.jl")

export NeuralEstimator, PointEstimator, IntervalEstimator, IntervalEstimatorCompactPrior, PointIntervalEstimator, QuantileEstimator, PiecewiseEstimator, initialise_estimator
include("Estimators.jl")

export GNN, UniversalPool, adjacencymatrix, WeightedGraphConv, maternclusterprocess
include("Graphs.jl")

export simulate, simulategaussianprocess, simulateschlather, simulateconditionalextremes
export matern, maternchols, scaledlogistic, scaledlogit
include("simulate.jl")

export gaussiandensity, schlatherbivariatedensity
include("densities.jl")

export train, trainx, subsetdata
include("train.jl")

export assess, Assessment, merge, risk
include("assess.jl")

export plotrisk, plotdistribution
include("plotting.jl")

export bootstrap, interval
include("bootstrap.jl")

export stackarrays, expandgrid, loadbestweights, loadweights, numberreplicates, nparams, samplesize, drop, containertype, estimateinbatches
include("utility.jl")

export NeuralEM, removedata, encodedata # TODO unit testing for NeuralEM
include("missingdata.jl")

end

#TODO
# - Add helper functions for censored data and write an example in the documentation.
# -	Plotting from Julia (which can act directly on the object of type assessment).
# -	Examples:
#   o	Add some figures to the examples in the documentation (e.g., show the sampling distribution in univariate example).
#   o	Give the formula for how to compute the input channels dimension in the gridded example.


# ---- long term:
# - turn some document examples into "doctests"
# - plotrisk and plotdistribution (wait until the R interface is finished)
# - Add "AR(k) time series" example. (An example using partially exchangeable neural networks.)
# - Precompile NeuralEstimators.jl to reduce latency: See https://julialang.org/blog/2021/01/precompile_tutorial/. It seems very easy, just need to add precompile(f, (arg_types…)) to whatever methods I want to precompile.
# - Get DeepSetExpert working optimally on the GPU (leaving this for now as we don't need it for the paper).
# - With the fixed parameters method of train, there seems to be overhead with my current implementation of just-in-time simulation. When epochs_per_Z_refresh = 1, the run-time increases by a factor of 4 for the Gaussian process with m = 1. For now, I’ve added an argument simulate_on_the_fly::Bool, which allows us to switch off just-in-time simulation.

# ---- once the software is properly polished:
# - Add NeuralEstimators.jl to the list of packages that use Documenter: see https://documenter.juliadocs.org/stable/man/examples/
# -	Add NeuralEstimators.jl to https://github.com/smsharma/awesome-neural-sbi#code-packages-and-benchmarks.
# -	Once NeuralEstimators is on the Julia package manager, add the following to index.md:
#
# Install `NeuralEstimators` from [Julia](https://julialang.org/)'s package manager using the following command inside Julia:
#
# ```
# using Pkg; Pkg.add("NeuralEstimators")
# ```
