module NeuralEstimators

using AlgebraOfGraphics
using Base: @propagate_inbounds, @kwdef
using Base.GC: gc
import Base: merge, show, size
using BSON: @save, load
using CairoMakie
using ChainRulesCore: @non_differentiable, @ignore_derivatives
using ColorSchemes
using CUDA
using CSV
using DataFrames
using Distances
using Distributions
using Distributions: Bernoulli, Product
using Folds
using Flux
using Flux: ofeltype, params, DataLoader, update!, glorot_uniform
using Functors: @functor
using GraphNeuralNetworks
using GraphNeuralNetworks: check_num_nodes
using GaussianRandomFields
using Graphs
using InvertedIndices
using LinearAlgebra
using NamedArrays
using NearestNeighbors
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

export DeepSet, Compress, CovarianceMatrix, CorrelationMatrix
export vectotril, vectotriu
include("Architectures.jl")

export NeuralEstimator, PointEstimator, IntervalEstimator, PiecewiseEstimator, initialise_estimator
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

export assess, Assessment, merge, risk, bias, rmse, coverage, plot
include("assess.jl")

export bootstrap, interval
include("bootstrap.jl")

export stackarrays, expandgrid, loadbestweights, loadweights, numberreplicates, nparams, samplesize, drop, containertype, estimateinbatches, rowwisenorm
include("utility.jl")

export samplesize, samplecorrelation, samplecovariance
include("summarystatistics.jl")

export EM, removedata, encodedata
include("missingdata.jl")

end

#TODO
# - Clean up my handling of GNN: do we really need a separate object for it, or can we just use DeepSet with the inner network a GNN?
# - Examples: in the univariate example, might be better to show an inversegamma prior for σ: a broader support for σ will hide some of the issues
# - Examples: show a plot of a single data set within each example. Can show a histogram for univariate data; a scatterplot for bivariate data; a heatmap for gridded data; and scatterplot for irregular spatial data.
# - Add functionality for storing and plotting the training-validation risk in the NeuralEstimator. This will involve changing _train() to return both the estimator and the risk, and then defining train(::NeuralEstimator) to update the slot containing the risk. We will also need _train() to take the argument "loss_vs_epoch", so that we can "continue training". Oncce I do this, I can then add a plotting method for plotting the risk.
# - Might also be useful to store the parameter_names in NeuralEstimator: if they are present in the estimator, they can be compared to other sources of parameter_names as a sanity check, and they can be used in bootstrap() so that the bootstrap estimates and resulting intervals are given informative names.
# - General purpose quantile estimator of the form (9) in the manuscript. Also look into monotonic networks.
# - Add helper functions for censored data and write an example in the documentation.
# - Check that training with CorrelationMatrix works well.
# - See if I can move WeightedGraphConv to GraphNeuralNetworks (bit untidy that it's in this package and not in the GNN package).


# ---- long term:
# - turn some document examples into "doctests"
# - Add "AR(k) time series" example, or a Ricker model. (An example using partially exchangeable neural networks.)
# - Precompile NeuralEstimators.jl to reduce latency: See https://julialang.org/blog/2021/01/precompile_tutorial/. It seems very easy, just need to add precompile(f, (arg_types…)) to whatever methods I want to precompile.
# - With the fixed parameters method of train, there seems to be overhead with my current implementation of just-in-time simulation. When epochs_per_Z_refresh = 1, the run-time increases by a factor of 4 for the Gaussian process with m = 1. For now, I’ve added an argument simulate_on_the_fly::Bool, which allows us to switch off just-in-time simulation.
# - Optimise DeepSetExpert on the GPU
# - NeuralRatioEstimator
# - Explicit learning of summary statistics

# ---- once the software is polished:
# - Add NeuralEstimators.jl to the list of packages that use Documenter: see https://documenter.juliadocs.org/stable/man/examples/
# -	Add NeuralEstimators.jl to https://github.com/smsharma/awesome-neural-sbi#code-packages-and-benchmarks.
# -	Once NeuralEstimators is on the Julia package manager, add the following to index.md:
#
# Install `NeuralEstimators` from [Julia](https://julialang.org/)'s package manager using the following command inside Julia:
#
# ```
# using Pkg; Pkg.add("NeuralEstimators")
# ```
