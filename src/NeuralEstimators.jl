module NeuralEstimators

using Accessors: @set
using Adapt
using ADTypes
using Base: @propagate_inbounds, @kwdef
using Base.GC: gc
import Base: join, merge, show, size, summary, getindex, length, eachindex
using BSON: @save, load
using ChainRulesCore: @non_differentiable, @ignore_derivatives
using CSV
using DataFrames
using Distances
using Folds
using Functors
using InvertedIndices
using LinearAlgebra
using MLDataDevices: reactant_device, cpu_device, reactant_device
using MLUtils: getobs, DataLoader, flatten
import MLUtils: numobs
using NamedArrays
import NamedArrays: NamedMatrix
using NNlib: logσ, softplus, softmax, relu, ⊠, batched_transpose, logsumexp, sigmoid
using Optimisers
using ParameterSchedulers
using Printf: @sprintf
using Random: randexp, shuffle, randperm
using SparseArrays
using SpecialFunctions: besselk, gamma, loggamma
using Statistics: mean, median, sum, quantile
using StatsBase
using StatsBase: wsample, sample

export tanhloss, kpowerloss, intervalscore, quantileloss
include("losses.jl")

export AbstractParameterSet, NamedMatrix
include("Parameters.jl")

export DataSet, Summaries
include("DataSet.jl")

export DeepSet, MLP, Compress, CovarianceMatrix, CorrelationMatrix, ResidualBlock, PowerDifference
export IndicatorWeights, KernelWeights
export vectotril, vectotriu
include("Architectures.jl")

export ApproximateDistribution, GaussianMixture, NormalisingFlow, logdensity, numdistributionalparams
export AffineCouplingBlock
include(joinpath("ApproximateDistributions", "ApproximateDistributions.jl"))
for file in sort(readdir(joinpath(@__DIR__, "ApproximateDistributions")))
    endswith(file, ".jl") || continue
    file != "ApproximateDistributions.jl" || continue
    include(joinpath("ApproximateDistributions", file))
end

export train
export plotrisk, loadrisk, loadoptimiser
include("train.jl")

export NeuralEstimator
export LuxEstimator
export BayesEstimator, PosteriorEstimator, RatioEstimator
export PointEstimator, IntervalEstimator, QuantileEstimator
export Ensemble, PiecewiseEstimator
export summarynetwork, setsummarynetwork, summarystatistics
include(joinpath("Estimators", "Estimators.jl"))
include(joinpath("Estimators", "Ensemble.jl"))
for file in sort(readdir(joinpath(@__DIR__, "Estimators")))
    endswith(file, ".jl") || continue
    file != "Estimators.jl" || continue
    file != "Ensemble.jl" || continue
    include(joinpath("Estimators", file))
end

export assess, Assessment, merge, join, risk, bias, rmse, coverage, intervalscore, empiricalprob
include("assess.jl")

export estimate, sampleposterior, logratio, posteriormean, posteriormedian, posteriormode, posteriorquantile, bootstrap, interval, quantiles
include("inference.jl")

export stackarrays, expandgrid, numberreplicates, samplesize, drop, containertype, rowwisenorm, subsetreplicates
include("utility.jl")

export samplesize, logsamplesize, invsqrtsamplesize, samplecorrelation, samplecovariance
include("summarystatistics.jl")

export EM, removedata, encodedata
include("missingdata.jl")

# Functions, function stubs, structs, and exports related to the functionality in the extension ext/NeuralEstimatorsGNNExt.jl
export spatialgraph, GNNSummary, SpatialGraphConv, PowerDifference, NeighbourhoodVariogram, adjacencymatrix, maternclusterprocess
include("Graphs.jl")

# Simulators and density functions that are useful to have but not needed in generic workflows
export simulategaussian, simulatepotts, simulateschlather
export matern, maternchols, paciorek, scaledlogistic, scaledlogit
export gaussiandensity, schlatherbivariatedensity
include("modelspecificfunctions.jl")

# Backwards compatability and deprecations
export loadbestweights, loadweights, simulate, trainx, mapestimate
include("deprecated.jl")

end
