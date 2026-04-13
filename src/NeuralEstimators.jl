module NeuralEstimators

using Accessors: @set
using Adapt
import Adapt: adapt_storage
using ADTypes
using Base: @propagate_inbounds, @kwdef
using Base.GC: gc
import Base: join, merge, show, size, summary, getindex, length, eachindex, hcat
using BSON: @save, load
using ChainRulesCore: @non_differentiable, @ignore_derivatives
using ConcreteStructs: @concrete
using CSV
using DataFrames
using Distances
using Folds
using Functors
using InvertedIndices
using LinearAlgebra
using MLDataDevices: cpu_device, gpu_device, reactant_device, CPUDevice, CUDADevice, ReactantDevice, AbstractDevice
using MLUtils: getobs, joinobs, DataLoader, flatten, zeros_like
import MLUtils: numobs
using NamedArrays
import NamedArrays: NamedMatrix
using NNlib: logσ, softplus, softmax, relu, ⊠, batched_transpose, logsumexp, sigmoid
using Optimisers
using ParameterSchedulers
using Printf: @sprintf
using Random: randexp, shuffle, randperm, AbstractRNG
using SparseArrays
using SpecialFunctions: besselk, gamma, loggamma
using Statistics: mean, median, sum, quantile
using StatsBase
using StatsBase: wsample, sample

function __init__()
    ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"
end

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

export ApproximateDistribution, GaussianMixture, NormalisingFlow, numdistributionalparams
export CouplingLayer, AffineCouplingBlock, ActNorm, Permutation
include(joinpath("ApproximateDistributions", "ApproximateDistributions.jl"))
for file in sort(readdir(joinpath(@__DIR__, "ApproximateDistributions")))
    endswith(file, ".jl") || continue
    file != "ApproximateDistributions.jl" || continue
    include(joinpath("ApproximateDistributions", file))
end

export NeuralEstimator, BayesEstimator
export PosteriorEstimator, RatioEstimator, PointEstimator, IntervalEstimator, QuantileEstimator
export Ensemble, PiecewiseEstimator
export LuxEstimator
export summarynetwork, setsummarynetwork, summarystatistics
include(joinpath("Estimators", "Estimators.jl"))
include(joinpath("Estimators", "Ensemble.jl"))
for file in sort(readdir(joinpath(@__DIR__, "Estimators")))
    endswith(file, ".jl") || continue
    file != "Estimators.jl" || continue
    file != "Ensemble.jl" || continue
    include(joinpath("Estimators", file))
end

export train
export plotrisk, loadrisk, loadoptimiser
include("train.jl")

export FluxTrainState
include("TrainState.jl")

export assess, Assessment, merge, join, risk, bias, rmse, coverage, intervalscore, empiricalprob
include("assess.jl")

export estimate, sampleposterior, logratio, posteriormean, posteriormedian, posteriorquantile, bootstrap, interval, quantiles
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

end
