module NeuralEstimators

using Accessors: @set
using ADTypes
using Base: @propagate_inbounds, @kwdef
using Base.GC: gc
import Base: join, merge, show, size, summary, getindex, length, eachindex
using BSON: @save, load
using CSV
using DataFrames
using Distances
using Flux
using Flux: getobs, ofeltype, DataLoader, update!, onehotbatch, _match_eltype, @non_differentiable, @ignore_derivatives
import Flux: numobs
using Folds
using InvertedIndices
using LinearAlgebra
using NamedArrays
using ParameterSchedulers
using ParameterSchedulers: Stateful, next!
using Printf
using Random: randexp, shuffle, randperm
using SparseArrays
using SpecialFunctions: besselk, gamma, loggamma
using Statistics: mean, median, sum, quantile
using StatsBase
using StatsBase: wsample, sample

export tanhloss, kpowerloss, intervalscore, quantileloss
include("losses.jl")

export AbstractParameterSet
include("Parameters.jl")

export DataSet
include("DataSet.jl")

export DeepSet, Compress, CovarianceMatrix, CorrelationMatrix, ResidualBlock, PowerDifference, DensePositive, MLP
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

export train, trainmultiple
export plotrisk, loadrisk, loadoptimiser
include("train.jl")

export NeuralEstimator
export BayesEstimator, PosteriorEstimator, RatioEstimator
export PointEstimator, IntervalEstimator, QuantileEstimatorContinuous, QuantileEstimator, QuantileEstimatorDiscrete
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

export sampleposterior, posteriormean, posteriormedian, posteriormode, posteriorquantile, bootstrap, interval, estimate, logratio
include("inference.jl")

export stackarrays, expandgrid, numberreplicates, nparams, samplesize, drop, containertype, rowwisenorm, subsetreplicates
include("utility.jl")

export samplesize, logsamplesize, invsqrtsamplesize, samplecorrelation, samplecovariance
include("summarystatistics.jl")

export EM, removedata, encodedata
include("missingdata.jl")

# Functions, function stubs, structs, and exports related to the functionality in the extension ext/NeuralEstimatorsGNNExt.jl
export spatialgraph, GNNSummary, SpatialGraphConv, IndicatorWeights, KernelWeights, PowerDifference, NeighbourhoodVariogram, adjacencymatrix, maternclusterprocess
include("Graphs.jl")

# Simulators and density functions that are useful to have but not needed in generic workflows
export simulategaussian, simulatepotts, simulateschlather
export matern, maternchols, paciorek, scaledlogistic, scaledlogit
export gaussiandensity, schlatherbivariatedensity
include("modelspecificfunctions.jl")

# Backwards compatability and deprecations
export loadbestweights, loadweights, simulate, trainx, mapestimate, initialise_estimator
include("deprecated.jl")

end
