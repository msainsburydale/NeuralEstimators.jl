module NeuralEstimators

using Base: @propagate_inbounds, @kwdef
using Base.GC: gc
import Base: join, merge, show, size, summary, getindex, length, eachindex
using BSON: @save, load
using CSV
using DataFrames
using Distances
using Flux
using Flux: getobs, numobs, ofeltype, DataLoader, update!, glorot_uniform, onehotbatch, _match_eltype, @non_differentiable, @ignore_derivatives
using Folds
using Graphs
using GraphNeuralNetworks
using GraphNeuralNetworks: check_num_nodes
import GraphNeuralNetworks: GraphConv
using InvertedIndices
using LinearAlgebra
using NamedArrays
using NearestNeighbors: KDTree, knn
using NNlib: scatter, gather
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
include("loss.jl")

export ParameterConfigurations, subsetparameters
include("Parameters.jl")

export DeepSet, summarystatistics, Compress, CovarianceMatrix, CorrelationMatrix, ResidualBlock, PowerDifference, DensePositive, MLP
export vectotril, vectotriu
include("Architectures.jl")

export ApproximateDistribution, GaussianMixture, NormalisingFlow, logdensity, numdistributionalparams
export AffineCouplingBlock
include("ApproximateDistributions.jl")

export NeuralEstimator 
export BayesEstimator, PosteriorEstimator, RatioEstimator
export PointEstimator, IntervalEstimator, QuantileEstimatorContinuous, QuantileEstimatorDiscrete, QuantileEstimator
export Ensemble, PiecewiseEstimator 
include("Estimators.jl")

export sampleposterior, posteriormean, posteriormedian, posteriormode, posteriorquantile, bootstrap, interval, estimate
include("inference.jl")

export adjacencymatrix, spatialgraph, maternclusterprocess, SpatialGraphConv, GNNSummary, IndicatorWeights, KernelWeights, PowerDifference
include("Graphs.jl")

export simulategaussian, simulatepotts, simulateschlather
export matern, maternchols, paciorek, scaledlogistic, scaledlogit
include("simulate.jl")

export gaussiandensity, schlatherbivariatedensity
include("densities.jl")

export train, trainmultiple, subsetdata
include("train.jl")

export assess, Assessment, merge, join, risk, bias, rmse, coverage, intervalscore, empiricalprob
include("assess.jl")

export stackarrays, expandgrid, numberreplicates, nparams, samplesize, drop, containertype, rowwisenorm
include("utility.jl")

export samplesize, samplecorrelation, samplecovariance, NeighbourhoodVariogram
include("summarystatistics.jl")

export EM, removedata, encodedata
include("missingdata.jl")

# Backwards compatability and deprecations:
export loadbestweights, loadweights, simulate, trainx, mapestimate, initialise_estimator 
include("deprecated.jl")

end