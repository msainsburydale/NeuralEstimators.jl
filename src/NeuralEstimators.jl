module NeuralEstimators

using Base: @propagate_inbounds, @kwdef
using Base.GC: gc
import Base: join, merge, show, size, summary, getindex, length, eachindex
using BSON: @save, load
using CSV
using DataFrames
using Distances
using Flux
using Flux: ofeltype, DataLoader, update!, glorot_uniform, onehotbatch, _match_eltype, @non_differentiable, @ignore_derivatives
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
using Random: randexp, shuffle, randperm
using RecursiveArrayTools: VectorOfArray, convert
using SparseArrays
using SpecialFunctions: besselk, gamma, loggamma
using Statistics: mean, median, sum, quantile
using StatsBase 
using StatsBase: wsample, sample

export tanhloss, kpowerloss, intervalscore, quantileloss
include("loss.jl")

export ParameterConfigurations, subsetparameters
include("Parameters.jl")

export DeepSet, summarystatistics, Compress, CovarianceMatrix, CorrelationMatrix, ResidualBlock, PowerDifference, DensePositive 
export vectotril, vectotriu
include("Architectures.jl")

export ApproximateDistribution, GaussianDistribution, NormalisingFlow, logdensity, numdistributionalparams
export MLP, AffineCouplingBlock
include("ApproximateDistributions.jl")

export NeuralEstimator 
export BayesEstimator, PosteriorEstimator, RatioEstimator
export PointEstimator, IntervalEstimator, QuantileEstimatorContinuous, QuantileEstimatorDiscrete
export Ensemble, PiecewiseEstimator 
include("Estimators.jl")

export sampleposterior, posteriormean, posteriormedian, posteriormode, mlestimate, bootstrap, interval, estimate
include("inference.jl")

export adjacencymatrix, spatialgraph, maternclusterprocess, SpatialGraphConv, GNNSummary, IndicatorWeights, KernelWeights, PowerDifference
include("Graphs.jl")

export simulategaussian, simulatepotts, simulateschlather
export matern, maternchols, paciorek, scaledlogistic, scaledlogit
include("simulate.jl")

export gaussiandensity, schlatherbivariatedensity
include("densities.jl")

export train, subsetdata
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

# ---- longer term/lower priority:
# - Functionality: Make Ensemble “play well” throughout the package. 
# - Functionality: assess(est::PosteriorEstimator) and assess(est::RatioEstimator) using simulation-based calibration (e.g., qq plots). Also a CRPS diagnostic for PosteriorEstimators. 
# - Functionality: Incorporate the following package (possibly as an extension) to greatly expand bootstrap functionality; https://github.com/juliangehring/Bootstrap.jl. Note also the "straps()" method that allows one to obtain the bootstrap distribution. I think what I can do is define a method of interval(bs::BootstrapSample). Maybe one difficulty will be how to re-sample... Not sure how the bootstrap method will know to sample from the independent replicates dimension (the last dimension) of each array.
# -	Functionality: Training, option to check validation risk (and save the optimal estimator) more frequently than the end of each epoch, which would avoid wasted computation when we have very large training sets. 
# - Functionality: Sequence (e.g., time-series) input https://jldc.ch/post/seq2one-flux/, and see also the new recurrent layers added to Flux. 
# - Functionality: Helper functions for censored data. 
# - Functionality: Explicit learning of summary statistics.
# - Polishing: Might be better to use Plots rather than {AlgebraOfGraphics, CairoMakie}.
# - Add NeuralEstimators.jl to the list of packages that use Documenter: see https://documenter.juliadocs.org/stable/man/examples/
# -	Add NeuralEstimators.jl to https://github.com/smsharma/awesome-neural-sbi#code-packages-and-benchmarks
# - Examples: Bivariate data in multivariate section.
# - Examples: Discrete parameters (e.g., Chan et al., 2018). Might need extra functionality for this.
# - Documentation: Turn some examples into "doctests" for automatic checking of examples.
# - Precompile to reduce latency: See https://julialang.org/blog/2021/01/precompile_tutorial/. 
# - Functionality: Add "AR(k) time series" example, or a Ricker model (an example using partially exchangeable neural networks)
# - Functionality: GNN: I set the code up to have ndata as a 3D array; with this format, non-parametric bootstrap would be fast (since we can just subset the array data). Non-parametric bootstrap is currently super slow because subsetdata() is super slow with graphical data... would be good to fix this so that non-parametric bootstrap is more efficient, and also so that train() is more efficient (and so that we don’t need to add qualifiers to the subsetting methods). Note that this may also be resolved by improvements to GraphNeuralNetworks.jl.