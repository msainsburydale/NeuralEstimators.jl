module NeuralEstimators

#TODO precompilation 

using Base: @propagate_inbounds, @kwdef
using Base.GC: gc
import Base: join, merge, show, size, summary
using BSON: @save, load
using ChainRulesCore: @non_differentiable, @ignore_derivatives
using Clustering #TODO remove? 
using CUDA #TODO NeuralEstimatorsCUDAExt
using CUDA: CuArray #TODO NeuralEstimatorsCUDAExt
using cuDNN #TODO NeuralEstimatorsCUDAExt
using CSV
using DataFrames
using Distances #TODO remove? 
using Distributions #TODO remove? 
using Folds #TODO remove?
using Flux
using Flux: ofeltype, params, DataLoader, update!, glorot_uniform, onehotbatch, _size_check, _match_eltype # @layer
using Flux: @functor; var"@layer" = var"@functor" # NB did this because even semi-recent versions of Flux do not include @layer
using GraphNeuralNetworks #TODO NeuralEstimatorsGNNExt
using GraphNeuralNetworks: check_num_nodes #TODO NeuralEstimatorsGNNExt
import GraphNeuralNetworks: GraphConv; export GraphConv # method that acts on 3D arrays #TODO NeuralEstimatorsGNNExt
using GaussianRandomFields #TODO remove 
using Graphs #TODO NeuralEstimatorsGNNExt
using InvertedIndices
using LinearAlgebra
using NamedArrays 
using NearestNeighbors #TODO remove? 
using Optim 
using Random: randexp, shuffle
using RecursiveArrayTools: VectorOfArray, convert
using SparseArrays
using SpecialFunctions: besselk, gamma, loggamma
using Statistics: mean, median, sum, quantile
using StatsBase
using StatsBase: wsample
using Suppressor #TODO remove? 
using Zygote 

export tanhloss, kpowerloss, intervalscore, quantileloss
include("loss.jl")

export ParameterConfigurations, subsetparameters
include("Parameters.jl")

export DeepSet, summarystatistics, Compress, CovarianceMatrix, CorrelationMatrix
export vectotril, vectotriu
include("Architectures.jl")

export NeuralEstimator, PointEstimator, IntervalEstimator, QuantileEstimatorContinuous, DensePositive, QuantileEstimatorDiscrete, RatioEstimator, PiecewiseEstimator, initialise_estimator
include("Estimators.jl")

export sampleposterior, mlestimate, mapestimate, bootstrap, interval
include("inference.jl")

export spatialgraph, SpatialGraphConv, SpatialPyramidPool, UniversalPool, GNNSummary, adjacencymatrix, maternclusterprocess, GraphSkipConnection, IndicatorWeights
include("Graphs.jl")

export simulate, simulategaussianprocess, simulateschlather, simulateconditionalextremes
export matern, maternchols, scaledlogistic, scaledlogit
include("simulate.jl")

export gaussiandensity, schlatherbivariatedensity
include("densities.jl")

export train, trainx, subsetdata
include("train.jl")

export assess, Assessment, merge, join, risk, bias, rmse, coverage, intervalscore, diagnostics
include("assess.jl")

export stackarrays, expandgrid, loadbestweights, loadweights, numberreplicates, nparams, samplesize, drop, containertype, estimateinbatches, rowwisenorm
include("utility.jl")

export samplesize, samplecorrelation, samplecovariance, NeighbourhoodVariogram, DistanceQuantiles
include("summarystatistics.jl")

export EM, removedata, encodedata
include("missingdata.jl")

end

#TODO
# - I sometimes use d to denote the dimension of the response variable, and sometimes q... try to be consistent
# - Fix warnings that appear when running the test code
# - assess(est::QuantileEstimatorDiscrete). Here, we can use simulation-based calibration (e.g., qq plots).
# - assess(est::RatioEstimator). Here, we can use simulation-based calibration (e.g., qq plots).
# - Incorporate the following package to very easily add a lot of bootstrap functionality: https://github.com/juliangehring/Bootstrap.jl. Note also the "straps()" method that allows one to obtain the bootstrap distribution. I think what I can do is define a method of interval(bs::BootstrapSample). Maybe one difficulty will be how to re-sample... Not sure how the bootstrap method will know to sample from the independent replicates dimension (the last dimension) of each array.
# - Examples: show a plot of a single data set within each example. Can show a histogram for univariate data; a scatterplot for bivariate data; a heatmap for gridded data; and scatterplot for irregular spatial data.
# - Examples: Bivariate data in multivariate section.

#TODO
# - See if there are any other places I can use reduce(vcat, x) instead of vcat(x…).
# - Examples: discrete parameter.
# - Examples: Add functionality for storing and plotting the training-validation risk in the NeuralEstimator. This will involve changing _train() to return both the estimator and the risk, and then defining train(::NeuralEstimator) to update the slot containing the risk. We will also need _train() to take the argument "loss_vs_epoch", so that we can "continue training". Oncce I do this, I can then add a plotting method for plotting the risk.
# - Add helper functions for censored data and write an example in the documentation.

# ---- long term:
# - SpatialPyramidPool for CNNs (maybe someone already has this code in Julia?)
# - Proper citations: https://juliadocs.org/DocumenterCitations.jl/stable/
# - Might also be useful to store the parameter_names in NeuralEstimator: if they are present in the estimator, they can be compared to other sources of parameter_names as a sanity check, and they can be used in bootstrap() so that the bootstrap estimates and resulting intervals are given informative names.
# - Would be good if interval(θ̂::IntervalEstimator, Z) and interval(bs) also displayed the parameter names... this could be done if the estimator stores the parameter names.
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
