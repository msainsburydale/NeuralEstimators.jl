module NeuralEstimators

using Base: @propagate_inbounds, @kwdef
using Base.GC: gc
import Base: join, merge, show, size, summary
using BSON: @save, load
using ChainRulesCore: @non_differentiable, @ignore_derivatives
using CSV
using DataFrames
using Distances 
using Distributions: Poisson, Bernoulli, product_distribution
using Flux
using Flux: ofeltype, params, DataLoader, update!, glorot_uniform, onehotbatch, _size_check, _match_eltype # @layer
using Flux: @functor; var"@layer" = var"@functor" # NB did this because even semi-recent versions of Flux do not include @layer
using Folds
using Graphs 
using GraphNeuralNetworks
using GraphNeuralNetworks: check_num_nodes, scatter, gather
import GraphNeuralNetworks: GraphConv 
using InvertedIndices
using LinearAlgebra
using NamedArrays 
using NearestNeighbors: KDTree, knn
using Optim # needed to obtain the MAP with neural ratio
using Random: randexp, shuffle
using RecursiveArrayTools: VectorOfArray, convert
using SparseArrays
using SpecialFunctions: besselk, gamma, loggamma
using Statistics: mean, median, sum, quantile
using StatsBase
using StatsBase: wsample
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

export adjacencymatrix, spatialgraph, maternclusterprocess, SpatialGraphConv, GNNSummary, IndicatorWeights, PowerDifference
include("Graphs.jl")

export simulate, simulategaussian, simulatepotts, simulateschlather
export matern, maternchols, scaledlogistic, scaledlogit
include("simulate.jl")

export gaussiandensity, schlatherbivariatedensity
include("densities.jl")

export train, trainx, subsetdata
include("train.jl")

export assess, Assessment, merge, join, risk, bias, rmse, coverage, intervalscore, empiricalprob
include("assess.jl")

export stackarrays, expandgrid, loadbestweights, loadweights, numberreplicates, nparams, samplesize, drop, containertype, estimateinbatches, rowwisenorm
include("utility.jl")

export samplesize, samplecorrelation, samplecovariance, NeighbourhoodVariogram
include("summarystatistics.jl")

export EM, removedata, encodedata
include("missingdata.jl")

# Backwards compatability:
simulategaussianprocess = simulategaussian; export simulategaussianprocess

end

#---- TODO
# - Documentation: sometimes use 'd' to denote the dimension of the response variable, and sometimes 'q'... try to be consistent
# - assess(est::RatioEstimator) using simulation-based calibration (e.g., qq plots)
# - Advanced usage: missing-data methodology in docs/src/devel, and also re-add documentation for struct "EM"
# - Examples: Bivariate data in multivariate section
# - Helper functions for censored data, and provide an example in the documentation (maybe tied in with the bivariate data example).
 
# ---- once the software is reasonably polished:
# - Add NeuralEstimators.jl to the list of packages that use Documenter: see https://documenter.juliadocs.org/stable/man/examples/
# -	Add NeuralEstimators.jl to https://github.com/smsharma/awesome-neural-sbi#code-packages-and-benchmarks.
# -	Once NeuralEstimators is on the Julia package manager, add the following to index.md:
#
# Install `NeuralEstimators` from [Julia](https://julialang.org/)'s package manager using the following command inside Julia:
#
# ```
# using Pkg; Pkg.add("NeuralEstimators")
# ```

# ---- long term:
# -	Add option to check validation risk (and save the optimal estimator) more frequently than the end of each epoch.
# - Should have initialise_estimator() as an internal function, and instead have the public API be based on constructors of the various estimator classes. This aligns more with the basic ideas of Julia, where functions returning a certain class should be made as a constructor rather than a separate function. 
# - Examples: discrete parameters (e.g., Chan et al., 2018). Might need extra functionality for this. 
# - Sequence (e.g., time-series) input: https://jldc.ch/post/seq2one-flux/
# - Precompile NeuralEstimators.jl to reduce latency: See https://julialang.org/blog/2021/01/precompile_tutorial/. Seems easy, just need to add precompile(f, (arg_types…)) to whichever methods we want to precompile
# - Examples: data plots within each example. Can show a histogram for univariate data; a scatterplot for bivariate data; a heatmap for gridded data; and scatterplot for irregular spatial data.
# - Extension: Incorporate the following package to greatly expand bootstrap functionality: https://github.com/juliangehring/Bootstrap.jl. Note also the "straps()" method that allows one to obtain the bootstrap distribution. I think what I can do is define a method of interval(bs::BootstrapSample). Maybe one difficulty will be how to re-sample... Not sure how the bootstrap method will know to sample from the independent replicates dimension (the last dimension) of each array.
# - GPU on MacOS with Metal.jl (already have extension written, need to wait until Metal.jl is further developed; in particular, need convolution layers to be implemented)
# - Explicit learning of summary statistics
# - Amortised posterior approximation (https://github.com/slimgroup/InvertibleNetworks.jl)
# - Amortised likelihood approximation (https://github.com/slimgroup/InvertibleNetworks.jl)
# - Functionality for storing and plotting the training-validation risk in the NeuralEstimator. This will involve changing _train() to return both the estimator and the risk, and then defining train(::NeuralEstimator) to update the slot containing the risk. We will also need _train() to take the argument "loss_vs_epoch", so that we can "continue training"
# - Separate GNN functionality (tried this with package extensions but not possible currently because we need to define custom structs)
# - SpatialPyramidPool for CNNs 
# - Optionally store parameter_names in NeuralEstimator: they can be used in bootstrap() so that the bootstrap estimates and resulting intervals are given informative names
# - Turn some document examples into "doctests"
# - Add "AR(k) time series" example, or a Ricker model (an example using partially exchangeable neural networks)
# - GNN: recall that I set the code up to have ndata as a 3D array; with this format, non-parametric bootstrap would be exceedingly fast (since we can just subset the array data). Non parametric bootstrap is super slow because subsetdata() is super slow with graphical data... would be good to fix this so that non-parametric bootstrap is more efficient, and also so that train() is more efficient (and so that we don’t need to add qualifiers to the subsetting methods). Note that this may also be resolved by improvements made to GraphNeuralNetworks.jl
# - Automatic checking of examples. 


