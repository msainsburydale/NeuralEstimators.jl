module NeuralEstimators

using AlgebraOfGraphics
using Base: @propagate_inbounds, @kwdef
using Base.GC: gc
import Base: join, merge, show, size
using BSON: @save, load
using CairoMakie
using ChainRulesCore: @non_differentiable, @ignore_derivatives
using ColorSchemes
using CUDA
using CUDA: CuArray
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

export assess, Assessment, merge, join, risk, bias, rmse, coverage, plot, intervalscore, diagnostics
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
# - Incorporate the following package to very easily add a lot of bootstrap functionality: https://github.com/juliangehring/Bootstrap.jl. Note also the "straps()" method that allows one to obtain the bootstrap distribution. I think what I can do is define a method of interval(bs::BootstrapSample). Maybe one difficulty will be how to re-sample... Not sure how the bootstrap method will know to sample from the independent replicates dimension (the last dimension) of each array.
# - Examples: show a plot of a single data set within each example. Can show a histogram for univariate data; a scatterplot for bivariate data; a heatmap for gridded data; and scatterplot for irregular spatial data.
# - Examples: Bivariate data in multivariate section.
# - Examples: Gridded spatial data. Parametric bootstrap:

# ```
# θ̂_test = estimateinbatches(θ̂, Z_test)
# B = 200
# Z_boot = [[simulate(θ, m) for b ∈ 1:B] for θ ∈ eachcol(θ̂_test)]
# assessment = assess(θ̂, θ_test, Z_test, boot = Z_boot)
# ```

#TODO
# - Clean up my handling of GNN: do we really need a separate object for it, or can we just use DeepSet with the inner network a GNN?
# - Examples: Add functionality for storing and plotting the training-validation risk in the NeuralEstimator. This will involve changing _train() to return both the estimator and the risk, and then defining train(::NeuralEstimator) to update the slot containing the risk. We will also need _train() to take the argument "loss_vs_epoch", so that we can "continue training". Oncce I do this, I can then add a plotting method for plotting the risk.
# - Examples: discrete parameter.
# - General purpose quantile estimator of the form (9) in the manuscript. Also look into monotonic networks.
# - Add helper functions for censored data and write an example in the documentation.
# - Check that training with CorrelationMatrix works well.

# More/better ways to assess intervals. For example, from Efron (2003):
    # Coverage, even appropriately defined, is not the end of the story. Stability
    # of the intervals, in length and location, is also important. Here is an example.
    # Suppose we are in a standard normal situation where the exact interval is
    # Student’s t with 10 degrees of freedom. Method A produces the exact 90%
    # interval except shortened by a factor of 0.90; method B produces the exact
    # 90% interval either shortened by a factor of 2/3 or lengthened by a factor of
    # 3/2, with equal probability. Both methods provide about 86% coverage, but
    # the intervals in method B will always be substantially misleading.

# ---- long term:
# - Might also be useful to store the parameter_names in NeuralEstimator: if they are present in the estimator, they can be compared to other sources of parameter_names as a sanity check, and they can be used in bootstrap() so that the bootstrap estimates and resulting intervals are given informative names.
# - Would be good if interval(θ̂::IntervalEstimator, Z) and interval(bs) also displayed the parameter names... this could be done if the estimator stores the parameter names.
# - See if I can move WeightedGraphConv to GraphNeuralNetworks (bit untidy that it's in this package and not in the GNN package).
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


#TODO # - L2 Regularisation. Think I have to "modernise" the training function to use the approach that is currently recommended by Flux, rather than the "implicit" approach that they previously recommended.
#TODO add this to "advanced usage (regularisation)" when I get it working
# Another class of regularisation techniques are implemented by modifying the loss function. For instance, L₂ regularisation (sometimes called ridge regression) adds to the loss a penalty based on the square of the neural-network parameters, and is intended to discourage complex models. It can be implemented manually through the loss function (i.e., using [`params`](https://fluxml.ai/Flux.jl/stable/training/reference/#Flux.params) to extract the neural-network parameters and incorporating them in the loss function), or by providing a custom optimiser to [`train`](@ref) that includes a [`WeightDecay`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.WeightDecay). For instance, to perform L₂ regularisation with penalty 0.001, one may define the optimiser as:
#
# ```
# optimiser = Flux.setup(OptimiserChain(WeightDecay(0.001), Adam()), θ̂)
# ```
#
# See the [`Flux` documentation](https://fluxml.ai/Flux.jl/stable/training/training/#Regularisation) for further details.
