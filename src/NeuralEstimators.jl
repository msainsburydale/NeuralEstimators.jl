module NeuralEstimators

# Note that functions must be explicitly imported to be extended with new
# methods. Be aware of type piracy, though.
using Base: @propagate_inbounds
using Base.GC: gc
using BSON: @save, load
using CUDA
using CSV
using ChainRulesCore: @non_differentiable
using DataFrames
using Distributions
import Distributions: cdf, logpdf, quantile, minimum, maximum, insupport, var, skewness
using Flux
using Flux: ofeltype
using Flux.Data: DataLoader
using Flux.Optimise: update!
using Functors: @functor
using LinearAlgebra
using Random: randexp
using RecursiveArrayTools: VectorOfArray, convert
using SpecialFunctions: besselk, gamma, loggamma
using Statistics: mean, median, sum
import Statistics: mean, quantile
using Zygote


export ParameterConfigurations, subsetparameters
include("Parameters.jl")

export DeepSet
include("DeepSet.jl")
# export DeepSetExpert, samplesize, inversesamplesize
# include("DeepSetExpert.jl")
export PiecewiseEstimator
include("PiecewiseEstimator.jl")

export simulate, simulategaussianprocess, simulateschlather, simulateconditionalextremes
export matern, maternchols, Subbotin, scaledlogistic, scaledlogit
include("Simulation.jl")
export incgamma
include("incgamma.jl")

export gaussiandensity, schlatherbivariatedensity
include("densities.jl")

export LP, LPsafe
include("loss.jl")

export train, trainMAP
include("Train.jl")

export assess, Assessment
include("Assess.jl")

export parametricbootstrap, nonparametricbootstrap, coverage
include("Bootstrap.jl")

export stackarrays, expandgrid, loadbestweights
include("UtilityFunctions.jl")

end


# TODO once I've made the repo public:
# •	Contact TravisCI to tell them that I am developing open-source software to get a free plan.
# •	code coverage widget should display the percentage.
# •	Get documentation online through Github pages:
#   See https://juliadocs.github.io/Documenter.jl/v0.13/man/hosting/
#   Get it working by manually deploying first, then get it working automatically by deploying with TravisCI.
#   Once documentation is online, add a widget in the README linking to the documentation.

# TODO long term:
# - Precompile NeuralEstimators.jl to reduce latency: See https://julialang.org/blog/2021/01/precompile_tutorial/. It seems very easy, just need to add precompile(f, (arg_types…)) to whatever methods I want to precompile.
# - Get DeepSetExpert working optimally on the GPU (leaving this for now as we don't need it for the paper).
# - See if DeepSet.jl is something that the Flux people would like to include. (They may also improve the code.)
# - With the fixed parameters method of train, there seems to be substantial overhead with my current implementation of simulation on the fly. When epochs_per_Z_refresh = 1, the run-time increases by a factor of 4 for the Gaussian process with nu varied and with m = 1. For now, I’ve added an argument simulate_on_the_fly::Bool, which allows us not to switch off on-the-fly simulation even when epochs_per_Z_refresh = 1. However, it would be good to reduce this overhead.
# - Callback function for plotting during training! See https://www.youtube.com/watch?v=ObYDHi_jJXk&ab_channel=TheJuliaProgrammingLanguage. Also, I know there is a specific module for call backs while training Flux models, so may this is already possible in Julia too. In either case, I think train() should have an additional argument, callback. See also the example at: https://github.com/stefan-m-lenz/JuliaConnectoR.
# - Frameworks based on Neyman inversion allow for confidence sets with correct conditional coverage.
# - Include julia versions of plotrisk() and plotjointdistribution(). Then, NeuralEstimators.jl will be self contained. A nice way to do this would be to Julia RCall() to NeuralEstimatorsR.


# Some points for writing the manuscript later.
# Why Julia? Julia has many [attractive features](https://julialang.org/); in
# particular, it has been designed to alleviate the so-called two-language
# problem, so that it is both easily developed and fast in its execution. This
# means that users can write performant code for data simulation without needing
# to vectorise (i.e., `for` loops are fine, which is often very helpful).
# Further, many Julia packages are written entirely in Julia and, hence, their
# source code is easily understood and extended; this includes
# `NeuralEstimators` and the deep learning framework on which it is built upon,
# [Flux](https://fluxml.ai/Flux.jl/stable/).
