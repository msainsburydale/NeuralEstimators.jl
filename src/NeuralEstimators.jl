module NeuralEstimators

#TODO maybe add structs for PointEstimator and CIEstimator, QuantileEstimator. Then, DeepSet,
#     DeepSetExpert, and GNNEstimator can just be a field of these estimators
#     (also need to rename GNNEstimator to something else).
#TODO Need to update subsetdata()
#TODO Need to update numberreplicates()


# Note that functions must be explicitly imported to be extended with new methods. Be aware of type piracy, though.
using Base: @propagate_inbounds, @kwdef
using Base.GC: gc
import Base: merge
import Base: size
using BSON: @save, load
using ChainRulesCore: @non_differentiable, @ignore_derivatives
using CUDA
using CSV
using DataFrames
using Distributions
import Distributions: cdf, logpdf, quantile, minimum, maximum, insupport, var, skewness
using Flux
using Flux: ofeltype, params, DataLoader, update!
using Functors: @functor
using GraphNeuralNetworks
using GaussianRandomFields
using Graphs
using LinearAlgebra
using NamedArrays
using Random: randexp
using RecursiveArrayTools: VectorOfArray, convert
using SpecialFunctions: besselk, gamma, loggamma
using Statistics: mean, median, sum
using Zygote

export kpowerloss, intervalscore, quantileloss
include("loss.jl")

export ParameterConfigurations, subsetparameters
include("Parameters.jl")

# ---- Estimator types ----

export CIEstimator, PiecewiseEstimator # TODO PointEstimator, QuantileEstimator
include("Estimators.jl")

# ---- Architectures and useful layers ----

export DeepSet, DeepSetExpert, GNNEstimator, Compress #TODO should change the name of GNNEstimator
include("Architectures.jl")

export simulate, simulategaussianprocess, simulateschlather, simulateconditionalextremes
export matern, maternchols, scaledlogistic, scaledlogit, incgamma
include("simulate.jl")

export gaussiandensity, schlatherbivariatedensity
include("densities.jl")

export train, subsetdata
include("train.jl")

export assess, Assessment, merge, risk
include("assess.jl")

export plotrisk, plotdistribution
include("plotting.jl")

export bootstrap, coverage, confidenceinterval
include("bootstrap.jl")

export stackarrays, expandgrid, loadbestweights, numberreplicates, nparams, samplesize
include("utility.jl")



end



# ---- long term:
# - Decide if we want a NeuralEstimator class (see src/devel/NeuralEstimator.jl)
# - record the example clip, but while ssh'd into my office computer (with the GPU).
# - README.md
# - plotrisk and plotdistribution (wait until the R interface is finished)
# - Add "AR(k) time series" and "Irregular spatial data" examples. (The former will be an example using partially exchangeable neural networks and the latter will be an example using GNNs.)
# - Precompile NeuralEstimators.jl to reduce latency: See https://julialang.org/blog/2021/01/precompile_tutorial/. It seems very easy, just need to add precompile(f, (arg_types…)) to whatever methods I want to precompile.
# - Get DeepSetExpert working optimally on the GPU (leaving this for now as we don't need it for the paper).
# - See if DeepSet.jl is something that the Flux people would like to include. (They may also improve the code.)
# - With the fixed parameters method of train, there seems to be substantial overhead with my current implementation of simulation on the fly. When epochs_per_Z_refresh = 1, the run-time increases by a factor of 4 for the Gaussian process with nu varied and with m = 1. For now, I’ve added an argument simulate_on_the_fly::Bool, which allows us not to switch off on-the-fly simulation even when epochs_per_Z_refresh = 1. However, it would be good to reduce this overhead.
# - Callback function for plotting during training. See https://www.youtube.com/watch?v=ObYDHi_jJXk&ab_channel=TheJuliaProgrammingLanguage. Also, I know there is a specific module for call backs while training Flux models, so may this is already possible in Julia too. In either case, I think train() should have an additional argument, callback. See also the example at: https://github.com/stefan-m-lenz/JuliaConnectoR.
# - Frameworks based on Neyman inversion allow for confidence sets with correct conditional coverage.
# - Keep the Bill Venables award in mind: https://statsocaus.github.io/venables-award/

# ---- once I've made the project public:
# - Add NeuralEstimators.jl to the list of packages that use Documenter: see https://documenter.juliadocs.org/stable/man/examples/
