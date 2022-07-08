module NeuralEstimators

# Note that functions must be explicitly imported to be extended with new
# methods. Be aware of type piracy, though.
using Base: @propagate_inbounds
using Base.GC: gc
using BSON: @save, load
using CUDA
using CSV
using DataFrames
using Distributions: Gamma, Normal, cdf, quantile
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: update!
using Functors: @functor
using LinearAlgebra
using Random: randexp
using RecursiveArrayTools: VectorOfArray, convert
using SpecialFunctions: besselk, gamma
using Statistics: mean, median, sum
using Zygote

export ParameterConfigurations, subsetparameters
include("Parameters.jl")

export DeepSet
include("DeepSet.jl")
export DeepSetExpert, samplesize, inversesamplesize
include("DeepSetExpert.jl")
export DeepSetPiecewise
include("DeepSetPiecewise.jl")



export simulate, simulategaussianprocess, simulateschlather, simulateconditionalextremes
export objectindices, matern, maternchols, fₛ, Fₛ, Fₛ⁻¹, scaledlogistic, scaledlogit
include("Simulation.jl")
export incgamma
include("incgamma.jl")

export gaussianloglikelihood, schlatherbivariatedensity
include("densities.jl")

export train
include("Train.jl")

export estimate, Estimates
import Base: merge
include("Estimate.jl")

export parametricbootstrap, nonparametricbootstrap
include("Bootstrap.jl")

export stackarrays, expandgrid, loadbestweights
include("UtilityFunctions.jl")

end
