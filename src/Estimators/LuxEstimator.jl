"""
    LuxEstimator(estimator::AbstractNeuralEstimator, ps, st)
    LuxEstimator(estimator::AbstractNeuralEstimator; rng::AbstractRNG = Random.default_rng())

Wraps a `AbstractNeuralEstimator` containing [Lux.jl](https://lux.csail.mit.edu/stable/) 
networks together with their parameters `ps` and states `st`.

The convenience constructor automatically calls `Lux.setup(rng, estimator)` to 
initialise `ps` and `st`.

Since this type stores the neural-network parameters and states, [`loadestimator`](@ref) 
returns a `LuxEstimator` whenever the estimator contains Lux networks, irrespective of whether 
the estimator passed to it was wrapped in one.

# Examples
```julia
using NeuralEstimators, Lux

network = Lux.Chain(Lux.Dense(10, 64, gelu), Lux.Dense(64, 2))
estimator = LuxEstimator(PointEstimator(network))

# Training, assessment, and inference proceed identically to the Flux API:
estimator = train(estimator, ...)
estimate(estimator, ...)
assess(estimator, ...)

# Access parameters and states directly if needed
estimator.ps
estimator.st
```
"""
@concrete struct LuxEstimator <: AbstractNeuralEstimator
    estimator
    ps
    st
end
function LuxEstimator(estimator::LuxEstimator)
    @warn "estimator is already a LuxEstimator"
    return estimator
end

# Methods - just pass on the estimator and its parameters/states
# NB bootstrap is not forwarded: its methods dispatch on later arguments and only pass the estimator on to estimate(), which
# is forwarded, so that a LuxEstimator needs no method of its own (one taking `args...` would be ambiguous with them)
for f in (:estimate, :infer, :assess, :sampleposterior, :spikeprobability, :interval, :quantiles, :posteriormean, :posteriormedian, :posteriorquantile, :logratio, :logposterior, :_logdensity, :logdensity, :forward, :inverse)
    @eval $f(estimator::LuxEstimator, args...; kwargs...) = $f(estimator.estimator, args..., estimator.ps, estimator.st; kwargs...)
end
# NB summarystatistics has methods for DataAndSummaries data, so it is forwarded with explicit signatures (a method taking
# `args...` would be ambiguous with them)
summarystatistics(estimator::LuxEstimator, Z; kwargs...) = summarystatistics(estimator.estimator, Z, estimator.ps, estimator.st; kwargs...)
summarystatistics(estimator::LuxEstimator, d::DataAndSummaries; kwargs...) = summarystatistics(estimator.estimator, d, estimator.ps, estimator.st; kwargs...)
# NB sample() is StatsBase.sample, so it is forwarded with explicit signatures: a method taking
# (estimator, args...) would be ambiguous with AbstractMCMC's sample(model, ::AbstractSampler, N)
sample(estimator::LuxEstimator; kwargs...) = sample(estimator.estimator, estimator.ps, estimator.st; kwargs...)
sample(estimator::LuxEstimator, N::Integer; kwargs...) = sample(estimator.estimator, N, estimator.ps, estimator.st; kwargs...)

# Summary network helpers
summarynetwork(estimator::LuxEstimator) = summarynetwork(estimator.estimator)
setsummarynetwork(estimator::LuxEstimator, network) = @set estimator.estimator.summary_network = network
_has_summary_network(e::LuxEstimator) = hasfield(typeof(e.estimator), :summary_network)

# The methods above take an untyped second argument, which is ambiguous with the methods that take another estimator
# (setsummarynetwork(::AbstractNeuralEstimator, ::AbstractNeuralEstimator) and likewise for setdatastandardiser) when both
# arguments are a LuxEstimator. These methods are the more specific ones.
function setdatastandardiser(estimator::LuxEstimator, standardiser)
    LuxEstimator(setdatastandardiser(estimator.estimator, standardiser), estimator.ps, estimator.st)
end
function setdatastandardiser(estimator::LuxEstimator, original::AbstractNeuralEstimator)
    setdatastandardiser(estimator, _unmark(datastandardiser(original)))
end
function setsummarynetwork(estimator::LuxEstimator, original::AbstractNeuralEstimator)
    estimator = setsummarynetwork(estimator, summarynetwork(original))
    # Flux stores the summary-network parameters in the network object, which the method above copies. Lux stores them in
    # ps and st, so they are copied too; they are shared with original, as the Flux network object is
    if original isa LuxEstimator
        estimator = @set estimator.ps.summary_network = original.ps.summary_network
        estimator = @set estimator.st.summary_network = original.st.summary_network
    end
    setdatastandardiser(estimator, original)
end

# Loss function inherited from estimator type
_loss(estimator::LuxEstimator, loss = nothing) = _loss(estimator.estimator, loss)

# Standardisers and bounds of the wrapped estimator (see _transformations)
datastandardiser(estimator::LuxEstimator) = datastandardiser(estimator.estimator)
_transformations(estimator::LuxEstimator) = _transformations(estimator.estimator)
_settransformations(estimator::LuxEstimator, entry::NamedTuple) = LuxEstimator(_settransformations(estimator.estimator, entry), estimator.ps, estimator.st)
