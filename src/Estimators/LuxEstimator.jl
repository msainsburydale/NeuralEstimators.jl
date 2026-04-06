"""
    LuxEstimator(estimator::NeuralEstimator, ps, st)
    LuxEstimator(estimator::NeuralEstimator; rng::AbstractRNG = Random.default_rng())

Wraps a `NeuralEstimator` containing [Lux.jl](https://lux.csail.mit.edu/stable/) 
networks together with their parameters `ps` and states `st`.

The convenience constructor automatically calls `Lux.setup(rng, estimator)` to 
initialise `ps` and `st`.

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
@concrete struct LuxEstimator <: NeuralEstimator
    estimator::Any
    ps::Any
    st::Any
end
function LuxEstimator(estimator::LuxEstimator)
    @warn "estimator is already a LuxEstimator"
    return estimator
end

# Methods - just pass on the estimator and its parameters/states
for f in (:estimate, :summarystatistics, :assess, :sampleposterior, :bootstrap, :interval, :quantiles, :posteriormean, :posteriormedian, :posteriorquantile, :logratio)
    @eval $f(estimator::LuxEstimator, args...; kwargs...) = $f(estimator.estimator, args..., estimator.ps, estimator.st; kwargs...)
end

# Summary network helpers
summarynetwork(estimator::LuxEstimator) = summarynetwork(estimator.estimator)
setsummarynetwork(estimator::LuxEstimator, network) = @set estimator.estimator.summary_network = network
_has_summary_network(e::LuxEstimator) = hasfield(typeof(e.estimator), :summary_network)

# Loss function inherited from estimator type
_loss(estimator::LuxEstimator, loss = nothing) = _loss(estimator.estimator, loss)
