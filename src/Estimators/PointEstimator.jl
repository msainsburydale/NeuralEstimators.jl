"""
	PointEstimator <: BayesEstimator
    PointEstimator(network)
    PointEstimator(summary_network, inference_network)
    PointEstimator(summary_network, num_parameters; num_summaries, kwargs...)
A neural point estimator mapping data to a point summary of the posterior distribution.

The neural network can be provided in two ways:
- As a single `network` that maps data directly to the parameter space.
- As a `summary_network` that maps data to a vector of summary statistics, with the `inference_network` constructed internally based on `num_parameters` and `num_summaries`.

# Examples
```julia
using NeuralEstimators, Flux, CairoMakie

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ N(0, 1) and σ ~ U(0, 1)
d, m = 2, 100  # dimension of θ and number of replicates
sampler(K) = NamedMatrix(μ = randn(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(m))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

# Neural network, an MLP mapping m inputs into d outputs
network = Chain(Dense(m, 64, gelu), Dense(64, 64, gelu), Dense(64, d))

# Initialise a neural point estimator
estimator = PointEstimator(network)

# Train the estimator
estimator = train(estimator, sampler, simulator)

# Plot the risk history
plotrisk()

# Assess the estimator
θ_test = sampler(1000)
Z_test = simulator(θ_test)
assessment = assess(estimator, θ_test, Z_test)
bias(assessment)
rmse(assessment)
plot(assessment)

# Apply to observed data (here, simulated as a stand-in)
θ = sampler(1)           # ground truth (not known in practice)
Z = simulator(θ)         # stand-in for real observations
estimate(estimator, Z)   # point estimate
```
"""
@concrete struct PointEstimator <: BayesEstimator
    summary_network
    inference_network
end

# Constructor: summary network, number of parameters, number of summaries => MLP inference network
function PointEstimator(summary_network, num_parameters::Integer, num_summaries::Integer; kwargs...)
    backend = _backendof(summary_network)
    inference_network = MLP(num_summaries, num_parameters; backend = backend, kwargs...)
    @info "PointEstimator: num_summaries = $num_summaries."
    estimator = PointEstimator(summary_network, inference_network)
    estimator
end

# Constructor: num_summaries as keyword
PointEstimator(summary_network, num_parameters::Integer; num_summaries, kwargs...) = PointEstimator(summary_network, num_parameters, num_summaries; kwargs...)

# Constructor: Old workflow, summary network represents the entire network
function PointEstimator(network)
    @info "Constructing PointEstimator with a single network. Consider separating the summary and inference networks as PointEstimator(summary_network, inference_network), which enables additional functionality." # such as transfer learning and model-misspecification detection."
    backend = _backendof(network)
    PointEstimator(network, _identity_layer(backend))
end
_identity_layer(backend::Module) = _identity_layer(Val(nameof(backend)))
_identity_layer(::Val{:Flux}) = identity  # plain Julia function, valid as a Flux layer
_is_identity(f) = f === identity || f isa typeof(identity) || f isa Identity

# Forward pass: Stateful (Flux)
(estimator::PointEstimator)(Z) = estimator.inference_network(_summarystatistics(estimator, Z))

# Forward pass: Stateless (Lux)
function (e::PointEstimator)(Z, ps, st)
    t, st_s = _summarystatistics(e, Z, ps.summary_network, st.summary_network)
    y, st_i = e.inference_network(t, ps.inference_network, st.inference_network)
    return y, (summary_network = st_s, inference_network = st_i)
end

