"""
	PointEstimator <: BayesEstimator
    PointEstimator(network)
    PointEstimator(summary_network, inference_network)
    PointEstimator(summary_network, num_parameters; num_summaries, kwargs...)
A neural point estimator, where the neural `network` is a mapping from the sample space to the parameter space.

# Examples
```julia
using NeuralEstimators, Flux, CairoMakie

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
num_parameters = 2     # dimension of the parameter vector θ
n = 1                  # dimension of each independent replicate of Z
m = 50                 # number of independent replicates in each data set
sampler(K) = rand32(num_parameters, K)
simulator(θ, m) = [ϑ[1] .+ ϑ[2] .* randn32(n, m) for ϑ in eachcol(θ)]

# Neural network
w = 128   
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, w, relu), Dense(w, num_parameters))
network = DeepSet(ψ, ϕ)

# Initialise the estimator
estimator = PointEstimator(network)

# Train the estimator
estimator = train(estimator, sampler, simulator, simulator_args = m, K = 1000)

# Plot the risk history
plotrisk()

# Assess the estimator
θ_test = sampler(500)
Z_test = simulator(θ_test, m);
assessment = assess(estimator, θ_test, Z_test)
plot(assessment)

# Inference with "observed" data 
θ = [0.8f0 0.1f0]'
Z = simulator(θ, m)
estimate(estimator, Z)
```
"""
struct PointEstimator{M, N} <: BayesEstimator
    summary_network::M
    inference_network::N
end

# Constructor: summary network, number of parameters, number of summaries => MLP inference network
function PointEstimator(summary_network, num_parameters::Integer, num_summaries::Integer; kwargs...)
    inference_network = MLP(num_summaries, num_parameters; kwargs...)
    @info "PointEstimator: num_summaries = $num_summaries."
    PointEstimator(summary_network, inference_network)
end

# Constructor: num_summaries as keyword
PointEstimator(summary_network, num_parameters::Integer; num_summaries, kwargs...) = PointEstimator(summary_network, num_parameters, num_summaries; kwargs...)

# Constructor: Old workflow, summary network represents the entire network
function PointEstimator(network)
    @info "Constructing PointEstimator with a single network. Consider separating the summary and inference networks as PointEstimator(summary_network, inference_network), which enables additional functionality such as transfer learning and model-misspecification detection."
    PointEstimator(network, identity)
end

# Forward pass: point estimates
(estimator::PointEstimator)(Z) = estimator.inference_network(_summarystatistics(estimator, Z))
