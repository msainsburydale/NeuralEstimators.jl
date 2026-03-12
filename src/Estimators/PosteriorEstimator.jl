@doc raw"""
	PosteriorEstimator <: NeuralEstimator
	PosteriorEstimator(summary_network, q::ApproximateDistribution)
	PosteriorEstimator(summary_network, num_parameters::Integer; num_summaries::Integer, q = NormalisingFlow, kwargs...)
A neural estimator that approximates the posterior distribution $p(\boldsymbol{\theta} \mid \boldsymbol{Z})$, based on a neural `summary_network` and an approximate distribution `q` (see the available in-built [Approximate distributions](@ref)).

The `summary_network` maps data $\boldsymbol{Z}$ to a vector of learned summary statistics $\boldsymbol{t} \in \mathbb{R}^{d^*}$, which are then used to condition the approximate distribution `q`. The precise way in which the summary statistics condition `q` depends on the choice of approximate distribution: for example, [`GaussianMixture`](@ref) uses an MLP to map $\boldsymbol{t}$ directly to distributional parameters, while [`NormalisingFlow`](@ref) uses $\boldsymbol{t}$ as a conditioning input at each coupling layer.

The convenience constructor builds `q` internally given `num_parameters` and `num_summaries`, with any additional keyword arguments passed to the constructor of `q`.

# Keyword arguments
- `num_summaries::Integer`: the number of summary statistics output by `summary_network`. Must match the output dimension of `summary_network`.
- `q::Type{<:ApproximateDistribution} = NormalisingFlow`: the type of approximate distribution to use.
- `kwargs...`: additional keyword arguments passed to the constructor of `q`.

# Examples
```julia
using NeuralEstimators, Flux, CairoMakie

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
num_parameters = 2  # dimension of the parameter vector θ
n = 1               # dimension of each independent replicate of Z
m = 50              # number of independent replicates in each data set
sampler(K) = rand32(num_parameters, K)
simulator(θ, m) = [ϑ[1] .+ ϑ[2] .* randn32(n, m) for ϑ in eachcol(θ)]

# Summary network 
num_summaries = 4num_parameters
w = 128   
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, w, relu), Dense(w, num_summaries))
summary_network = DeepSet(ψ, ϕ)

# Initialise the estimator, with q built internally
estimator = PosteriorEstimator(summary_network, num_parameters; num_summaries = num_summaries)

# Or, build q explicitly
q = NormalisingFlow(num_parameters; num_summaries = num_summaries)
estimator = PosteriorEstimator(summary_network, q)

# Train the estimator
estimator = train(estimator, sampler, simulator, simulator_args = m, K = 3000)

# Plot the risk history
plotrisk()

# Assess the estimator
θ_test = sampler(500)
Z_test = simulator(θ_test, m);
assessment = assess(estimator, θ_test, Z_test)
plot(assessment)

# Inference with observed data 
θ = [0.8f0 0.1f0]'
Z = simulator(θ, m)
sampleposterior(estimator, Z) # posterior draws 
posteriormean(estimator, Z)   # point estimate
```
"""
struct PosteriorEstimator{Q, N} <: NeuralEstimator
    q::Q
    summary_network::N
end

# Constructor: summary network, number of parameters, number of summaries => build approximate distribution automatically
function PosteriorEstimator(summary_network, num_parameters::Integer, num_summaries::Integer; q = NormalisingFlow, kwargs...)
    # Convert string to type if needed
    q = if q isa String
        getfield(@__MODULE__, Symbol(q))
    else
        q
    end
    @info "PosteriorEstimator: num_summaries = $num_summaries."
    return PosteriorEstimator(q(num_parameters, num_summaries; kwargs...), summary_network)
end

# Constructor: keyword num_summaries
PosteriorEstimator(summary_network, num_parameters::Integer; num_summaries::Integer, kwargs...) = PosteriorEstimator(summary_network, num_parameters, num_summaries; kwargs...)

# Constructor: consistent argument ordering 
function PosteriorEstimator(summary_network, q::A) where {A <: ApproximateDistribution}
    return PosteriorEstimator(q, summary_network)
end

# Forward pass: log density
"""
	logdensity(estimator::PosteriorEstimator, θ, Z)
Evaluates the log-density of the approximate posterior implied by `estimator` at parameters `θ` given data `Z`,
```math
\\log q(\\boldsymbol{\\theta} \\mid \\boldsymbol{Z}),
```
where ``q`` denotes the approximate posterior distribution.

`θ` should be a ``d \\times K`` matrix of parameter vectors and `Z` a collection of `K` data sets.

See also [`sampleposterior`](@ref).
"""
logdensity(estimator::PosteriorEstimator, θ, Z) = logdensity(estimator.q, f32(θ), _summarystatistics(estimator, f32(Z)))
(estimator::PosteriorEstimator)(Zθ::Tuple) = logdensity(estimator, Zθ[2], Zθ[1]) # internal method only used during training 

numdistributionalparams(estimator::PosteriorEstimator) = numdistributionalparams(estimator.q)

# Always use the KL divergence to train objects of type PosteriorEstimator
_loss(estimator::PosteriorEstimator, loss = nothing) = (q, θ) -> -mean(q)

function _inputoutput(estimator::PosteriorEstimator, Z, θ)
    input = (Z, θ)           # combine data and parameters into a single tuple which will be used to compute the log-density at θ
    output_placeholder = θ   # currently irrelevant what we use for the output, since the loss is defined only in terms of the density at θ. However, this will change with sequential training.
    return input, output_placeholder
end

function sampleposterior(estimator::PosteriorEstimator, Z, N::Integer = 1000; use_gpu::Bool = true, kwargs...)

    # Compute the summary statistics
    t = summarystatistics(estimator, Z; use_gpu = use_gpu) #TODO batchsize argument

    # Sample from the approximate posterior given the summary statistics 
    θ = sampleposterior(estimator.q, t, N; kwargs...)

    if length(θ) == 1
        θ = θ[1]
    end

    return θ
end
