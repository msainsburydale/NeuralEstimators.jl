@doc raw"""
	PosteriorEstimator <: NeuralEstimator
	PosteriorEstimator(network, q::ApproximateDistribution)
    PosteriorEstimator(network, d::Integer, dstar::Integer = d; q::ApproximateDistribution = NormalisingFlow, kwargs...)
A neural estimator that approximates the posterior distribution $p(\boldsymbol{\theta} \mid \boldsymbol{Z})$, based on a neural `network` and an approximate distribution `q` (see the available in-built [Approximate distributions](@ref)). 

The neural network is a mapping from the sample space to $\mathbb{R}^{d^*}$, where $d^*$ is a user-specified number of summary statistics. The learned summary statistics are then transformed into parameters $\boldsymbol{\kappa} \in \mathcal{K}$ of the approximate distribution  using a conventional multilayer perceptron (MLP).

The convenience constructor `PosteriorEstimator(network, d, dstar; q::ApproximateDistribution)` builds the approximate distribution `q` internally, with the keyword arguments passed onto the constructor of `q`.  

# Examples
```julia
using NeuralEstimators, Flux, CairoMakie

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d = 2     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 50    # number of independent replicates in each data set
sample(K) = rand32(d, K)
simulate(θ, m) = [ϑ[1] .+ ϑ[2] .* randn32(n, m) for ϑ in eachcol(θ)]

# Distribution used to approximate the posterior 
q = NormalisingFlow(d, d) 

# Neural network (outputs d summary statistics)
w = 128   
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, w, relu), Dense(w, d))
network = DeepSet(ψ, ϕ)

estimator = PosteriorEstimator(network, GaussianMixture(d, d))

# Initialise the estimator
estimator = PosteriorEstimator(network, q)

# Train the estimator
estimator = train(estimator, sample, simulate, simulator_args = m, K = 3000)

# Plot the risk history
plotrisk()

# Assess the estimator
θ_test = sample(500)
Z_test = simulate(θ_test, m);
assessment = assess(estimator, θ_test, Z_test)
plot(assessment)

# Inference with observed data 
θ = [0.8f0 0.1f0]'
Z = simulate(θ, m)
sampleposterior(estimator, Z) # posterior draws 
posteriormean(estimator, Z)   # point estimate
```
"""
struct PosteriorEstimator{Q, N} <: NeuralEstimator
    q::Q
    network::N
end
numdistributionalparams(estimator::PosteriorEstimator) = numdistributionalparams(estimator.q)
logdensity(estimator::PosteriorEstimator, θ, Z) = logdensity(estimator.q, f32(θ), estimator.network(f32(Z)))
(estimator::PosteriorEstimator)(Zθ::Tuple) = logdensity(estimator, Zθ[2], Zθ[1]) # internal method only used during training # TODO not ideal that we assume an ordering here

# Convenience constructor
function PosteriorEstimator(network, d::Integer, dstar::Integer = d; q = NormalisingFlow, kwargs...)

    # Convert string to type if needed
    q = if q isa String
        # Get the type from the string name
        getfield(@__MODULE__, Symbol(q))
    else
        q
    end

    # Distribution used to approximate the posterior 
    q = q(d, dstar; kwargs...)

    # Initialise the estimator
    return PosteriorEstimator(q, network)
end

# Constructor for consistent argument ordering
function PosteriorEstimator(network, q::A) where {A <: ApproximateDistribution}
    return PosteriorEstimator(q, network)
end

# Always use the KL divergence to train objects of type PosteriorEstimator
_loss(estimator::PosteriorEstimator, loss = nothing) = (q, θ) -> -mean(q)

function _inputoutput(estimator::PosteriorEstimator, Z, θ::P) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    θ = _extractθ(θ)
    input = (Z, θ)           # combine data and parameters into a single tuple which will be used to compute the log-density at θ
    output_placeholder = θ   # irrelevant what we use for the output, since the loss is defined only in terms of the density at θ
    return input, output_placeholder
end

function sampleposterior(estimator::PosteriorEstimator, Z, N::Integer = 1000; use_gpu::Bool = true, kwargs...)

    # Compute the summary statistics 
    t = estimate(estimator.network, Z; use_gpu = use_gpu)

    # Sample from the approximate posterior given the summary statistics 
    θ = sampleposterior(estimator.q, t, N; kwargs...)

    if length(θ) == 1
        θ = θ[1]
    end

    return θ
end
