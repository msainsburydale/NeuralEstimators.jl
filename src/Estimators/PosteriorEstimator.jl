@doc raw"""
	PosteriorEstimator <: NeuralEstimator
	PosteriorEstimator(summary_network, q::ApproximateDistribution)
	PosteriorEstimator(summary_network, num_parameters::Integer; num_summaries::Integer, q = nothing, kwargs...)
A neural estimator that approximates the posterior distribution $p(\boldsymbol{\theta} \mid \boldsymbol{Z})$, based on a neural `summary_network` and an approximate distribution `q` (see the available in-built [Approximate distributions](@ref)).

The `summary_network` maps data $\boldsymbol{Z}$ to a vector of learned summary statistics $\boldsymbol{t} \in \mathbb{R}^{d^*}$, which are then used to condition the approximate distribution `q`. The precise way in which the summary statistics condition `q` depends on the choice of approximate distribution: for example, [`GaussianMixture`](@ref) uses an MLP to map $\boldsymbol{t}$ directly to distributional parameters, while [`NormalisingFlow`](@ref) uses $\boldsymbol{t}$ as a conditioning input at each coupling layer.

The convenience constructor builds `q` internally given `num_parameters` and `num_summaries`, with any additional keyword arguments passed to the constructor of `q`.

# Keyword arguments
- `num_summaries::Integer`: the number of summary statistics output by `summary_network`. Must match the output dimension of `summary_network`.
- `q::Type{<:ApproximateDistribution}`: the type of approximate distribution to use. Defaults to `NormalisingFlow` when using `Flux`, and `GaussianMixture` when using `Lux`.
- `kwargs...`: additional keyword arguments passed to the constructor of `q`.

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ N(0, 1) and σ ~ U(0, 1)
d, m = 2, 100  # dimension of θ and number of replicates
sampler(K) = NamedMatrix(μ = randn(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(m))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

# Neural network
num_summaries = 3d
summary_network = Chain(Dense(m, 64, gelu), Dense(64, 64, gelu), Dense(64, num_summaries))

# Initialise the estimator, with q built internally
estimator = PosteriorEstimator(summary_network, d; num_summaries = num_summaries)

# Or, build q explicitly
q = NormalisingFlow(d; num_summaries = num_summaries)
estimator = PosteriorEstimator(summary_network, q)

# Training
estimator = train(estimator, sampler, simulator, K = 3000)

# Assess the estimator
θ_test = sampler(250)
Z_test = simulator(θ_test);
assessment = assess(estimator, θ_test, Z_test)

# Inference with observed data 
θ = sampler(1)
Z = simulator(θ)
sampleposterior(estimator, Z) # posterior draws
posteriormean(estimator, Z)   # point estimate
```
"""
@concrete struct PosteriorEstimator <: NeuralEstimator
    summary_network
    q
end

# Constructor: summary network, number of parameters, number of summaries => build approximate distribution automatically
function PosteriorEstimator(summary_network, num_parameters::Integer, num_summaries::Integer; q = nothing, kwargs...)
    # Convert string to type if needed
    q = if q isa String
        getfield(@__MODULE__, Symbol(q))
    else
        q
    end
    @info "PosteriorEstimator: num_summaries = $num_summaries."
    backend = _backendof(summary_network)
    # Default approximate distribution depends on backend:
    # GaussianMixture for Lux, NormalisingFlow for Flux
    if isnothing(q)
        lux = get(Base.loaded_modules, _LUX_UUID, nothing)
        q = backend === lux ? GaussianMixture : NormalisingFlow
    end
    PosteriorEstimator(summary_network, q(num_parameters, num_summaries; backend = backend, kwargs...))
end

# Constructor: keyword num_summaries
PosteriorEstimator(summary_network, num_parameters::Integer; num_summaries::Integer, kwargs...) = PosteriorEstimator(summary_network, num_parameters, num_summaries; kwargs...)

# Constructor: consistent argument ordering 
function PosteriorEstimator(q::A, summary_network) where {A <: ApproximateDistribution}
    return PosteriorEstimator(summary_network, q)
end

numdistributionalparams(estimator::PosteriorEstimator) = numdistributionalparams(estimator.q)
_loss(estimator::PosteriorEstimator, loss = nothing) = (logdensity, θ) -> -mean(logdensity)

function _inputoutput(estimator::PosteriorEstimator, Z, θ)
    input = (Z, θ)           # combine data and parameters into a single tuple which will be used to compute the log-density at θ
    output_placeholder = θ   # currently irrelevant what we use for the output, since the loss is defined only in terms of the density at θ. However, this will change with sequential training.
    return input, output_placeholder
end

# TODO this is not made public yet because (i) need to define an public/internal versions (logdensity/_logdensity) since Lux versions returns st (just need the public wrapper to drop st from the return type), and (ii) it is not currently set up to handle a single data set + multiple parameter values efficiently, which clashes with the behaviour of logratio

"""
	_logdensity(estimator::PosteriorEstimator, θ, Z)
    _logdensity(estimator::PosteriorEstimator, θ, Z, ps, st)
Evaluates the log-density of the approximate posterior implied by `estimator` at parameters `θ` given data `Z`,
```math
\\log q(\\boldsymbol{\\theta} \\mid \\boldsymbol{Z}),
```
where ``q`` denotes the approximate posterior distribution.

`θ` should be a ``d \\times K`` matrix of parameter vectors and `Z` a collection of `K` data sets.

# Returns 
A ``1 \\times K`` matrix of log-densities, where entry ``k`` is ``\\log q(\\boldsymbol{\\theta}_k \\mid \\boldsymbol{Z}_k)``.

When using a Lux model, returns a tuple `(log_densities, st)` where `log_densities` is the ``1 \\times K`` matrix and `st` is the updated network state.

See also [`sampleposterior`](@ref).
"""
function _logdensity end

# Forward pass: Stateful (Flux)
_logdensity(estimator::PosteriorEstimator, θ, Z) = _logdensity(estimator.q, θ, _summarystatistics(estimator, Z))

# Forward pass: Stateless (Lux)
function _logdensity(estimator::PosteriorEstimator, θ, Z, ps, st)
    tz, st_s = _summarystatistics(estimator, Z, ps.summary_network, st.summary_network)
    ld, st_q = _logdensity(estimator.q, θ, tz, ps.q, st.q)
    return ld, (summary_network = st_s, q = st_q)
end

# Tuple methods used internally during training
(estimator::PosteriorEstimator)(Zθ::Tuple) = _logdensity(estimator, Zθ[2], Zθ[1])
(estimator::PosteriorEstimator)(Zθ::Tuple, ps, st) = _logdensity(estimator, Zθ[2], Zθ[1], ps, st)

# Inference: Stateful (Flux)
function sampleposterior(estimator::PosteriorEstimator, Z; N::Integer = 1000, device = nothing, use_gpu::Bool = true, kwargs...)
    device = _resolvedevice(device = device, use_gpu = use_gpu, verbose = false)
    t = summarystatistics(estimator, Z; device = device, kwargs...)
    θ = sampleposterior(estimator.q, t, N; device = device)
    return length(θ) == 1 ? θ[1] : θ
end

# Inference: Stateless (Lux)
function sampleposterior(estimator::PosteriorEstimator, Z, ps, st; N::Integer = 1000, device = nothing, use_gpu::Bool = true, kwargs...)
    device = _resolvedevice(device = device, use_gpu = use_gpu, verbose = false)
    t = summarystatistics(estimator, Z, ps, st; device = device, kwargs...)
    θ = sampleposterior(estimator.q, t, N, ps.q, st.q; device = device)
    return length(θ) == 1 ? θ[1] : θ
end
