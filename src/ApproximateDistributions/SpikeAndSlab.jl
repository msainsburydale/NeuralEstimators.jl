@doc raw"""
    SpikeAndSlab <: AbstractApproximateDistribution
    SpikeAndSlab(num_parameters::Integer, num_summaries::Integer; slab = GaussianMixture, kwargs...)
A spike-and-slab distribution for amortised inference with a [`PosteriorEstimator`](@ref). 

!!! note
    `SpikeAndSlab` currently supports only univariate parameters (i.e., `num_parameters == 1`).

The density of the distribution is a two-component mixture of a point mass (the "spike") at a fixed value $c$ and a continuous distribution (the "slab"): 
```math 
q(\theta; \boldsymbol{\kappa}) = \pi \delta_c(\theta) + (1 - \pi) f_{\textrm{slab}}(\theta), 
```
where $\delta_c(\cdot)$ denotes a point mass at the spike location $c$ (`spike`), $\pi \in (0, 1)$ is the probability of the spike, and $f_{\textrm{slab}}(\cdot)$ is the density of the slab (an [`AbstractApproximateDistribution`](@ref), e.g., a [`GaussianMixture`](@ref)). 

When using a `SpikeAndSlab` distribution as the approximate distribution of a [`PosteriorEstimator`](@ref), the (learned) summary statistics are mapped to the spike probability $\pi$ using a neural network (`classifier`), and to the parameters of the slab as described in the documentation of the chosen slab distribution. 

The slab models the parameter on the real line. The functions `transform` and `invtransform` map between the parameter space and the real line: `transform` is applied to parameters before evaluating the slab density, and `invtransform` is applied to slab samples to map them back to the parameter space. Both default to `identity`. 

# Keyword arguments
- `slab = GaussianMixture`: the slab distribution. May be either a subtype of [`AbstractApproximateDistribution`](@ref) (constructed internally with `num_parameters`, `num_summaries`, and any additional `kwargs`) or a pre-constructed [`AbstractApproximateDistribution`](@ref).
- `spike::Real = 0`: the location of the spike (point mass) in the parameter space.
- `transform = identity`: a function mapping parameters to the real line. 
- `invtransform = identity`: a function mapping the real line to the parameter space. 
- `classifier_kwargs = (;)`: keyword arguments passed to the [`MLP`](@ref) used for the classifier. 
- `kwargs`: additional keyword arguments passed to the constructor of the `slab` distribution. 

The posterior probability of the spike (i.e., $\pi$) for observed data can be obtained using [`spikeprobability`](@ref). 

# Examples
```julia
using NeuralEstimators, Flux

# Simple linear regression Z = (x, y) with y = βx + ε, ε ~ N(0, 0.1²) and covariate x ~ U(0, 1).
# Spike-and-slab prior on the slope β: β = 0 with probability 1/2, else β ~ U(-1, 1).
d = 1   # number of parameters (the slope β)
m = 30  # number of (x, y) pairs in each data set

function sampler(K)
    spike = rand(K) .< 0.5
    β = ifelse.(spike, 0f0, 2f0 .* rand(Float32, K) .- 1f0)
    NamedMatrix(β = β)
end

function simulator(θ::AbstractMatrix, m::Integer)
    map(eachcol(θ)) do θₖ
        x = rand(Float32, 1, m)
        y = θₖ["β"] .* x .+ 0.1f0 .* randn(Float32, 1, m)
        vcat(x, y)  # 2×m: each column is one (x, y) pair
    end
end

# Summary network: a DeepSet over the exchangeable (x, y) pairs
num_summaries = 32
ψ = Chain(Dense(2, 64, relu), Dense(64, 64, relu))
ϕ = Chain(Dense(64, 64, relu), Dense(64, num_summaries))
summary_network = DeepSet(ψ, ϕ)

# Spike-and-slab posterior estimator (spike at β = 0)
estimator = PosteriorEstimator(summary_network, d; num_summaries = num_summaries, q = SpikeAndSlab)
estimator = train(estimator, sampler, simulator, simulator_args = m, K = 5000, epochs = 20)

# Inference for data simulated with β = 0 (the spike)
Z = simulator(NamedMatrix(β = [0f0]), m)
spikeprobability(estimator, Z)   # posterior probability that β = 0
sampleposterior(estimator, Z)    # posterior draws (a mix of exact zeros and slab draws)
```
"""
@concrete struct SpikeAndSlab <: AbstractApproximateDistribution
    classifier
    slab
    spike
    transform
    invtransform
end
Optimisers.trainable(q::SpikeAndSlab) = (classifier = q.classifier, slab = q.slab)

function SpikeAndSlab(
    num_parameters::Integer,
    num_summaries::Integer;
    slab = GaussianMixture,
    spike::Real = 0,
    transform = identity,
    invtransform = identity,
    classifier_kwargs = (;),
    backend::Union{Nothing, Module} = nothing,
    kwargs...
)
    @assert num_parameters == 1 "SpikeAndSlab currently supports only univariate parameters (num_parameters == 1)"
    B = _resolvebackend(backend)
    classifier = MLP(num_summaries, 1; backend = B, output_activation = identity, classifier_kwargs...)
    slab_dist = slab isa AbstractApproximateDistribution ? slab : slab(num_parameters, num_summaries; backend = B, kwargs...)
    SpikeAndSlab(classifier, slab_dist, spike, transform, invtransform)
end

# Constructor from pre-built classifier and slab
SpikeAndSlab(classifier, slab::AbstractApproximateDistribution; spike::Real = 0, transform = identity, invtransform = identity) =
    SpikeAndSlab(classifier, slab, spike, transform, invtransform)

# One distributional parameter for the spike probability, plus those of the slab
numdistributionalparams(q::SpikeAndSlab) = 1 + numdistributionalparams(q.slab)

# ── Spike probability ───────────────────────────────────────────────────────────
# π given the (learned) summary statistics t; returns a 1×K matrix.

function spikeprobability(q::SpikeAndSlab, t::AbstractMatrix; device = nothing)
    device = cpu_device()
    q = q |> device
    t = t |> device
    return sigmoid.(q.classifier(t))
end

function spikeprobability(q::SpikeAndSlab, t::AbstractMatrix, ps_q, st_q; device = nothing)
    device = cpu_device()
    ps_q = ps_q |> device
    st_q = st_q |> device
    t = t |> device
    logit, _ = q.classifier(t, ps_q.classifier, st_q.classifier)
    return sigmoid.(logit)
end

# ── Stateful (Flux) ────────────────────────────────────────────────────────────

function _logdensity(q::SpikeAndSlab, θ::AbstractMatrix, tz::AbstractMatrix)
    d, K = size(θ)
    @assert d == 1
    @assert K == size(tz, 2)

    logit = q.classifier(tz)        # 1×K (unconstrained spike logit)

    is_spike = θ .== q.spike        # 1×K

    z = q.transform.(θ)             # map parameters to the real line
    # Replace spike entries with a safe value so that an undefined transform(spike)
    # (e.g., log(0)) cannot poison the slab density or its gradient
    z = ifelse.(is_spike, zero(eltype(z)), z)

    slab_ld = _logdensity(q.slab, z, tz) # 1×K

    log_spike = logσ.(logit)              # log π
    log_slab = logσ.(-logit) .+ slab_ld   # log(1 - π) + slab log-density
    T = promote_type(eltype(log_spike), eltype(log_slab))
    log_densities = ifelse.(is_spike, T.(log_spike), T.(log_slab))
    return log_densities
end

function sampleposterior(q::SpikeAndSlab, tz::AbstractMatrix, N::Integer; device = nothing)
    device = cpu_device()
    q = q |> device
    tz = tz |> device

    π = sigmoid.(q.classifier(tz))                       # 1×K
    slab_samples = sampleposterior(q.slab, tz, N; device = device) # 1 × N × K on the real line

    x = q.invtransform.(slab_samples)   # 1 × N × K in parameter space
    spike = convert(eltype(x), q.spike)
    spike_draws = rand(1, N, size(tz, 2)) .< reshape(π, 1, 1, :)
    return ifelse.(spike_draws, spike, x)
end

# ── Stateless (Lux) ────────────────────────────────────────────────────────────

function _logdensity(q::SpikeAndSlab, θ::AbstractMatrix, tz::AbstractMatrix, ps_q, st_q)
    d, K = size(θ)
    @assert d == 1
    @assert K == size(tz, 2)

    logit, st_c = q.classifier(tz, ps_q.classifier, st_q.classifier)

    is_spike = θ .== q.spike

    z = q.transform.(θ)
    z = ifelse.(is_spike, zero(eltype(z)), z)

    slab_ld, st_s = _logdensity(q.slab, z, tz, ps_q.slab, st_q.slab)

    log_spike = logσ.(logit)              # log π
    log_slab = logσ.(-logit) .+ slab_ld   # log(1 - π) + slab log-density
    T = promote_type(eltype(log_spike), eltype(log_slab))
    log_densities = ifelse.(is_spike, T.(log_spike), T.(log_slab))

    st_q = merge(st_q, (classifier = st_c, slab = st_s))
    return log_densities, st_q
end

function sampleposterior(q::SpikeAndSlab, tz::AbstractMatrix, N::Integer, ps_q, st_q; device = nothing)
    device = cpu_device()
    ps_q = ps_q |> device
    st_q = st_q |> device
    tz = tz |> device

    logit, _ = q.classifier(tz, ps_q.classifier, st_q.classifier)
    π = sigmoid.(logit)                                       # 1×K
    slab_samples = sampleposterior(q.slab, tz, N, ps_q.slab, st_q.slab; device = device)

    x = q.invtransform.(slab_samples)
    spike = convert(eltype(x), q.spike)
    spike_draws = rand(1, N, size(tz, 2)) .< reshape(π, 1, 1, :)
    return ifelse.(spike_draws, spike, x)
end
