@doc raw"""
	PosteriorEstimator <: AbstractNeuralEstimator
	PosteriorEstimator(summary_network, q::AbstractApproximateDistribution; lower = nothing, upper = nothing, transform = identity, invtransform = identity, standardise_data = true, standardise_parameters = true)
	PosteriorEstimator(num_parameters::Integer, summary_network = identity; num_summaries::Integer, q::AbstractApproximateDistribution = NormalisingFlow, lower = nothing, upper = nothing, transform = identity, invtransform = identity, standardise_data = true, standardise_parameters = true, kwargs...)
A neural estimator that approximates the posterior distribution $p(\boldsymbol{\theta} \mid \boldsymbol{Z})$, based on a neural `summary_network` and an approximate distribution `q` (see the available in-built [Approximate distributions](@ref)).

The `summary_network` maps data $\boldsymbol{Z}$ to a vector of learned summary statistics $\boldsymbol{t} \in \mathbb{R}^{d^*}$, which are then used to condition the approximate distribution `q`. The precise way in which the summary statistics condition `q` depends on the choice of approximate distribution: for example, [`Gaussian`](@ref) and [`GaussianMixture`](@ref) use an MLP to map $\boldsymbol{t}$ directly to distributional parameters, while [`NormalisingFlow`](@ref) uses $\boldsymbol{t}$ as a conditioning input at each coupling layer. 

The convenience constructor builds `q` internally given `num_parameters` and `num_summaries`, with any additional keyword arguments passed to the constructor of `q`.

The posterior draws can be constrained to the support of the prior distribution with the bounds `lower` and `upper` (see Parameter constraints below).

# Keyword arguments
- `num_summaries::Integer`: the number of summary statistics output by `summary_network`. Must match the output dimension of `summary_network`.
- `q::AbstractApproximateDistribution`: the type of approximate distribution to use.
- `lower = nothing`, `upper = nothing`: lower and upper bounds of the parameters, each given as a vector of length $d$ or as a scalar that applies to all parameters. Bounds may be infinite, and `nothing` leaves the parameters unbounded on that side.
- `transform = identity`, `invtransform = identity`: more generally, a function mapping a $d$ × $K$ matrix of parameters (one parameter vector per column) to an unbounded space, and its inverse. They are set by `lower` and `upper`, so cannot be given with them.
- `standardise_data = true`: whether, or how, to standardise the data before they enter the summary network (see [`Standardiser`](@ref)).
- `standardise_parameters = true`: whether, or how, to standardise the parameters (after `transform`), so that `q` approximates the posterior distribution of the standardised parameters. The default is `false` when `q` is a [`SpikeAndSlab`](@ref), which locates its spike on the original scale.
- `kwargs...`: additional keyword arguments passed to the constructor of `q`.

# Parameter constraints and standardisation
Given `lower` and `upper`, `q` approximates the posterior distribution of the parameters mapped to the real line by [`scaledlogit`](@ref): a scaled logit function if both bounds of a parameter are finite, a log function if only one is finite, and the identity otherwise. That is, `transform = θ -> scaledlogit.(θ, lower, upper)`. Draws from `q` are mapped back by [`scaledlogistic`](@ref), that is, `invtransform = θ -> scaledlogistic.(θ, lower, upper)`, so that they lie within the bounds. More generally, `q` approximates the posterior distribution of `transform(θ)`, and its draws are mapped back by `invtransform`. These functions act on the matrix as a whole, so functions of each element must be broadcast (e.g., `transform = θ -> log.(θ)`, not `transform = log`). The transformed parameters are then standardised (see `standardise_parameters` above).

The transformations are applied outside the neural networks: to the parameters before training, and to the draws after sampling. Hence, the sampler, simulator, true parameters given to [`assess`](@ref), and posterior draws all remain on the original scale. Note also that:
- The training and validation risks are the negative log density of the *transformed and standardised* parameters. This differs from the negative log density of the parameters by a term that does not depend on the neural networks, so these risks cannot be compared with those of an estimator with a different transformation or standardisation.
- The bounds and the standardisers are saved with the estimator (see [`train`](@ref) and [`saveestimator`](@ref)) and restored by [`loadestimator`](@ref). Custom functions `transform` and `invtransform` cannot be saved, so the same functions must be passed when constructing the estimator given to [`loadestimator`](@ref).
- A training parameter on a bound (e.g., a `rand(Float32)` draw equal to zero), or just beyond it through rounding, is mapped to a finite value, while one further beyond a bound throws an error (see [`scaledlogit`](@ref)).
- With `lower` and `upper`, the posterior draws are returned in double precision, so that only extreme draws are rounded onto a bound.
- A [`SpikeAndSlab`](@ref) locates its spike on the original scale, so it cannot be combined with these keyword arguments: to constrain its slab, use the `transform` and `invtransform` keyword arguments of [`SpikeAndSlab`](@ref) itself.

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
estimator = PosteriorEstimator(d, summary_network; num_summaries = num_summaries)

# Or, build q explicitly
q = NormalisingFlow(d; num_summaries = num_summaries)
estimator = PosteriorEstimator(summary_network, q)

# Or, constrain the posterior draws to the prior support, μ ∈ ℝ and σ ∈ (0, 1)
estimator = PosteriorEstimator(d, summary_network; num_summaries = num_summaries, lower = [-Inf, 0], upper = [Inf, 1])

# Training
estimator = train(estimator, sampler, simulator, K = 3000)

# Assess the estimator
θ_test = sampler(250)
Z_test = simulator(θ_test);
assessment = assess(estimator, θ_test, Z_test)

# Inference 
θ = sampler(1)
Z = simulator(θ)
sampleposterior(estimator, Z) # posterior draws
posteriormean(estimator, Z)   # point estimate
```
"""
@concrete struct PosteriorEstimator <: AbstractNeuralEstimator
    summary_network
    q
    transform              # maps the parameters to the scale on which q is defined (identity by default)
    invtransform           # inverse of transform
    data_standardiser      # see Standardiser
    parameter_standardiser # standardises the transformed parameters
end

# As with the sampler of a RatioEstimator, the transformations and standardisers are applied on the host, outside the
# neural networks, so they are kept out of the functor: they are never moved to a device or optimised, and they are saved
# separately (see _transformations), so that the Flux state (and hence the loading of estimators saved previously) is unchanged
@functor PosteriorEstimator (summary_network, q)

# Constructor: summary network and approximate distribution, with optional parameter constraints
# NB the parameters are not standardised by default when q is a SpikeAndSlab, which locates its spike on the original scale
function PosteriorEstimator(summary_network, q; lower::Union{Nothing, Real, AbstractVector} = nothing, upper::Union{Nothing, Real, AbstractVector} = nothing, transform = identity, invtransform = identity, standardise_data = true, standardise_parameters = !(q isa SpikeAndSlab))
    transform, invtransform = _transforms(lower, upper, transform, invtransform)
    if q isa SpikeAndSlab && transform !== identity
        throw(ArgumentError("A SpikeAndSlab locates its spike on the original scale of the parameter, so it cannot be combined with `lower`, `upper`, `transform`, or `invtransform`. To constrain its slab, use the keyword arguments `transform` and `invtransform` of SpikeAndSlab, e.g., `PosteriorEstimator(summary_network, SpikeAndSlab(1, num_summaries; transform, invtransform))`."))
    end
    parameter_standardiser = _standardiser(standardise_parameters)
    if q isa SpikeAndSlab && !isnothing(parameter_standardiser)
        throw(ArgumentError("A SpikeAndSlab locates its spike on the original scale of the parameter, so the parameters cannot be standardised: set `standardise_parameters = false`."))
    end
    return PosteriorEstimator(summary_network, q, transform, invtransform, _standardiser(standardise_data), parameter_standardiser)
end

# Constructor: number of parameters and optional summary network => build approximate distribution automatically
# NB the parameter constraints and standardisation are keyword arguments here, so that they are not passed to the constructor of q
function PosteriorEstimator(num_parameters::Integer, summary_network = identity; num_summaries::Integer, q = NormalisingFlow, lower = nothing, upper = nothing, transform = identity, invtransform = identity, standardise_data = true, standardise_parameters = !(q === SpikeAndSlab || q == "SpikeAndSlab"), kwargs...)
    summary_network = _resolvesummarynetwork(summary_network; kwargs...)
    # Convert string to type if needed
    if q isa String
        q = getfield(@__MODULE__, Symbol(q))
    end
    @info "PosteriorEstimator: num_summaries = $num_summaries, q = $q."
    backend = _backendof(summary_network)
    PosteriorEstimator(summary_network, q(num_parameters, num_summaries; backend = backend, _dropbackend(kwargs)...); lower, upper, transform, invtransform, standardise_data, standardise_parameters)
end

# Constructor: consistent argument ordering
PosteriorEstimator(summary_network, num_parameters::Integer; kwargs...) = PosteriorEstimator(num_parameters, summary_network; kwargs...)

# Constructor: consistent argument ordering
function PosteriorEstimator(q::A, summary_network; kwargs...) where {A <: AbstractApproximateDistribution}
    return PosteriorEstimator(summary_network, q; kwargs...)
end

numdistributionalparams(estimator::PosteriorEstimator) = numdistributionalparams(estimator.q)

# ---- Parameter constraints: saved with the estimator and restored by loadestimator() (see _transformations) ----

_hasbounds(::PosteriorEstimator) = true
_loss(estimator::PosteriorEstimator, loss = nothing) = (logdensity, θ) -> -mean(logdensity)

function _inputoutput(estimator::PosteriorEstimator, Z, θ)
    # Standardised once, on the host (see _Prestandardised); q is defined on the transformed and standardised scale
    Z = _standardisedata(_unmark(estimator.data_standardiser), Z)
    θ = _toqscale(estimator, θ)
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

`θ` should be a ``d \\times K`` matrix of parameter vectors and `Z` a collection of `K` data sets. The parameters are on the scale on which ``q`` is defined, that is, after `estimator.transform` and the parameter standardiser have been applied (see [`PosteriorEstimator`](@ref)), while the data are on the original scale.

# Returns 
A ``1 \\times K`` matrix of log-densities, where entry ``k`` is ``\\log q(\\boldsymbol{\\theta}_k \\mid \\boldsymbol{Z}_k)``.

When using a Lux model, returns a tuple `(log_densities, st)` where `log_densities` is the ``1 \\times K`` matrix and `st` is the updated network state.

See also [`sampleposterior`](@ref).
"""
function _logdensity end

# Forward pass: the original when the data standardiser applies no map (e.g., during training, when the data are
# standardised on the host); otherwise, the data are standardised first (see _NoMap)
const _PosteriorEstimatorNoMap = PosteriorEstimator{<:Any, <:Any, <:Any, <:Any, <:_NoMap}

# Stateful (Flux)
_logdensity(estimator::_PosteriorEstimatorNoMap, θ, Z) = _logdensity(estimator.q, θ, _summarystatistics(estimator, Z))
_logdensity(estimator::PosteriorEstimator, θ, Z) = _logdensity(_withoutstandardisers(estimator), θ, _standardisedata(estimator.data_standardiser, Z))

# Stateless (Lux)
function _logdensity(estimator::_PosteriorEstimatorNoMap, θ, Z, ps, st)
    tz, st_s = _summarystatistics(estimator, Z, ps.summary_network, st.summary_network)
    ld, st_q = _logdensity(estimator.q, θ, tz, ps.q, st.q)
    return ld, (summary_network = st_s, q = st_q)
end
_logdensity(estimator::PosteriorEstimator, θ, Z, ps, st) = _logdensity(_withoutstandardisers(estimator), θ, _standardisedata(estimator.data_standardiser, Z), ps, st)

# Tuple methods used internally during training
(estimator::PosteriorEstimator)(Zθ::Tuple) = _logdensity(estimator, Zθ[2], Zθ[1])
(estimator::PosteriorEstimator)(Zθ::Tuple, ps, st) = _logdensity(estimator, Zθ[2], Zθ[1], ps, st)

# Inference: Stateful (Flux)
function sampleposterior(estimator::PosteriorEstimator, Z; N::Integer = 1000, device = nothing, use_gpu::Bool = true, kwargs...)
    device = _resolvedevice(device = device, use_gpu = use_gpu, verbose = false)
    t = summarystatistics(estimator, Z; device = device, kwargs...)
    θ̃ = sampleposterior(estimator.q, t, N; device = device) # draws from q, a d × N × K array on the CPU
    θ = _fromqscale(estimator, reshape(θ̃, size(θ̃, 1), :))
    return reshape(θ, size(θ, 1), N, size(θ̃, 3))
end

# Inference: Stateless (Lux)
function sampleposterior(estimator::PosteriorEstimator, Z, ps, st; N::Integer = 1000, device = nothing, use_gpu::Bool = true, kwargs...)
    device = _resolvedevice(device = device, use_gpu = use_gpu, verbose = false)
    t = summarystatistics(estimator, Z, ps, st; device = device, kwargs...)
    θ̃ = sampleposterior(estimator.q, t, N, ps.q, st.q; device = device) # draws from q, a d × N × K array on the CPU
    θ = _fromqscale(estimator, reshape(θ̃, size(θ̃, 1), :))
    return reshape(θ, size(θ, 1), N, size(θ̃, 3))
end

@doc raw"""
	spikeprobability(estimator::PosteriorEstimator, Z)
For a [`PosteriorEstimator`](@ref) whose approximate distribution is a [`SpikeAndSlab`](@ref),
returns the estimated posterior probability that the parameter equals the spike value (i.e., the
mixture probability ``\pi``), given data `Z`.

Returns a scalar for a single data set, or a vector of probabilities for a collection of data sets.

See also [`SpikeAndSlab`](@ref) and [`sampleposterior`](@ref).
"""
function spikeprobability end

# Inference: Stateful (Flux)
function spikeprobability(estimator::PosteriorEstimator, Z; device = nothing, use_gpu::Bool = true, kwargs...)
    estimator.q isa SpikeAndSlab || throw(ArgumentError("`spikeprobability` is only defined for a `PosteriorEstimator` with a `SpikeAndSlab` approximate distribution"))
    device = _resolvedevice(device = device, use_gpu = use_gpu, verbose = false)
    t = summarystatistics(estimator, Z; device = device, kwargs...)
    π = spikeprobability(estimator.q, t)
    return size(π, 2) == 1 ? π[1] : vec(π)
end

# Inference: Stateless (Lux)
function spikeprobability(estimator::PosteriorEstimator, Z, ps, st; device = nothing, use_gpu::Bool = true, kwargs...)
    estimator.q isa SpikeAndSlab || throw(ArgumentError("`spikeprobability` is only defined for a `PosteriorEstimator` with a `SpikeAndSlab` approximate distribution"))
    device = _resolvedevice(device = device, use_gpu = use_gpu, verbose = false)
    t = summarystatistics(estimator, Z, ps, st; device = device, kwargs...)
    π = spikeprobability(estimator.q, t, ps.q, st.q)
    return size(π, 2) == 1 ? π[1] : vec(π)
end
