@doc raw"""
	RatioEstimator <: AbstractNeuralEstimator
	RatioEstimator(summary_network, num_parameters; num_summaries, kwargs...)
A neural estimator that estimates the likelihood-to-evidence ratio,
```math
r(\boldsymbol{Z}, \boldsymbol{\theta}) \equiv p(\boldsymbol{Z} \mid \boldsymbol{\theta})/p(\boldsymbol{Z}),
```
where $p(\boldsymbol{Z} \mid \boldsymbol{\theta})$ is the likelihood and $p(\boldsymbol{Z})$
is the marginal likelihood, also known as the model evidence.

The estimator jointly summarises the data $\boldsymbol{Z}$ and parameters $\boldsymbol{\theta}$ 
using separate summary networks, whose outputs are concatenated and passed to an MLP inference network. 
The parameter summary network maps $\boldsymbol{\theta}$ to a vector of `2 * num_parameters` summaries by default.

For numerical stability, training is done on the log-scale using the relation 
$\log r(\boldsymbol{Z}, \boldsymbol{\theta}) = \text{logit}(c^*(\boldsymbol{Z}, \boldsymbol{\theta}))$, 
where $c^*(\cdot, \cdot)$ denotes the Bayes classifier as described in the [methodology](@ref "Neural ratio estimators") section. 

Given data `Z` and parameters `θ`, the estimated ratio can be obtained using [logratio](@ref) 
and can be used in various Bayesian
(e.g., [Hermans et al., 2020](https://proceedings.mlr.press/v119/hermans20a.html))
or frequentist
(e.g., [Walchessen et al., 2024](https://doi.org/10.1016/j.spasta.2024.100848))
inferential algorithms. For Bayesian inference, posterior samples can be obtained with [sampleposterior](@ref) using the NUTS algorithm or by grid-based sampling.

# Keyword arguments
- `num_summaries::Integer`: the number of summaries output by `summary_network`. Must match the output dimension of `summary_network`.
- `num_summaries_θ::Integer = 2 * num_parameters`: the number of summaries output by the parameter summary network.
- `summary_network_θ_kwargs::NamedTuple = (;)`: keyword arguments passed to the MLP constructor for the parameter summary network.
- `sampler::Union{Nothing, Function} = nothing`: a function returnting `K` independent draws from the prior as a `d x K` matrix, the same as passed to [`train`](@ref). This can also be ignored, i.e., `nothing`, for backwards compatibility, see below. 
- `kwargs...`: additional keyword arguments passed to the MLP constructor for the inference network.

# Examples
```julia
using NeuralEstimators, Flux, CairoMakie

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d, m = 2, 100  # dimension of θ and number of replicates
sampler(K) = NamedMatrix(μ = rand(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(m))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

# Neural network
num_summaries = 3d
summary_network = Chain(Dense(m, 64, gelu), Dense(64, 64, gelu), Dense(64, num_summaries))

# Initialise the estimator

estimator = RatioEstimator(summary_network, d; num_summaries = num_summaries, sampler = sampler)
# backwards compatibility version
# estimator = RatioEstimator(summary_network, d; num_summaries = num_summaries)

# Train the estimator
estimator = train(estimator, sampler, simulator, K = 1000)

# Plot the risk history
plotrisk()

# Assess the estimator
θ_test = sampler(250)
Z_test = simulator(θ_test);
grid = expandgrid(0:0.01:1, 0:0.01:1)'  # fine gridding of the parameter space


### to discuss with Matt

assessment = assess(estimator, θ_test, Z_test; grid = grid)
plot(assessment)

# Generate "observed" data 
θ = sampler(1)
z = simulator(θ)

# Grid-based sampling
logratio(estimator, z; grid = grid)                # log of likelihood-to-evidence ratios
sampleposterior(estimator, z; grid = grid)         # posterior sampling

# NUTS sampling (requires `using AdvancedHMC, ForwardDiff, LogDensityProblems`)
sampleposterior(estimator, z; lower = [0.0, 0.0], upper = [1.0, 1.0]) # requires passing the parameter bounds
```
"""
@concrete struct RatioEstimator <: AbstractNeuralEstimator
    summary_network   # summary network for data Z (called summary_network for consistency with other estimators)
    summary_network_θ # summary network for θ 
    inference_network
    sampler     # same sampler that generates theta, or nothing, for backwards compatibility
end

# Backwards compatibility to allow for a sampler to not be passed to the RatioEstimator class
# there is no clash with the constructor below, where the Integer type specification differs
RatioEstimator(summary_network, summary_network_θ, inference_network) = 
    RatioEstimator(summary_network, summary_network_θ, inference_network, nothing)

# The sampler is intentionally kept out of the network
# e.g., to prevent silent transformations from cpu to gpu before _inputoutput invokes it
# e.g., to not make the optimizer compute gradients w.r.t. sampler
@functor RatioEstimator (summary_network, summary_network_θ, inference_network)

# Constructor: summary network, number of parameters, number of summaries => MLP inference network
function RatioEstimator(
    summary_network, num_parameters::Integer, num_summaries::Integer;
    num_summaries_θ::Integer = 2num_parameters,
    summary_network_θ_kwargs::NamedTuple = (;),
    sampler::Union{Nothing,Function} = nothing,
    kwargs...
)
    backend = _backendof(summary_network)
    summary_network_θ = MLP(num_parameters, num_summaries_θ; backend = backend, output_activation = identity, summary_network_θ_kwargs...)
    inference_network = MLP(num_summaries + num_summaries_θ, 1; backend = backend, output_activation = identity, kwargs...)
    @info "RatioEstimator: num_summaries = $num_summaries."
    RatioEstimator(summary_network, summary_network_θ, inference_network, sampler)
end

# Constructor: keyword num_summaries
RatioEstimator(summary_network, num_parameters::Integer; num_summaries::Integer, kwargs...) = RatioEstimator(summary_network, num_parameters, num_summaries; kwargs...)

function _mergedata(Z, Z̃)
    if Z isa AbstractVector
        vcat(Z, Z̃)
    elseif Z isa AbstractMatrix || Z isa Summaries
        hcat(Z, Z̃)
    else
        getobs(joinobs(Z, Z̃), 1:2numobs(Z)) #TODO this doesn't work as intended... just want to combine along the batch dimension for an arbitrary data format
    end
end

function _inputoutput(estimator::RatioEstimator, Z, θ)

    # Create independent pairs
    # Ideally, these should use fresh (prior) theta draws when a sampler is available,
    # otherwise  permutation of the already sampled parameters is used; in this case, 
    # with probability 1 - 1/e ~= 0.63  https://math.stackexchange.com/questions/399500/why-is-the-derangement-probability-so-close-to-frac1e
    # we mislabel at least one dependent pair as indepndent.
    # Further, fresh theta draws lower the variance of the loss function estimators. 

    K = numobs(Z)
    θ̃ = if isnothing(estimator.sampler)
        getobs(θ, shuffle(1:K))
    else
        # this is the same sampler as the one passed to train(), hence it 
        # might return a parameter set with names and any float type;
        # strip names below, as the data loader will handle the conversion to Float32
        θ̃  =_stripnames(_extractθ(estimator.sampler(K)))
        @assert size(θ̃ ) == size(θ) "sampler must generate parameters of the same dimensions as theta; 
                                        expected $(size(θ)) got $(size(θ̃))"
        θ̃ 
    end
    Z̃ = Z

    # Combine dependent and independent pairs
    θ = hcat(θ, θ̃)
    Z = _mergedata(Z, Z̃)

    # Binary class labels: 1 for dependent pairs (Z, θ), 0 for independent pairs (Z̃, θ̃)
    output = [ones(Float32, 1, K) zeros(Float32, 1, K)]

    # Shuffle everything just in case batching isn't shuffled properly downstream
    idx = shuffle(1:2K)
    Z = getobs(Z, idx)
    θ = getobs(θ, idx)
    output = output[:, idx]

    input = (Z, θ)
    return input, output
end

_loss(estimator::RatioEstimator, loss = nothing) = logitbinarycrossentropy

# Forward pass: Stateful (Flux)
function (estimator::RatioEstimator)(Z, θ)
    tz = _summarystatistics(estimator, Z)
    tθ = estimator.summary_network_θ(θ)
    estimator.inference_network(vcat(tz, tθ))
end

# Forward pass: Stateless (Lux)
function (e::RatioEstimator)(Z, θ, ps, st)
    tz, st_s = _summarystatistics(e, Z, ps.summary_network, st.summary_network)
    # NB: Enzyme's static analyis doesn't like it when we extract data 
    #     from a custom struct (e.g., Summaries or DataAndSummaries) and then 
    #     concatenate it with something else later. 
    #     So, here we materialise (copy) to break Enzyme's trace (needed when freeze_summary_network = true and the summaries are stored in the field S of a Summaries object)
    tz = tz |> copy

    tθ, st_sθ = e.summary_network_θ(θ, ps.summary_network_θ, st.summary_network_θ)
    logr, st_i = e.inference_network(vcat(tz, tθ), ps.inference_network, st.inference_network)
    return logr, (summary_network = st_s, summary_network_θ = st_sθ, inference_network = st_i)
end

# Tuple methods used internally during training
(estimator::RatioEstimator)(Zθ::Tuple) = estimator(Zθ[1], Zθ[2])
(estimator::RatioEstimator)(Zθ::Tuple, ps, st) = estimator(Zθ[1], Zθ[2], ps, st)

# ---- Inference: Stateful (Flux) ----

"""
    logratio(estimator::RatioEstimator, Z; grid)
Compute the log likelihood-to-evidence ratio for each parameter configuration in `grid`.

# Arguments
- `estimator`: a `RatioEstimator`
- `Z`: observed data
- `grid`: matrix of parameter values, where each column is a parameter configuration

# Returns
A matrix of log ratios with one row per data set and one column per grid point.
"""
function logratio(estimator::RatioEstimator, Z; grid, kwargs...)
    grid = f32(grid)
    summary_stats_Z = summarystatistics(estimator, Z; kwargs...)
    summary_stats_θ = estimator.summary_network_θ(grid)
    _gridlogratio(estimator, summary_stats_Z, summary_stats_θ)
end

function _gridlogratio(estimator::RatioEstimator, summary_stats_Z, summary_stats_θ::AbstractMatrix)
    K = size(summary_stats_Z, 2)    # number of data sets
    G = size(summary_stats_θ, 2)    # number of grid points
    # Repeat so that both sets of summary stats have GxK columns
    summary_stats_Z_rep = repeat(summary_stats_Z, inner = (1, G))
    summary_stats_θ_rep = repeat(summary_stats_θ, outer = (1, K))
    log_ratios = estimator.inference_network(vcat(summary_stats_Z_rep, summary_stats_θ_rep))
    return permutedims(reshape(log_ratios, G, K))  # K x G matrix
end

@doc raw"""
	sampleposterior(estimator::RatioEstimator, Z; method = :auto, kwargs...)
Draw `N` posterior samples for each data set in `Z`, by one of two methods:

- `method = :hmc` (recommended): NUTS (via the AdvancedHMC extension) on the continuous
  pdf that is proportional to `exp(logratio(θ) + logprior(θ))` on the box `[lower, upper]`, one
  chain per data set with `warmup` adaptation steps discarded. Off-grid, scales to larger
  `d`, and is the default . Caveat: any MCMC algorithm may struggle with multi-modality / challenging geometries of the posterior.
  Requires `using AdvancedHMC, ForwardDiff, LogDensityProblems`.
- `method = :grid`: self-normalised sampling on a fixed `grid` (a `d x G` matrix, one
  candidate parameter configuration per column): grid atoms are drawn with replacement,
  with weights proporional to `exp.(logprior + logratio)`. Samples live on the grid, so resolution is
  limited by `G` and the grid size is cursed in `d, but every mode on the grid is seen.

`method = :auto` (the default) resolves to `:grid` when a `grid` is supplied and to
`:hmc` otherwise, so existing `grid`-based calls keep their previous behaviour verbatim
while `lower`/`upper` calls get HMC.

# Keyword arguments
- `N::Integer = 1000`: number of posterior samples per data set.
- `logprior::Function = θ -> 0f0`: log prior density evaluated on a `d`-vector, up to a constant; the default is uniform. Used by both methods. For `:hmc` it must be differentiable in the ForwardDiff sense (plain arithmetic is fine).
- `grid`: required for `:grid`.
- `lower`, `upper`: prior box bounds, required for `:hmc`. The chain runs in unconstrained space through a sigmoid map onto the box, with the Jacobian accounted for.
- `warmup::Integer = 750`: adaptation steps (`:hmc` only), discarded from the output; Stan uses between 500 and 1000.

# Returns
A `d x N` matrix for a single data set, or a vector of such matrices.
"""
function sampleposterior(
    estimator::RatioEstimator, Z;
    method::Symbol = :auto,
    grid = nothing,
    N::Integer = 1000,
    logprior::Function = θ -> 0.0f0,
    lower::Union{Nothing, AbstractVector} = nothing,
    upper::Union{Nothing, AbstractVector} = nothing,
    warmup::Integer = 750, # Stan uses between 500 and 1000; acceptance rate is 0.8 by default 
    kwargs...
)
    @assert method in (:auto, :grid, :hmc) "method must be :auto, :grid or :hmc"
    if method === :auto
        # if the user supplies the grid, we keep it that way 
        # and importantly pre-existing `sampleposterior(estimator, Z; grid = ...)` call behaves as
        # before, for backwards-compatibility. Otherwise default to HMC, which is the the recommended method.
        @assert !isnothing(grid) || (!isnothing(lower) && !isnothing(upper)) "supply either `grid` (grid sampling) or `lower` and `upper` (HMC sampling)"
        method = isnothing(grid) ? :hmc : :grid
    end
    if method === :hmc
        @assert !isnothing(lower) && !isnothing(upper) "method = :hmc requires the prior box bounds `lower` and `upper`"
        @assert length(lower) == length(upper) && all(lower .< upper) "lower bounds must be strictly below upper bounds"
        isempty(methods(_sampleposterior_hmc)) &&
            error("method = :hmc requires the AdvancedHMC extension: run `using AdvancedHMC, ForwardDiff, LogDensityProblems` and retry")
        return _sampleposterior_hmc(estimator, Z; N = N, lower = lower, upper = upper, logprior = logprior, warmup = warmup, kwargs...)
    end
    @assert !isnothing(grid) "method = :grid requires a `grid` of candidate parameter values"
    grid = f32(grid)

    summary_stats = summarystatistics(estimator, Z; kwargs...)
    logpθ = logprior.(eachcol(grid))
    summary_stats_θ = estimator.summary_network_θ(grid)

    log_ratios = _gridlogratio(estimator, summary_stats, summary_stats_θ)

    samples = map(1:size(log_ratios, 1)) do k
        weights = exp.(logpθ .+ log_ratios[k, :])
        reduce(hcat, StatsBase.wsample(eachcol(grid), weights, N; replace = true))
    end

    return length(samples) == 1 ? samples[1] : samples
end

# Implemented by the AdvancedHMC extension (ext/NeuralEstimatorsAdvancedHMCExt.jl); no
# methods exist unless AdvancedHMC, ForwardDiff and LogDensityProblems are loaded, which
# the :hmc branch above checks for with a readable error.
function _sampleposterior_hmc end

# ---- Inference: Stateless (Lux) ----

function logratio(estimator::RatioEstimator, Z, ps, st; grid, kwargs...)
    grid = f32(grid)
    summary_stats_Z = summarystatistics(estimator, Z, ps, st; kwargs...)
    summary_stats_θ = first(estimator.summary_network_θ(grid, ps.summary_network_θ, st.summary_network_θ))
    _gridlogratio(estimator, summary_stats_Z, summary_stats_θ, ps.inference_network, st.inference_network)
end

function _gridlogratio(estimator::RatioEstimator, summary_stats_Z, summary_stats_θ::AbstractMatrix, ps_inference, st_inference)
    K = size(summary_stats_Z, 2)
    G = size(summary_stats_θ, 2)
    summary_stats_Z_rep = repeat(summary_stats_Z, inner = (1, G))
    summary_stats_θ_rep = repeat(summary_stats_θ, outer = (1, K))
    log_ratios = first(estimator.inference_network(vcat(summary_stats_Z_rep, summary_stats_θ_rep), ps_inference, st_inference))
    return permutedims(reshape(log_ratios, G, K))  # K x G matrix
end

function sampleposterior(estimator::RatioEstimator, Z, ps, st;
    grid,
    N::Integer = 1000,
    logprior::Function = θ -> 0.0f0,
    kwargs...
)
    grid = f32(grid)

    summary_stats_Z = summarystatistics(estimator, Z, ps, st; kwargs...)
    summary_stats_θ = first(estimator.summary_network_θ(grid, ps.summary_network_θ, st.summary_network_θ))
    logpθ = logprior.(eachcol(grid))

    log_ratios = _gridlogratio(estimator, summary_stats_Z, summary_stats_θ, ps.inference_network, st.inference_network)

    samples = map(1:size(log_ratios, 1)) do k
        weights = exp.(logpθ .+ log_ratios[k, :])
        reduce(hcat, StatsBase.wsample(eachcol(grid), weights, N; replace = true))
    end

    return length(samples) == 1 ? samples[1] : samples
end
