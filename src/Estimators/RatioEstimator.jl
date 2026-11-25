@doc raw"""
	RatioEstimator <: AbstractNeuralEstimator
	RatioEstimator(num_parameters, summary_network = identity; num_summaries, kwargs...)
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
During training, each data set is paired both with its own parameters and with independent parameters (drawn with
`sampler`, or otherwise obtained by permuting the training parameters), so that its summary statistics are computed once
for both pairs; hence, the `batchsize` of [`train`](@ref) is the number of data sets, each giving two pairs.

Given data `Z` and parameters `θ`, the estimated ratio can be obtained using [`logratio`](@ref) 
and can be used in various Bayesian
(e.g., [Hermans et al., 2020](https://proceedings.mlr.press/v119/hermans20a.html))
or frequentist
(e.g., [Walchessen et al., 2024](https://doi.org/10.1016/j.spasta.2024.100848))
inferential algorithms. For Bayesian inference, posterior samples can be obtained with [`sampleposterior`](@ref) using the NUTS algorithm or by grid-based sampling.

# Keyword arguments
- `num_summaries::Integer`: the number of summaries output by `summary_network`. Must match the output dimension of `summary_network`.
- `num_summaries_θ::Integer = 2 * num_parameters`: the number of summaries output by the parameter summary network.
- `summary_network_θ_kwargs::NamedTuple = (;)`: keyword arguments passed to the MLP constructor for the parameter summary network.
- `sampler::Union{Nothing, Function} = nothing`: a function returnting `K` independent draws from the prior as a `d x K` matrix, the same as passed to [`train`](@ref). This can also be ignored, i.e., `nothing`, for backwards compatibility, see below. 
- `standardise_data = true`: whether, or how, to standardise the data before they enter the summary network (see [`Standardiser`](@ref)).
- `standardise_parameters = true`: whether, or how, to standardise the parameters before they enter the parameter summary network. Neither form of standardisation changes the likelihood-to-evidence ratio being estimated.
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
estimator = RatioEstimator(d, summary_network; num_summaries = num_summaries, sampler = sampler)

# Train the estimator
estimator = train(estimator, sampler, simulator, K = 1000)

# Plot the risk history
plotrisk()

# Generate "observed" data 
θ = sampler(1)
z = simulator(θ)

# Grid-based sampling, over a grid that covers the support of the prior
grid = expandgrid(0:0.01:1, 0:0.01:1)'
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
    data_standardiser      # see Standardiser
    parameter_standardiser # standardises θ before the parameter summary network
end

# Constructor: networks and (optionally, for backwards compatibility) the sampler
# there is no clash with the constructor below, where the Integer type specification differs
function RatioEstimator(summary_network, summary_network_θ, inference_network, sampler = nothing; standardise_data = true, standardise_parameters = true)
    RatioEstimator(summary_network, summary_network_θ, inference_network, sampler, _standardiser(standardise_data), _standardiser(standardise_parameters))
end

# The sampler is intentionally kept out of the network
# e.g., to prevent silent transformations from cpu to gpu before _inputoutput invokes it
# e.g., to not make the optimizer compute gradients w.r.t. sampler
# The standardisers are likewise applied outside the networks and saved separately (see _transformations)
@functor RatioEstimator (summary_network, summary_network_θ, inference_network)

# Constructor: number of parameters and optional summary network => MLP inference network
function RatioEstimator(
    num_parameters::Integer, summary_network = identity;
    num_summaries::Integer,
    num_summaries_θ::Integer = 2num_parameters,
    summary_network_θ_kwargs::NamedTuple = (;),
    sampler::Union{Nothing, Function} = nothing,
    standardise_data = true,
    standardise_parameters = true,
    kwargs...
)
    summary_network = _resolvesummarynetwork(summary_network; kwargs...)
    backend = _backendof(summary_network)
    nt = _dropbackend(kwargs)
    summary_network_θ = MLP(num_parameters, num_summaries_θ; backend = backend, output_activation = identity, summary_network_θ_kwargs...)
    inference_network = MLP(num_summaries + num_summaries_θ, 1; backend = backend, output_activation = identity, nt...)
    summary_network_θ = _matchsimplechains(summary_network_θ, num_parameters, summary_network)
    inference_network = _matchsimplechains(inference_network, num_summaries + num_summaries_θ, summary_network, summary_network_θ)
    @info "RatioEstimator: num_summaries = $num_summaries."
    RatioEstimator(summary_network, summary_network_θ, inference_network, sampler; standardise_data, standardise_parameters)
end

# Constructor: consistent argument ordering
RatioEstimator(summary_network, num_parameters::Integer; kwargs...) = RatioEstimator(num_parameters, summary_network; kwargs...)

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
    # we mislabel at least one dependent pair as independent.
    # Further, fresh theta draws lower the variance of the loss function estimators. 

    K = numobs(Z)
    θ̃ = if isnothing(estimator.sampler)
        getobs(θ, shuffle(1:K))
    else
        # this is the same sampler as the one passed to train(), hence it
        # might return a parameter set with names and any float type;
        # strip names below, as the data loader will handle the conversion to Float32
        θ̃ = _stripnames(_extractθ(estimator.sampler(K)))
        @assert size(θ̃) == size(θ) "sampler must generate parameters of the same dimensions as theta; expected $(size(θ)) got $(size(θ̃))"
        θ̃
    end

    # Each data set is paired with its own parameters (a dependent pair, label 1) and with the independent parameters θ̃
    # (an independent pair, label 0), and the training forward pass computes its summary statistics once for both pairs.
    # The labels of the two pairs are stacked along rows, so that every component of the input has K observations, as
    # required by the data loader, which keeps each (Z, θ, θ̃) together when shuffled
    output = vcat(ones(Float32, 1, K), zeros(Float32, 1, K))

    # Standardised once, on the host (see _Prestandardised)
    sθ = _unmark(estimator.parameter_standardiser)
    input = (_standardisedata(_unmark(estimator.data_standardiser), Z), _standardise(sθ, θ), _standardise(sθ, θ̃))
    return input, output
end

_loss(estimator::RatioEstimator, loss = nothing) = logitbinarycrossentropy

# Forward pass: the original when the standardisers apply no map (e.g., during training, when the data and parameters are
# standardised on the host); otherwise, the data and parameters are standardised first (see _NoMap)
const _RatioEstimatorNoMap = RatioEstimator{<:Any, <:Any, <:Any, <:Any, <:_NoMap, <:_NoMap}

# Forward pass: Stateful (Flux)
function (estimator::_RatioEstimatorNoMap)(Z, θ)
    tz = _summarystatistics(estimator, Z)
    tθ = estimator.summary_network_θ(θ)
    estimator.inference_network(vcat(tz, tθ))
end
(estimator::RatioEstimator)(Z, θ) = _withoutstandardisers(estimator)(_standardisedata(estimator.data_standardiser, Z), _standardise(estimator.parameter_standardiser, θ))

# Forward pass: Stateless (Lux)
function (e::_RatioEstimatorNoMap)(Z, θ, ps, st)
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
(e::RatioEstimator)(Z, θ, ps, st) = _withoutstandardisers(e)(_standardisedata(e.data_standardiser, Z), _standardise(e.parameter_standardiser, θ), ps, st)

# Training forward pass: the 2 × K logits of the dependent pairs (Z, θ) and of the independent pairs (Z, θ̃), matching the
# class labels constructed in _inputoutput. The summary statistics of each data set are computed once for both pairs, and
# the other networks are applied to all 2K pairs at once
function (estimator::_RatioEstimatorNoMap)(Z, θ, θ̃)
    tz = _summarystatistics(estimator, Z)
    tθ = estimator.summary_network_θ(hcat(θ, θ̃))
    logr = estimator.inference_network(vcat(hcat(tz, tz), tθ))
    K = size(tz, 2)
    vcat(logr[:, 1:K], logr[:, (K + 1):(2K)])
end
function (e::_RatioEstimatorNoMap)(Z, θ, θ̃, ps, st)
    tz, st_s = _summarystatistics(e, Z, ps.summary_network, st.summary_network)
    tz = tz |> copy # NB see the note on Enzyme above
    tθ, st_sθ = e.summary_network_θ(hcat(θ, θ̃), ps.summary_network_θ, st.summary_network_θ)
    logr, st_i = e.inference_network(vcat(hcat(tz, tz), tθ), ps.inference_network, st.inference_network)
    K = size(tz, 2)
    return vcat(logr[:, 1:K], logr[:, (K + 1):(2K)]), (summary_network = st_s, summary_network_θ = st_sθ, inference_network = st_i)
end
(e::RatioEstimator)(Z, θ, θ̃) = _withoutstandardisers(e)(_standardisedata(e.data_standardiser, Z), _standardise(e.parameter_standardiser, θ), _standardise(e.parameter_standardiser, θ̃))
(e::RatioEstimator)(Z, θ, θ̃, ps, st) = _withoutstandardisers(e)(_standardisedata(e.data_standardiser, Z), _standardise(e.parameter_standardiser, θ), _standardise(e.parameter_standardiser, θ̃), ps, st)

# Tuple methods used internally during training (the input constructed by _inputoutput)
(estimator::RatioEstimator)(input::Tuple) = estimator(input...)
(estimator::RatioEstimator)(input::Tuple, ps, st) = estimator(input..., ps, st)
(estimator::_RatioEstimatorNoMap)(input::Tuple, ps, st) = estimator(input..., ps, st) # resolves the ambiguity with the training forward pass

# ---- Inference: Stateful (Flux) ----

"""
    logratio(estimator::Union{RatioEstimator, TelescopingRatioEstimator}, Z; grid)
Compute the log likelihood-to-evidence ratio for each parameter configuration in `grid`.

# Arguments
- `Z`: observed data
- `grid`: matrix of parameter values, where each column is a parameter configuration

# Returns
A matrix of log ratios with one row per data set and one column per grid point.
"""
function logratio(estimator::RatioEstimator, Z; grid, kwargs...)
    grid = f32(grid)
    summary_stats_Z = summarystatistics(estimator, Z; kwargs...)
    summary_stats_θ = estimator.summary_network_θ(_standardise(estimator.parameter_standardiser, grid))
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

function sampleposterior(
    estimator::RatioEstimator, Z;
    grid = nothing,
    N::Integer = 1000,
    logprior::Function = θ -> 0.0f0,
    lower::Union{Nothing, AbstractVector} = nothing,
    upper::Union{Nothing, AbstractVector} = nothing,
    warmup::Integer = 750, # Stan uses between 500 and 1000; acceptance rate is 0.8 by default 
    kwargs...
)
    has_grid = !isnothing(grid)
    has_box = !isnothing(lower) && !isnothing(upper)
    @assert xor(has_grid, has_box) "supply either `grid` (grid sampling) or both `lower` and `upper` (HMC sampling), but not both"
    if has_box
        @assert length(lower) == length(upper) && all(lower .< upper) "lower bounds must be strictly below upper bounds"
        isempty(methods(_sampleposterior_hmc)) &&
            error("HMC sampling requires the AdvancedHMC extension: run `using AdvancedHMC, ForwardDiff, LogDensityProblems` and retry")
        return _sampleposterior_hmc(estimator, Z; N = N, lower = lower, upper = upper, logprior = logprior, warmup = warmup, kwargs...)
    end
    grid = f32(grid)

    summary_stats = summarystatistics(estimator, Z; kwargs...)
    logpθ = logprior.(eachcol(grid))
    summary_stats_θ = estimator.summary_network_θ(_standardise(estimator.parameter_standardiser, grid))

    log_ratios = _gridlogratio(estimator, summary_stats, summary_stats_θ)

    samples = map(1:size(log_ratios, 1)) do k
        weights = exp.(logpθ .+ log_ratios[k, :])
        reduce(hcat, StatsBase.wsample(eachcol(grid), weights, N; replace = true))
    end

    return stack(samples)
end

# Implemented by the AdvancedHMC extension (ext/NeuralEstimatorsAdvancedHMCExt.jl); no
# methods exist unless AdvancedHMC, ForwardDiff and LogDensityProblems are loaded, which
# the HMC branch above checks for with a readable error.
function _sampleposterior_hmc end

# ---- Inference: Stateless (Lux) ----

function logratio(estimator::RatioEstimator, Z, ps, st; grid, kwargs...)
    grid = f32(grid)
    summary_stats_Z = summarystatistics(estimator, Z, ps, st; kwargs...)
    summary_stats_θ = first(estimator.summary_network_θ(_standardise(estimator.parameter_standardiser, grid), ps.summary_network_θ, st.summary_network_θ))
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

function sampleposterior(
    estimator::RatioEstimator, Z, ps, st;
    grid = nothing,
    N::Integer = 1000,
    logprior::Function = θ -> 0.0f0,
    lower::Union{Nothing, AbstractVector} = nothing,
    upper::Union{Nothing, AbstractVector} = nothing,
    warmup::Integer = 750,
    kwargs...
)
    has_grid = !isnothing(grid)
    has_box = !isnothing(lower) && !isnothing(upper)
    @assert xor(has_grid, has_box) "supply either `grid` (grid sampling) or both `lower` and `upper` (HMC sampling), but not both"
    if has_box
        @assert length(lower) == length(upper) && all(lower .< upper) "lower bounds must be strictly below upper bounds"
        isempty(methods(_sampleposterior_hmc)) &&
            error("HMC sampling requires the AdvancedHMC extension: run `using AdvancedHMC, ForwardDiff, LogDensityProblems` and retry")
        return _sampleposterior_hmc(estimator, Z, ps, st; N = N, lower = lower, upper = upper, logprior = logprior, warmup = warmup, kwargs...)
    end
    grid = f32(grid)

    summary_stats_Z = summarystatistics(estimator, Z, ps, st; kwargs...)
    summary_stats_θ = first(estimator.summary_network_θ(_standardise(estimator.parameter_standardiser, grid), ps.summary_network_θ, st.summary_network_θ))
    logpθ = logprior.(eachcol(grid))

    log_ratios = _gridlogratio(estimator, summary_stats_Z, summary_stats_θ, ps.inference_network, st.inference_network)

    samples = map(1:size(log_ratios, 1)) do k
        weights = exp.(logpθ .+ log_ratios[k, :])
        reduce(hcat, StatsBase.wsample(eachcol(grid), weights, N; replace = true))
    end

    return stack(samples)
end
