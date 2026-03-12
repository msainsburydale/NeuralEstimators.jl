@doc raw"""
	RatioEstimator <: NeuralEstimator
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
inferential algorithms. For Bayesian inference, posterior samples can be obtained via simple grid-based sampling using [sampleposterior](@ref).

# Keyword arguments
- `num_summaries::Integer`: the number of summaries output by `summary_network`. Must match the output dimension of `summary_network`.
- `num_summaries_θ::Integer = 2 * num_parameters`: the number of summaries output by the parameter summary network.
- `summary_network_θ_kwargs::NamedTuple = (;)`: keyword arguments passed to the MLP constructor for the parameter summary network.
- `kwargs...`: additional keyword arguments passed to the MLP constructor for the inference network.

# Examples
```julia
using NeuralEstimators, Flux, CairoMakie

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
num_parameters = 2     # dimension of the parameter vector θ
n = 1                  # dimension of each independent replicate of Z
m = 30                 # number of independent replicates in each data set
sampler(K) = rand32(num_parameters, K)
simulator(θ, m) = [ϑ[1] .+ ϑ[2] .* randn32(n, m) for ϑ in eachcol(θ)]

# Summary network
num_summaries = 4num_parameters
w = 128   
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, w, relu), Dense(w, num_summaries))
summary_network = DeepSet(ψ, ϕ)

# Initialise the estimator
estimator = RatioEstimator(summary_network, num_parameters; num_summaries = num_summaries)

# Train the estimator
estimator = train(estimator, sampler, simulator, simulator_args = m, K = 1000)

# Plot the risk history
plotrisk()

# Assess the estimator
θ_test = sampler(500)
Z_test = simulator(θ_test, m);
θ_grid = f32(expandgrid(0:0.01:1, 0:0.01:1))'  # fine gridding of the parameter space
assessment = assess(estimator, θ_test, Z_test; θ_grid = θ_grid)
plot(assessment)

# Generate "observed" data 
θ = sampler(1)
z = simulator(θ, 200)

# Grid-based optimization and sampling
logratio(estimator, z, θ_grid = θ_grid)                # log of likelihood-to-evidence ratios
posteriormode(estimator, z; θ_grid = θ_grid)           # posterior mode 
sampleposterior(estimator, z; θ_grid = θ_grid)         # posterior sample

# Gradient-based optimization
using Optim
θ₀ = [0.5, 0.5]                                        # initial estimate
posteriormode(estimator, z; θ₀ = θ₀)                   # posterior mode 
```
"""
struct RatioEstimator{M1, M2, N} <: NeuralEstimator
    summary_network::M1   # summary network for data Z (called summary_network for consistency with other estimators)
    summary_network_θ::M2 # summary network for θ 
    inference_network::N
end

# Constructor: summary network, number of parameters, number of summaries => MLP inference network
function RatioEstimator(
    summary_network, num_parameters::Integer, num_summaries::Integer;
    num_summaries_θ::Integer = 2num_parameters,
    summary_network_θ_kwargs::NamedTuple = (;),
    kwargs...
)
    summary_network_θ = MLP(num_parameters, num_summaries_θ; summary_network_θ_kwargs...)
    inference_network = MLP(num_summaries + num_summaries_θ, 1; kwargs...)
    @info "RatioEstimator: num_summaries = $num_summaries."
    RatioEstimator(summary_network, summary_network_θ, inference_network)
end

# Constructor: keyword num_summaries
RatioEstimator(summary_network, num_parameters::Integer; num_summaries::Integer, kwargs...) = RatioEstimator(summary_network, num_parameters, num_summaries; kwargs...)

# Forward pass: log ratio
function (estimator::RatioEstimator)(Z, θ)
    logr = estimator.inference_network(_summarystatistics(estimator, Z), estimator.summary_network_θ(θ))
    if typeof(logr) <: AbstractVector
        logr = reduce(vcat, logr)
    end
    return logr
end
(estimator::RatioEstimator)(Zθ::Tuple) = estimator(Zθ[1], Zθ[2]) # Tuple method used internally during training

function _inputoutput(estimator::RatioEstimator, Z, θ)

    # Create independent pairs
    K = numobs(Z)
    θ̃ = getobs(θ, shuffle(1:K))
    Z̃ = Z

    # Combine dependent and independent pairs
    θ = hcat(θ, θ̃)
    if Z isa AbstractVector
        Z = vcat(Z, Z̃)
    elseif Z isa AbstractMatrix
        Z = hcat(Z, Z̃)
    else
        Z = getobs(joinobs(Z, Z̃), 1:2K)
    end

    # Create class labels for output
    labels = [:dependent, :independent]
    output = onehotbatch(repeat(labels, inner = K), labels)[1:1, :]

    # Shuffle everything in case batching isn't shuffled properly downstream
    idx = shuffle(1:2K)
    Z = getobs(Z, idx)
    θ = getobs(θ, idx)
    output = output[1:1, idx]

    input = (Z, θ)
    return input, output
end

_loss(estimator::RatioEstimator, loss = nothing) = Flux.logitbinarycrossentropy

"""
    logratio(estimator::RatioEstimator, Z, θ_grid)
    logratio(estimator::RatioEstimator, Z; θ_grid)

Compute the log likelihood-to-evidence ratios over a grid of parameter values `θ_grid` 
for the data `Z`.

# Arguments
- `estimator`: a `RatioEstimator`
- `Z`: observed data
- `θ_grid`: matrix of parameter values, where each column is a parameter configuration

# Returns
A vector of log ratios, one for each column of `θ_grid`.
"""
function logratio(estimator::RatioEstimator, Z, θ_grid_posarg = nothing; θ_grid = nothing)
    @assert !(!isnothing(θ_grid_posarg) && !isnothing(θ_grid)) "θ_grid must be provided exactly once, either as a positional or keyword argument, but not both"
    @assert !(isnothing(θ_grid_posarg) && isnothing(θ_grid)) "θ_grid must be provided either as a positional or keyword argument"
    θ_grid = isnothing(θ_grid_posarg) ? θ_grid : θ_grid_posarg

    summary_stats = summarystatistics(estimator, Z)
    summary_stats_θ = estimator.summary_network_θ(θ_grid)
    _gridlogratio(estimator, summary_stats, summary_stats_θ)
end

function _gridlogratio(estimator::RatioEstimator, summary_stats, summary_stats_θ::AbstractMatrix)
    @assert size(summary_stats, 2) == 1 "gridlogratio currently only supports a single data set"
    summary_stats_rep = repeat(summary_stats, 1, size(summary_stats_θ, 2))
    log_ratios = estimator.inference_network(vcat(summary_stats_rep, summary_stats_θ))
    return log_ratios
end

function sampleposterior(
    estimator::RatioEstimator, Z, N::Integer = 1000;
    logprior::Function = θ -> 0.0f0,
    θ_grid = nothing, theta_grid = nothing,
    kwargs...
)
    @assert isnothing(θ_grid) || isnothing(theta_grid) "Only one of `θ_grid` or `theta_grid` should be given"
    if !isnothing(theta_grid)
        θ_grid = theta_grid
    end
    @assert !isnothing(θ_grid) "θ_grid must be provided for RatioEstimator"
    θ_grid = f32(θ_grid)

    # Log prior over the grid
    logpθ = logprior.(eachcol(θ_grid))

    # θ embeddings over the grid
    summary_stats_θ = estimator.summary_network_θ(θ_grid)

    # Summary statistics for each data set 
    summary_stats = summarystatistics(estimator, Z; kwargs...)

    # For each data set, pass summary stats and θ embeddings through the inference net, then sample
    samples = map(eachcol(summary_stats)) do s
        logrZθ = vec(_gridlogratio(estimator, s, summary_stats_θ))
        weights = exp.(logpθ .+ logrZθ)
        samples = StatsBase.wsample(eachcol(θ_grid), weights, N; replace = true)
        reduce(hcat, samples)
    end

    if length(samples) == 1
        samples = samples[1]
    end

    return samples
end

function posteriormode(
    estimator::RatioEstimator, Z;
    logprior::Function = θ -> 0.0f0, penalty::Union{Function, Nothing} = nothing,
    θ_grid = nothing, theta_grid = nothing,
    θ₀ = nothing, theta0 = nothing,
    kwargs...
)

    # Check duplicated arguments that are needed so that the R interface uses ASCII characters only
    @assert isnothing(θ_grid) || isnothing(theta_grid) "Only one of `θ_grid` or `theta_grid` should be given"
    @assert isnothing(θ₀) || isnothing(theta0) "Only one of `θ₀` or `theta0` should be given"
    if !isnothing(theta_grid)
        θ_grid = theta_grid
    end
    if !isnothing(theta0)
        θ₀ = theta0
    end

    # Change "penalty" to "prior"
    if !isnothing(penalty)
        logprior = penalty
    end

    # Check that we have either a grid to search over or initial estimates
    @assert !isnothing(θ_grid) || !isnothing(θ₀) "One of `θ_grid` or `θ₀` should be given"
    @assert isnothing(θ_grid) || isnothing(θ₀) "Only one of `θ_grid` and `θ₀` should be given"

    if !isnothing(θ_grid)
        logpθ = logprior.(eachcol(θ_grid))
        summary_stats = summarystatistics(estimator, Z)
        summary_stats_θ = estimator.summary_network_θ(θ_grid)

        modes = map(eachcol(summary_stats)) do s
            logrZθ = vec(_gridlogratio(estimator, s, summary_stats_θ))
            logdensity = logpθ .+ logrZθ
            θ_grid[:, argmax(logdensity)]
        end

        return reduce(hcat, modes)

    else
        return _optimdensity(θ₀, logprior, estimator) #TODO doesn't work for multiple data sets; _optimdensity needs to take Z as input, I guess. Also, can be done more efficiently by computing the summary statistics. Try it out, could be interesting!
    end
end
