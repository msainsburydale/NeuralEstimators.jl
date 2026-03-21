@doc raw"""
	IntervalEstimator <: BayesEstimator
	IntervalEstimator(summary_network, num_parameters; num_summaries, kwargs...)
	IntervalEstimator(summary_network, num_parameters, num_summaries; kwargs...)
A neural estimator that jointly estimates marginal posterior credible intervals based on the probability levels `probs` (by default, 95% central credible intervals).

The estimator summarises the data ``\boldsymbol{Z}`` using a `summary_network` whose output is passed to two MLP inference networks, ``\boldsymbol{u}(\cdot)`` and ``\boldsymbol{v}(\cdot)``, each mapping from the summary space to ``\mathbb{R}^d``.

The estimator employs a representation that prevents quantile crossing. Specifically, given data ``\boldsymbol{Z}``, 
it constructs intervals for each parameter
``\theta_i``, ``i = 1, \dots, d,``  of the form,
```math
[c_i(u_i(\boldsymbol{Z})), \;\; c_i(u_i(\boldsymbol{Z})) + g(v_i(\boldsymbol{Z})))],
```
where  ``\boldsymbol{u}(⋅) \equiv (u_1(\cdot), \dots, u_d(\cdot))'`` and
``\boldsymbol{v}(⋅) \equiv (v_1(\cdot), \dots, v_d(\cdot))'`` are neural networks
that map from the sample space to ``\mathbb{R}^d``; $g(\cdot)$ is a
monotonically increasing function (e.g., exponential or softplus); and each
``c_i(⋅)`` is a monotonically increasing function that maps its input to the
prior support of ``\theta_i``.

The functions ``c_i(⋅)`` may be collectively defined by a ``d``-dimensional [`Compress`](@ref) object, which can constrain the interval estimator's output to the prior support. If these functions are unspecified, they will be set to the identity function so that the range of the intervals will be unrestricted.

The return value when applied to data using [`estimate`()](@ref) is a matrix with ``2d`` rows, where the first and second ``d`` rows correspond to the lower and upper bounds, respectively. The function [`interval()`](@ref) can be used to format this output in a readable ``d`` × 2 matrix.  

See also [`QuantileEstimator`](@ref).

# Keyword arguments
- `num_summaries::Integer`: the number of summaries output by `summary_network`. Must match the output dimension of `summary_network`.
- `c::Union{Function, Compress} = identity`: monotonically increasing function(s) mapping to the prior support of each parameter.
- `probs = [0.025, 0.975]`: probability levels for the lower and upper bounds.
- `g = softplus`: monotonically increasing function used to ensure a positive interval width.
- `kwargs...`: additional keyword arguments passed to the MLP constructors for the inference networks `u` and `v`.

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d, m = 2, 100  # dimension of θ and number of replicates
sampler(K) = NamedMatrix(μ = rand(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(m))
simulator(θ::AbstractMatrix) = reduce(hcat, simulator.(eachcol(θ)))

# Neural network
num_summaries = 3d
summary_network = Chain(Dense(m, 64, relu), Dense(64, 64, relu), Dense(64, num_summaries))

# Initialise and train the estimator
estimator = IntervalEstimator(summary_network, d; num_summaries = num_summaries)
estimator = train(estimator, sampler, simulator, K = 3000)

# Assessment
θ_test = sampler(1000)
Z_test = simulator(θ_test)
assessment = assess(estimator, θ_test, Z_test)
coverage(assessment)

# Inference
θ = sampler(1)
Z = simulator(θ);
estimate(estimator, Z)
interval(estimator, Z)
```
"""
struct IntervalEstimator{M, N, H, C, G} <: BayesEstimator
    summary_network::M
    u::N  # inference network for lower bound
    v::N  # inference network for interval width
    c::C
    probs::H
    g::G
end

# Constructor: summary network, number of parameters, number of summaries => two MLP inference networks
function IntervalEstimator(
    summary_network, num_parameters::Integer, num_summaries::Integer;
    c::Union{Function, Compress} = identity,
    probs = [0.025, 0.975],
    g = softplus,
    kwargs...
)
    if !isa(probs, AbstractArray)
        probs = [probs]
    end
    @assert all(0 .< probs .< 1)
    # NB enforce output_activation = identity for both inference MLPs
    u = MLP(num_summaries, num_parameters; output_activation = identity, kwargs...)
    v = MLP(num_summaries, num_parameters; output_activation = identity, kwargs...)
    @info "IntervalEstimator: num_summaries = $num_summaries."
    IntervalEstimator(summary_network, u, v, c, probs, g)
end

# Constructor: keyword num_summaries
IntervalEstimator(summary_network, num_parameters::Integer; num_summaries::Integer, kwargs...) =
    IntervalEstimator(summary_network, num_parameters, num_summaries; kwargs...)

Optimisers.trainable(est::IntervalEstimator) = (summary_network = est.summary_network, u = est.u, v = est.v)

function (est::IntervalEstimator)(Z)
    S  = _summarystatistics(est, Z)  # shared summary statistics
    bₗ = est.u(S)                    # lower bound
    bᵤ = bₗ .+ est.g.(est.v(S))     # upper bound (guaranteed > lower)
    vcat(est.c(bₗ), est.c(bᵤ))
end

@doc raw"""
	QuantileEstimator <: BayesEstimator
	QuantileEstimator(summary_network, num_parameters; num_summaries, kwargs...)
A neural estimator that jointly estimates a fixed set of marginal posterior
quantiles, with probability levels $\{\tau_1, \dots, \tau_T\}$ controlled by the
keyword argument `probs`. This generalises [`IntervalEstimator`](@ref) to support an arbitrary number of probability levels. 

Given data ``\boldsymbol{Z}``, by default the estimator approximates quantiles of the distributions of 
```math
\theta_i \mid \boldsymbol{Z}, \quad i = 1, \dots, d, 
```
for parameters $\boldsymbol{\theta} \equiv (\theta_1, \dots, \theta_d)'$.
Alternatively, if initialised with `i` set to a positive integer, the estimator approximates quantiles of
the full conditional distribution of  
```math
\theta_i \mid \boldsymbol{Z}, \boldsymbol{\theta}_{-i},
```
where $\boldsymbol{\theta}_{-i}$ denotes the parameter vector with its $i$th
element removed. In this case a parameter summary network maps $\boldsymbol{\theta}_{-i}$
to a vector of summaries, whose output is concatenated with the data summaries before
being passed to the inference networks.

The estimator employs a representation that prevents quantile crossing, namely,
```math
\begin{aligned}
\boldsymbol{q}^{(\tau_1)}(\boldsymbol{Z}) &= \boldsymbol{v}^{(\tau_1)}(\boldsymbol{Z}),\\
\boldsymbol{q}^{(\tau_t)}(\boldsymbol{Z}) &= \boldsymbol{v}^{(\tau_1)}(\boldsymbol{Z}) + \sum_{j=2}^t g(\boldsymbol{v}^{(\tau_j)}(\boldsymbol{Z})), \quad t = 2, \dots, T,
\end{aligned}
```
where $\boldsymbol{q}^{(\tau)}(\boldsymbol{Z})$ denotes the vector of $\tau$-quantiles 
for parameters $\boldsymbol{\theta} \equiv (\theta_1, \dots, \theta_d)'$; 
$\boldsymbol{v}^{(\tau_t)}(\cdot)$, $t = 1, \dots, T$, are MLP inference networks
that map from the summary space to ``\mathbb{R}^d``; and $g(\cdot)$ is a
monotonically increasing function (e.g., exponential or softplus) applied elementwise to
its arguments. If `g = nothing`, the quantiles are estimated independently through the representation
```math
\boldsymbol{q}^{(\tau_t)}(\boldsymbol{Z}) = \boldsymbol{v}^{(\tau_t)}(\boldsymbol{Z}), \quad t = 1, \dots, T.
```

The return value is a matrix with $Td$ rows, where the first $d$ rows correspond
to the estimated quantiles at the first probability level $\tau_1$, the next $d$ rows
to the quantiles at $\tau_2$, and so on. When `i` is specified, $d = 1$ and the
return value is simply a matrix with $T$ rows, one per quantile level. 
The function [`quantiles`](@ref) can be used to format this output in a readable ``d`` × ``T`` matrix.

# Keyword arguments
- `num_summaries::Integer`: the number of summaries output by `summary_network`. Must match the output dimension of `summary_network`.
- `probs = [0.025, 0.5, 0.975]`: probability levels for the quantiles.
- `g = softplus`: monotonically increasing function applied to enforce non-crossing quantiles.
- `i::Union{Integer, Nothing} = nothing`: if set to a positive integer, the estimator targets the full conditional distribution of $\theta_i \mid \boldsymbol{Z}, \boldsymbol{\theta}_{-i}$.
- `num_summaries_θ::Integer = 2 * (num_parameters - 1)`: number of summaries for the parameter summary network (only used when `i` is specified).
- `summary_network_θ_kwargs::NamedTuple = (;)`: keyword arguments for the parameter summary network MLP (only used when `i` is specified).
- `kwargs...`: additional keyword arguments passed to the MLP constructors for the inference networks.

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ N(0, 1) and σ ~ U(0, 1)
d, m = 2, 100  # dimension of θ and number of replicates
sampler(K) = NamedMatrix(μ = rand(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(m))
simulator(θ::AbstractMatrix) = reduce(hcat, simulator.(eachcol(θ)))

# Neural network
num_summaries = 3d
summary_network = Chain(Dense(m, 64, gelu), Dense(64, 64, gelu), Dense(64, num_summaries))

# ---- Quantiles of θᵢ ∣ 𝐙, i = 1, …, d ----

# Initialise the estimator
estimator = QuantileEstimator(summary_network, d; num_summaries = num_summaries)

# Training
estimator = train(estimator, sampler, simulator)

# Assessment
θ_test = sampler(1000)
Z_test = simulator(θ_test)
assessment = assess(estimator, θ_test, Z_test)

# Inference
θ = sampler(1)
Z = simulator(θ);
estimate(estimator, Z)
quantiles(estimator, Z)

# ---- Quantiles of θᵢ ∣ 𝐙, θ₋ᵢ ----

# Initialise estimators respectively targeting quantiles of μ∣Z,σ and σ∣Z,μ
q₁ = QuantileEstimator(summary_network, d; num_summaries = num_summaries, i = 1)
q₂ = QuantileEstimator(summary_network, d; num_summaries = num_summaries, i = 2)

# Training
q₁ = train(q₁, sampler, simulator)
q₂ = train(q₂, sampler, simulator)

# Inference: Estimate quantiles of μ∣Z,σ with known σ
σ₀ = θ["σ", 1]
θ₋ᵢ = [σ₀;]
estimate(q₁, Z, θ₋ᵢ)
quantiles(q₁, Z, θ₋ᵢ)
```
"""
struct QuantileEstimator{M1, M2, V, P, G, I} <: BayesEstimator
    summary_network::M1          # summary network for data Z
    summary_network_θ::M2        # summary network for θ₋ᵢ (nothing when i is nothing)
    v::V                         # vector of T MLP inference networks
    probs::P
    g::G
    i::I
end

# QuantileEstimatorDiscrete is deprecated, use QuantileEstimator instead
const QuantileEstimatorDiscrete = QuantileEstimator
export QuantileEstimatorDiscrete

# Constructor: summary network, number of parameters, number of summaries => MLP inference networks
function QuantileEstimator(
    summary_network, num_parameters::Integer, num_summaries::Integer;
    probs = [0.025, 0.5, 0.975],
    g = softplus,
    i::Union{Integer, Nothing} = nothing,
    num_summaries_θ::Integer = 2 * (num_parameters - 1),
    summary_network_θ_kwargs::NamedTuple = (;),
    kwargs...
)
    if !isa(probs, AbstractArray)
        probs = [probs]
    end
    @assert all(0 .< probs .< 1)
    if !isnothing(i)
        @assert i > 0
        @assert num_parameters >= i "i must be ≤ num_parameters"
    end

    T = length(probs)

    # Number of output parameters from inference networks: 1 for full conditional, d for marginals
    num_out = isnothing(i) ? num_parameters : 1

    if isnothing(i)
        # Marginal posteriors: inference networks take only data summaries
        inference_input_dim = num_summaries
        summary_network_θ = nothing
    else
        # Full conditional: concatenate data summaries with parameter summaries
        # θ₋ᵢ has (num_parameters - 1) components
        summary_network_θ = MLP(num_parameters - 1, num_summaries_θ; output_activation = identity, summary_network_θ_kwargs...)
        inference_input_dim = num_summaries + num_summaries_θ
    end

    # NB enforce output_activation = identity for all inference MLPs
    v = [MLP(inference_input_dim, num_out; output_activation = identity, kwargs...) for _ in 1:T]

    @info "QuantileEstimator: num_summaries = $num_summaries, T = $T quantile levels$(isnothing(i) ? "" : ", full conditional i = $i")."
    QuantileEstimator(summary_network, summary_network_θ, v, probs, g, i)
end

# Constructor: keyword num_summaries
QuantileEstimator(summary_network, num_parameters::Integer; num_summaries::Integer, kwargs...) =
    QuantileEstimator(summary_network, num_parameters, num_summaries; kwargs...)

Optimisers.trainable(est::QuantileEstimator) = isnothing(est.summary_network_θ) ?
    (summary_network = est.summary_network, v = est.v) :
    (summary_network = est.summary_network, summary_network_θ = est.summary_network_θ, v = est.v)

# Internal forward pass: accepts pre-computed summary statistics S (already concatenated with θ summaries if needed)
function _apply_quantile_networks(est::QuantileEstimator, S)
    # Apply each inference network to the shared summary statistics
    v_out = map(net -> net(S), est.v)

    # Impose monotonicity if g is specified
    if isnothing(est.g)
        q = v_out
    else
        gv = broadcast.(est.g, v_out[2:end])
        q = cumsum([v_out[1], gv...])
    end

    reduce(vcat, q)
end

# Forward pass
function (est::QuantileEstimator)(Z)
    @assert isnothing(est.i) "This estimator targets a full conditional: call est(Z, θ₋ᵢ) instead"
    S = _summarystatistics(est, Z)
    _apply_quantile_networks(est, S)
end

# Forward pass for full conditional: accepts Z and θ₋ᵢ
function (est::QuantileEstimator)(Z, θ₋ᵢ)
    @assert !isnothing(est.i) "This estimator targets marginal posteriors: call est(Z) instead"

    S_Z = _summarystatistics(est, Z)

    # Handle scalar, vector, or matrix θ₋ᵢ inputs
    θ₋ᵢ = _prepare_θ₋ᵢ(θ₋ᵢ, S_Z)

    S_θ = est.summary_network_θ(θ₋ᵢ)
    S   = vcat(S_Z, S_θ)
    _apply_quantile_networks(est, S)
end

# Tuple method used internally during training for full conditional case
(est::QuantileEstimator)(Zθ::Tuple) = est(Zθ[1], Zθ[2])

# Helper: coerce θ₋ᵢ to a matrix with columns matching the number of data sets
function _prepare_θ₋ᵢ(θ₋ᵢ::Number, S_Z)
    K = size(S_Z, 2)
    repeat(Float32[θ₋ᵢ;;], 1, K)
end
function _prepare_θ₋ᵢ(θ₋ᵢ::AbstractVector{<:Number}, S_Z)
    K = size(S_Z, 2)
    # If the vector has the same length as the number of data sets, treat each
    # element as the conditioning value for the corresponding data set.
    # Otherwise treat it as a single conditioning vector replicated K times.
    if length(θ₋ᵢ) == K
        # ambiguous case: assume single-parameter conditioning replicated K times
        reshape(Float32.(θ₋ᵢ), 1, K)
    else
        repeat(Float32.(θ₋ᵢ), 1, K)
    end
end
function _prepare_θ₋ᵢ(θ₋ᵢ::AbstractMatrix, S_Z)
    f32(θ₋ᵢ)
end
# Vector-of-vectors (one conditioning vector per data set)
function _prepare_θ₋ᵢ(θ₋ᵢ::AbstractVector{<:AbstractVector}, S_Z)
    reduce(hcat, Float32.(θ₋ᵢ))
end

function _inputoutput(estimator::QuantileEstimator, Z, θ)
    i = estimator.i
    if isnothing(i)
        input  = Z
        output = θ
    else
        @assert size(θ, 1) >= i "The number of parameters in the model (size(θ, 1) = $(size(θ, 1))) must be at least as large as the value of i stored in the estimator (estimator.i = $(estimator.i))"
        θᵢ  = θ[i:i, :]
        θ₋ᵢ = θ[Not(i), :]
        input  = (Z, θ₋ᵢ)
        output = θᵢ
    end
    return input, output
end

function _loss(estimator::Union{IntervalEstimator, QuantileEstimator}, loss = nothing)
    # NB: probs is on the CPU but CUDA handles the implicit transfer in quantileloss
    (estimate, θ) -> quantileloss(estimate, θ, estimator.probs)
end


function assess(
    estimator::Union{IntervalEstimator, Ensemble{<:IntervalEstimator}},
    θ::P, Z;
    parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(θ, 1)],
    estimator_name::Union{Nothing, String} = nothing,
    estimator_names::Union{Nothing, String} = nothing,
    use_gpu::Bool = true
) where {P <: Union{AbstractMatrix, AbstractParameterSet}}
    θ, parameter_names, d, K, J, m = _assess_setup(θ, Z, parameter_names)

    # Apply the estimator to data 
    runtime = @elapsed estimates = estimate(estimator, Z, use_gpu = use_gpu)
    runtime = DataFrame(runtime = runtime)

    # Empirical risk
    empirical_risk = _computerisk(estimator, θ, Z)

    # Convert to DataFrame and add information
    estimate_names = repeat(parameter_names, outer = 2) .* repeat(["_lower", "_upper"], inner = d)
    estimates = _estimates_to_df(estimates, estimate_names, K, J, m)
    θ_df = _truth_to_df(θ, parameter_names, nrow(estimates))

    # Merge true parameters and estimates
    df = _merge_interval(θ_df, estimates)
    probs = estimator isa Ensemble{<:IntervalEstimator} ? estimator[1].probs : estimator.probs
    df[:, "α"] .= 1 - (probs[2] - probs[1])

    # Add estimator name if it was provided
    estimator_name = _resolve_estimator_name(estimator_name, estimator_names)
    _add_estimator_name!(df, runtime, estimator_name)

    return Assessment(df, runtime, nothing, empirical_risk)
end

function assess(
    estimator::Union{QuantileEstimator, Ensemble{<:QuantileEstimator}},
    θ::P, Z;
    parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(θ, 1)],
    estimator_name::Union{Nothing, String} = nothing,
    estimator_names::Union{Nothing, String} = nothing,
    use_gpu::Bool = true
) where {P <: Union{AbstractMatrix, AbstractParameterSet}}
    θ, parameter_names, d, K, J, m = _assess_setup(θ, Z, parameter_names)

    probs = estimator isa Ensemble{<:QuantileEstimator} ? estimator[1].probs : estimator.probs
    n_probs = length(probs)

    # NB: _computerisk not set up for Ensemble
    empirical_risk = estimator isa QuantileEstimator ? _computerisk(estimator, θ, Z) : nothing

    i = estimator.i
    if isnothing(i)
        set_info = nothing
    else
        θ₋ᵢ = θ[Not(i), :]
        set_info = eachcol(θ₋ᵢ)
        θ = θ[i:i, :]
        parameter_names = parameter_names[i:i]
        d = 1
    end

    runtime = @elapsed estimates = estimate(estimator, Z, set_info, use_gpu = use_gpu)
    runtime = DataFrame(runtime = runtime)

    df = DataFrame(
        parameter = repeat(repeat(parameter_names, outer = n_probs), K),
        truth     = mapreduce(col -> repeat(col, n_probs), vcat, eachcol(θ)),
        prob      = repeat(repeat(probs, inner = d), K),
        estimate  = vec(estimates),
        k         = repeat(1:K, inner = n_probs*d)
    )

    estimator_name = _resolve_estimator_name(estimator_name, estimator_names)
    _add_estimator_name!(df, runtime, estimator_name)

    return Assessment(df, runtime, nothing, empirical_risk)
end

