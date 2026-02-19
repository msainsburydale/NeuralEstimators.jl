@doc raw"""
	QuantileEstimator <: BayesEstimator
	QuantileEstimator(v; probs = [0.025, 0.5, 0.975], g = Flux.softplus, i = nothing)
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
element removed. 

The estimator employs a representation that prevents quantile crossing, namely,
```math
\begin{aligned}
\boldsymbol{q}^{(\tau_1)}(\boldsymbol{Z}) &= \boldsymbol{v}^{(\tau_1)}(\boldsymbol{Z}),\\
\boldsymbol{q}^{(\tau_t)}(\boldsymbol{Z}) &= \boldsymbol{v}^{(\tau_1)}(\boldsymbol{Z}) + \sum_{j=2}^t g(\boldsymbol{v}^{(\tau_j)}(\boldsymbol{Z})), \quad t = 2, \dots, T,
\end{aligned}
```
where $\boldsymbol{q}^{(\tau)}(\boldsymbol{Z})$ denotes the vector of $\tau$-quantiles 
for parameters $\boldsymbol{\theta} \equiv (\theta_1, \dots, \theta_d)'$; 
$\boldsymbol{v}^{(\tau_t)}(\cdot)$, $t = 1, \dots, T$, are neural networks
that map from the sample space to ``\mathbb{R}^d``; and $g(\cdot)$ is a
monotonically increasing function (e.g., exponential or softplus) applied elementwise to
its arguments. If `g = nothing`, the quantiles are estimated independently through the representation
```math
\boldsymbol{q}^{(\tau_t)}(\boldsymbol{Z}) = \boldsymbol{v}^{(\tau_t)}(\boldsymbol{Z}), \quad t = 1, \dots, T.
```

When the neural networks are [`DeepSet`](@ref) objects, two requirements must be met. 
First, the number of input neurons in the first layer of the outer network must equal the number of
neurons in the final layer of the inner network plus $\text{dim}(\boldsymbol{\theta}_{-i})$, where we define 
$\text{dim}(\boldsymbol{\theta}_{-i}) \equiv 0$ when targetting marginal posteriors of the form $\theta_i \mid \boldsymbol{Z}$ (the default behaviour). 
Second, the number of output neurons in the final layer of the outer network must equal $d - \text{dim}(\boldsymbol{\theta}_{-i})$. 

The return value is a matrix with $\{d - \text{dim}(\boldsymbol{\theta}_{-i})\} \times T$ rows, where the
first ``T`` rows correspond to the estimated quantiles for the first
parameter, the second ``T`` rows corresponds to the estimated quantiles for the second parameter, and so on.

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d = 2     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 30    # number of independent replicates in each data set
sample(K) = rand32(d, K)
simulate(θ, m) = [ϑ[1] .+ ϑ[2] .* randn32(n, m) for ϑ in eachcol(θ)]

# ---- Quantiles of θᵢ ∣ 𝐙, i = 1, …, d ----

# Neural network
w = 64   # width of each hidden layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, d))
v = DeepSet(ψ, ϕ)

# Initialise the estimator
estimator = QuantileEstimator(v)

# Train the estimator
estimator = train(estimator, sample, simulate, m = m)

# Inference with "observed" data 
θ = [0.8f0; 0.1f0]
Z = simulate(θ, m)
estimate(estimator, Z) 

# ---- Quantiles of θᵢ ∣ 𝐙, θ₋ᵢ ----

# Neural network
w = 64  # width of each hidden layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w + 1, w, relu), Dense(w, d - 1))
v = DeepSet(ψ, ϕ)

# Initialise estimators respectively targetting quantiles of μ∣Z,σ and σ∣Z,μ
q₁ = QuantileEstimator(v; i = 1)
q₂ = QuantileEstimator(v; i = 2)

# Train the estimators
q₁ = train(q₁, sample, simulate, m = m)
q₂ = train(q₂, sample, simulate, m = m)

# Estimate quantiles of μ∣Z,σ with σ = 0.5 and for many data sets
θ₋ᵢ = 0.5f0
q₁(Z, θ₋ᵢ)

# Estimate quantiles of μ∣Z,σ with σ = 0.5 for a single data set
q₁(Z[1], θ₋ᵢ)
```
"""
struct QuantileEstimator{V, P, G, I} <: BayesEstimator #TODO function for neat output as dxT matrix like interval() 
    v::V
    probs::P
    g::G
    i::I
end
function QuantileEstimator(v; probs = [0.025, 0.5, 0.975], g = Flux.softplus, i::Union{Integer, Nothing} = nothing)
    if !isa(probs, AbstractArray)
        probs = [probs]
    end
    @assert all(0 .< probs .< 1)
    if !isnothing(i)
        @assert i > 0
    end
    QuantileEstimator(deepcopy.(repeat([v], length(probs))), probs, g, i)
end
Flux.trainable(est::QuantileEstimator) = (v = est.v,)
function (est::QuantileEstimator)(input) # input might be Z, or a tuple (Z, θ₋ᵢ)

    # Apply each neural network to Z
    v = map(est.v) do v
        v(input)
    end

    # If g is specified, impose monotonicity
    if isnothing(est.g)
        q = v
    else
        gv = broadcast.(est.g, v[2:end])
        q = cumsum([v[1], gv...])
    end

    # Convert to matrix
    reduce(vcat, q)
end
# user-level convenience methods (not used internally) for full conditional estimation
function (est::QuantileEstimator)(Z, θ₋ᵢ::Vector)
    i = est.i
    @assert !isnothing(i) "slot i must be specified when approximating a full conditional"
    if isa(Z, Vector) # repeat θ₋ᵢ to match the number of data sets
        θ₋ᵢ = [θ₋ᵢ for _ in eachindex(Z)]
    end
    est((Z, θ₋ᵢ))  # "Tupleise" the input and apply the estimator
end
(est::QuantileEstimator)(Z, θ₋ᵢ::Number) = est(Z, [θ₋ᵢ])
const QuantileEstimatorDiscrete = QuantileEstimator # alias


@doc raw"""
	IntervalEstimator <: BayesEstimator
	IntervalEstimator(u, v = u, c::Union{Function, Compress} = identity; probs = [0.025, 0.975], g = exp)
	IntervalEstimator(u, c::Union{Function, Compress}; probs = [0.025, 0.975], g = exp)
A neural estimator that jointly estimates marginal posterior credible intervals based on the probability levels `probs` (by default, 95% central credible intervals).

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

The functions ``c_i(⋅)`` may be collectively defined by a ``d``-dimensional [`Compress`](@ref) object, which can constrain the interval estimator's output to the prior support. If these functions are unspecified, they will be set to the identity function so that the range of the intervals will be unrestricted. If only a single neural-network architecture is provided, it will be used for both ``\boldsymbol{u}(⋅)`` and ``\boldsymbol{v}(⋅)``.

The return value when applied to data using [`estimate`()](@ref) is a matrix with ``2d`` rows, where the first and second ``d`` rows correspond to the lower and upper bounds, respectively. The function [`interval()`](@ref) can be used to format this output in a readable ``d`` × 2 matrix.  

See also [`QuantileEstimator`](@ref).

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d = 2     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 100   # number of independent replicates
sample(K) = rand32(d, K)
simulate(θ, m) = [ϑ[1] .+ ϑ[2] .* randn(n, m) for ϑ in eachcol(θ)]

# Neural network
w = 128   # width of each hidden layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, d))
u = DeepSet(ψ, ϕ)

# Initialise the estimator
estimator = IntervalEstimator(u)

# Train the estimator
estimator = train(estimator, sample, simulate, m = m)

# Inference with "observed" data 
θ = [0.8f0; 0.1f0]
Z = simulate(θ, m)
estimate(estimator, Z) 
interval(estimator, Z)
```
"""
struct IntervalEstimator{N, H, C, G} <: BayesEstimator
    u::N
    v::N
    c::C
    probs::H
    g::G
end
function IntervalEstimator(u, v = u, c::Union{Function, Compress} = identity; probs = [0.025, 0.975], g = exp)
    if !isa(probs, AbstractArray)
        probs = [probs]
    end
    @assert all(0 .< probs .< 1)
    IntervalEstimator(deepcopy(u), deepcopy(v), c, probs, g)
end
IntervalEstimator(u, c::Union{Function, Compress}; kwargs...) = IntervalEstimator(deepcopy(u), deepcopy(u), c; kwargs...)
Flux.trainable(est::IntervalEstimator) = (u = est.u, v = est.v)
function (est::IntervalEstimator)(Z)
    bₗ = est.u(Z)                # lower bound
    bᵤ = bₗ .+ est.g.(est.v(Z))  # upper bound
    vcat(est.c(bₗ), est.c(bᵤ))
end



# function posterior(Z; μ₀ = 0, σ₀ = 1, σ² = 1)
# 	μ̃ = (1/σ₀^2 + length(Z)/σ²)^-1 * (μ₀/σ₀^2 + sum(Z)/σ²)
# 	σ̃ = sqrt((1/σ₀^2 + length(Z)/σ²)^-1)
# 	Normal(μ̃, σ̃)
# end

# ; and see [`QuantileEstimatorContinuous`](@ref) for estimating posterior quantiles based on a continuous probability level provided as input to the neural network.
@doc raw"""
	QuantileEstimatorContinuous <: BayesEstimator
	QuantileEstimatorContinuous(network; i = nothing, num_training_probs::Integer = 1)
A neural estimator that estimates marginal posterior quantiles, with the probability level `τ` given as input to the neural network.

Given data $\boldsymbol{Z}$ and the desired probability level 
$\tau ∈ (0, 1)$, by default the estimator approximates the $\tau$-quantile of the distributions of 
```math
\theta_i \mid \boldsymbol{Z}, \quad i = 1, \dots, d, 
```
for parameters $\boldsymbol{\theta} \equiv (\theta_1, \dots, \theta_d)'$.
Alternatively, if initialised with `i` set to a positive integer, the estimator
approximates the $\tau$-quantile of the full conditional distribution of 
```math
\theta_i \mid \boldsymbol{Z}, \boldsymbol{\theta}_{-i},
```
where $\boldsymbol{\theta}_{-i}$ denotes the parameter vector with its $i$th element removed. 

Although not a requirement, one may employ a (partially) monotonic neural
network to prevent quantile crossing (i.e., to ensure that the
$\tau_1$-quantile does not exceed the $\tau_2$-quantile for any
$\tau_2 > \tau_1$). There are several ways to construct such a neural network:
one simple yet effective approach is to ensure that all weights associated with
$\tau$ are strictly positive
(see, e.g., [Cannon, 2018](https://link.springer.com/article/10.1007/s00477-018-1573-6)),
and this can be done using the [`DensePositive`](@ref) layer as shown in the example below.

When the neural network is a [`DeepSet`](@ref), two requirements must be met. First, the number of input neurons in the first layer of the outer network must equal the number of
neurons in the final layer of the inner network plus $1 + \text{dim}(\boldsymbol{\theta}_{-i})$, where we define 
$\text{dim}(\boldsymbol{\theta}_{-i}) \equiv 0$ when targetting marginal posteriors of the form $\theta_i \mid \boldsymbol{Z}$ (the default behaviour). 
Second, the number of output neurons in the final layer of the outer network must equal $d - \text{dim}(\boldsymbol{\theta}_{-i})$. 

The return value is a matrix with $d - \text{dim}(\boldsymbol{\theta}_{-i})$ rows,
corresponding to the estimated quantile for each parameter not in $\boldsymbol{\theta}_{-i}$.

See also [`QuantileEstimator`](@ref).

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d = 2     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 30    # number of independent replicates in each data set
sample(K) = rand32(d, K)
simulateZ(θ, m) = [ϑ[1] .+ ϑ[2] .* randn32(n, m) for ϑ in eachcol(θ)]
simulateτ(K)    = [rand32(10) for k in 1:K]
simulate(θ, m)  = simulateZ(θ, m), simulateτ(size(θ, 2))

# ---- Quantiles of θᵢ ∣ 𝐙, i = 1, …, d ----

# Neural network: partially monotonic network to preclude quantile crossing
w = 64  # width of each hidden layer
ψ = Chain(
	Dense(n, w, relu),
	Dense(w, w, relu),
	Dense(w, w, relu)
	)
ϕ = Chain(
	DensePositive(Dense(w + 1, w, relu); last_only = true),
	DensePositive(Dense(w, w, relu)),
	DensePositive(Dense(w, d))
	)
network = DeepSet(ψ, ϕ)

# Initialise the estimator
q̂ = QuantileEstimatorContinuous(network)

# Train the estimator
q̂ = train(q̂, sample, simulate, m = m)

# Test data 
θ = sample(1000)
Z = simulateZ(θ, m)

# Estimate 0.1-quantile for each parameter and for many data sets
τ = 0.1f0
q̂(Z, τ)

# Estimate multiple quantiles for each parameter and for many data sets
# (note that τ is given as a row vector)
τ = f32([0.1, 0.25, 0.5, 0.75, 0.9])'
q̂(Z, τ)

# Estimate multiple quantiles for a single data set 
q̂(Z[1], τ)

# ---- Quantiles of θᵢ ∣ 𝐙, θ₋ᵢ ----

# Neural network: partially monotonic network to preclude quantile crossing
w = 64  # width of each hidden layer
ψ = Chain(
	Dense(n, w, relu),
	Dense(w, w, relu),
	Dense(w, w, relu)
	)
ϕ = Chain(
	DensePositive(Dense(w + 2, w, relu); last_only = true),
	DensePositive(Dense(w, w, relu)),
	DensePositive(Dense(w, d - 1))
	)
network = DeepSet(ψ, ϕ)

# Initialise the estimator targetting μ∣Z,σ
i = 1
q̂ᵢ = QuantileEstimatorContinuous(network; i = i)

# Train the estimator
q̂ᵢ = train(q̂ᵢ, prior, simulate, m = m)

# Test data 
θ = sample(1000)
Z = simulateZ(θ, m)

# Estimate quantiles of μ∣Z,σ with σ = 0.5 and for many data sets
# (can use θ[InvertedIndices.Not(i), :] to determine the order in which the conditioned parameters should be given)
θ₋ᵢ = 0.5f0
τ = f32([0.1, 0.25, 0.5, 0.75, 0.9])
q̂ᵢ(Z, θ₋ᵢ, τ)

# Estimate quantiles of μ∣Z,σ with σ = 0.5 and for a single data set
q̂ᵢ(Z[1], θ₋ᵢ, τ)
```
"""
struct QuantileEstimatorContinuous{N, I} <: NeuralEstimator
    network::N
    i::I
end
function QuantileEstimatorContinuous(network; i::Union{Integer, Nothing} = nothing)
    if !isnothing(i)
        @assert i > 0
    end
    QuantileEstimatorContinuous(network, i)
end
# core method (used internally)
(est::QuantileEstimatorContinuous)(tup::Tuple) = est.network(tup)
# user-level convenience functions (not used internally)
function (est::QuantileEstimatorContinuous)(Z, τ)
    if !isnothing(est.i)
        error("To estimate the τ-quantile of the full conditional θᵢ|Z,θ₋ᵢ the call should be of the form estimator(Z, θ₋ᵢ, τ)")
    end
    est((Z, τ)) # "Tupleise" input and pass to Tuple method
end
function (est::QuantileEstimatorContinuous)(Z, τ::Number)
    est(Z, [τ])
end
function (est::QuantileEstimatorContinuous)(Z::V, τ::Number) where {V <: AbstractVector{A}} where {A}
    est(Z, repeat([[τ]], length(Z)))
end
# user-level convenience functions (not used internally) for full conditional estimation
function (est::QuantileEstimatorContinuous)(Z, θ₋ᵢ::Matrix, τ::Matrix)
    i = est.i
    @assert !isnothing(i) "slot i must be specified when approximating a full conditional"
    if size(θ₋ᵢ, 2) != size(τ, 2)
        @assert size(θ₋ᵢ, 2) == 1 "size(θ₋ᵢ, 2)=$(size(θ₋ᵢ, 2)) and size(τ, 2)=$(size(τ, 2)) do not match"
        θ₋ᵢ = repeat(θ₋ᵢ, outer = (1, size(τ, 2)))
    end
    θ₋ᵢτ = vcat(θ₋ᵢ, τ) # combine parameters and probability level into single pxK matrix
    q = est((Z, θ₋ᵢτ))  # "Tupleise" the input and pass to tuple method
    if !isa(q, Vector)
        q = [q]
    end
    reduce(hcat, permutedims.(q))
end
(est::QuantileEstimatorContinuous)(Z, θ₋ᵢ::Matrix, τ::Vector) = est(Z, θ₋ᵢ, permutedims(reduce(vcat, τ)))
(est::QuantileEstimatorContinuous)(Z, θ₋ᵢ::Matrix, τ::Number) = est(Z, θ₋ᵢ, repeat([τ], size(θ₋ᵢ, 2)))
(est::QuantileEstimatorContinuous)(Z, θ₋ᵢ::Vector, τ::Vector) = est(Z, reshape(θ₋ᵢ, :, 1), permutedims(τ))
(est::QuantileEstimatorContinuous)(Z, θ₋ᵢ::Vector, τ::Number) = est(Z, θ₋ᵢ, [τ])
(est::QuantileEstimatorContinuous)(Z, θ₋ᵢ::Number, τ::Number) = est(Z, [θ₋ᵢ], τ)
(est::QuantileEstimatorContinuous)(Z, θ₋ᵢ::Number, τ::Vector) = est(Z, [θ₋ᵢ], τ)


function train(estimator::Union{IntervalEstimator, QuantileEstimatorDiscrete}, args...; kwargs...)

    # Get the keyword arguments
    kwargs = (; kwargs...)

    # Define the loss function based on the given probabiltiy levels
    τ = f32(estimator.probs)
    # Determine if we need to move τ to the GPU
    use_gpu = haskey(kwargs, :use_gpu) ? kwargs.use_gpu : true
    device = _checkgpu(use_gpu, verbose = false)
    τ = device(τ)
    # Define the loss function
    qloss = (estimator, θ) -> quantileloss(estimator, θ, τ)

    # Notify the user if "loss" is in the keyword arguments
    if haskey(kwargs, :loss)
        @info "The keyword argument `loss` is not required when training a $(typeof(estimator)), since in this case the quantile loss is always used"
    end
    # Add our quantile loss to the list of keyword arguments
    kwargs = merge(kwargs, (loss = qloss,))

    # Train the estimator
    _train(estimator, args...; kwargs...)
end

function train(estimator::QuantileEstimatorContinuous, args...; kwargs...)
    # We define the loss function in the method _risk(estimator::QuantileEstimatorContinuous)
    # Here, just notify the user if they've assigned a loss function
    kwargs = (; kwargs...)
    if haskey(kwargs, :loss)
        @info "The keyword argument `loss` is not required when training a $(typeof(estimator)), since in this case the quantile loss is always used"
    end
    _train(estimator, args...; kwargs...)
end

function _constructset(estimator::QuantileEstimatorDiscrete, Z, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    Z = f32(Z)
    θ = f32(_extractθ(θ))

    i = estimator.i
    if isnothing(i)
        input = Z
        output = θ
    else
        @assert size(θ, 1) >= i "The number of parameters in the model (size(θ, 1) = $(size(θ, 1))) must be at least as large as the value of i stored in the estimator (estimator.i = $(estimator.i))"
        θᵢ = θ[i:i, :]
        θ₋ᵢ = θ[Not(i), :]
        input = (Z, θ₋ᵢ) # "Tupleise" the input
        output = θᵢ
    end

    _DataLoader((input, output), batchsize)
end

function _constructset(estimator::QuantileEstimatorContinuous, Zτ, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    θ = f32(_extractθ(θ))
    Z, τ = Zτ
    Z = f32(Z)
    τ = f32(τ)

    i = estimator.i
    if isnothing(i)
        input = (Z, τ)
        output = θ
    else
        @assert size(θ, 1) >= i "The number of parameters in the model (size(θ, 1) = $(size(θ, 1))) must be at least as large as the value of i stored in the estimator (estimator.i = $(estimator.i))"
        θᵢ = θ[i:i, :]
        θ₋ᵢ = θ[Not(i), :]
        # Combine each θ₋ᵢ with the corresponding vector of
        # probability levels, which requires repeating θ₋ᵢ appropriately
        θ₋ᵢτ = map(eachindex(τ)) do k
            τₖ = τ[k]
            θ₋ᵢₖ = repeat(θ₋ᵢ[:, k:k], inner = (1, length(τₖ)))
            vcat(θ₋ᵢₖ, τₖ')
        end
        input = (Z, θ₋ᵢτ)   # "Tupleise" the input
        output = θᵢ
    end

    _DataLoader((input, output), batchsize)
end


function _risk(estimator::QuantileEstimatorContinuous, loss, set::DataLoader, device, optimiser = nothing)
    sum_loss = 0.0f0
    K = 0
    for (input, output) in set
        k = size(output)[end]
        input, output = input |> device, output |> device

        if isnothing(estimator.i)
            Z, τ = input
            input1 = Z
            input2 = permutedims.(τ)
            input = (input1, input2)
            τ = reduce(hcat, τ)                # reduce from vector of vectors to matrix
        else
            Z, θ₋ᵢτ = input
            τ = [x[end, :] for x ∈ θ₋ᵢτ] # extract probability levels
            τ = reduce(hcat, τ)          # reduce from vector of vectors to matrix
        end

        # Repeat τ and θ to facilitate broadcasting and indexing
        # Note that repeat() cannot be differentiated by Zygote
        p = size(output, 1)
        @ignore_derivatives τ = repeat(τ, inner = (p, 1))
        @ignore_derivatives output = repeat(output, inner = (size(τ, 1) ÷ p, 1))

        if !isnothing(optimiser)
            ls, ∇ = Flux.withgradient(estimator -> quantileloss(estimator(input), output, τ), estimator)
            Flux.update!(optimiser, estimator, ∇[1])
        else
            ls = quantileloss(estimator(input), output, τ)
        end
        # Convert average loss to a sum and add to total
        sum_loss += ls * k
        K += k
    end
    return cpu(sum_loss/K)
end


function assess(
    estimator::Union{IntervalEstimator, Ensemble{<:IntervalEstimator}},
    θ::P, Z;
    parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(θ, 1)],
    estimator_name::Union{Nothing, String} = nothing,
    estimator_names::Union{Nothing, String} = nothing,
    use_gpu::Bool = true
) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

    # Extract the matrix of parameters and check that the parameter names match the dimension of θ
    θ = _extractθ(θ)
    d, K = size(θ)
    if θ isa NamedMatrix
        parameter_names = names(θ, 1)
    end
    @assert length(parameter_names) == d

    # Get the number of data sets and check that it conforms with the number of parameter vectors stored in θ
    KJ = numobs(Z)
    @assert KJ % K == 0 "The number of data sets in `Z` must be a multiple of the number of parameter vectors stored in `θ`"
    J = KJ ÷ K
    if J > 1
        θ = repeat(θ, outer = (1, J))
    end

    # If the data are stored as a vector, get the number of replicates stored in each element 
    if Z isa AbstractVector
        m = numberreplicates(Z)
    else
        m = fill(1, KJ)
    end

    # Apply the estimator to data 
    runtime = @elapsed θ̂ = estimate(estimator, Z, use_gpu = use_gpu)

    # Convert to DataFrame and add information
    runtime = DataFrame(runtime = runtime)
    estimate_names = repeat(parameter_names, outer = 2) .* repeat(["_lower", "_upper"], inner = d)
    θ̂ = DataFrame(θ̂', estimate_names)
    θ̂[!, "m"] = m
    θ̂[!, "k"] = repeat(1:K, J)
    θ̂[!, "j"] = repeat(1:J, inner = K)

    # Add estimator name if it was provided
    if !isnothing(estimator_names)
        estimator_name = estimator_names
    end # deprecation coercion
    if !isnothing(estimator_name)
        θ̂[!, "estimator"] .= estimator_name
        runtime[!, "estimator"] .= estimator_name
    end

    # Dataframe containing the true parameters, repeated if necessary 
    θ = convert(Matrix, θ)
    θ = DataFrame(θ', parameter_names)
    θ = repeat(θ, outer = nrow(θ̂) ÷ nrow(θ))
    θ = stack(θ, variable_name = :parameter, value_name = :truth) # transform to long form

    # Merge true parameters and estimates
    df = _merge2(θ, θ̂)
    probs = estimator isa Ensemble{<:IntervalEstimator} ? estimator[1].probs : estimator.probs
    df[:, "α"] .= 1 - (probs[2] - probs[1])

    return Assessment(df, runtime)
end

function assess(
    estimator::Union{QuantileEstimatorContinuous, QuantileEstimatorDiscrete, Ensemble{<:QuantileEstimatorContinuous}, Ensemble{<:QuantileEstimatorDiscrete}},
    θ::P, Z;
    parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(θ, 1)],
    estimator_name::Union{Nothing, String} = nothing,
    estimator_names::Union{Nothing, String} = nothing, # for backwards compatibility
    use_gpu::Bool = true,
    probs = f32(range(0.01, stop = 0.99, length = 100))
) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

    # Extract the matrix of parameters and check that the parameter names match the dimension of θ
    θ = _extractθ(θ)
    d, K = size(θ)
    if θ isa NamedMatrix
        parameter_names = names(θ, 1)
    end
    @assert length(parameter_names) == d

    # Get the number of data sets and check that it conforms with the number of parameter vectors stored in θ
    KJ = numobs(Z)
    @assert KJ % K == 0 "The number of data sets in `Z` must be a multiple of the number of parameter vectors stored in `θ`"
    J = KJ ÷ K
    if J > 1
        θ = repeat(θ, outer = (1, J))
    end

    # If the data are stored as a vector, get the number of replicates stored in each element 
    if Z isa AbstractVector
        m = numberreplicates(Z)
    else
        m = fill(1, KJ)
    end

    # Get the probability levels 
    if estimator isa Union{QuantileEstimatorDiscrete, Ensemble{<:QuantileEstimatorDiscrete}}
        probs = estimator isa Ensemble{<:QuantileEstimatorDiscrete} ? estimator[1].probs : estimator.probs
    else
        τ = [permutedims(probs) for _ in eachindex(Z)] # convert from vector to vector of matrices
    end
    n_probs = length(probs)

    # Construct input set
    i = estimator.i
    if isnothing(i)
        if estimator isa Union{QuantileEstimatorDiscrete, Ensemble{<:QuantileEstimatorDiscrete}}
            set_info = nothing
        else
            set_info = τ
        end
    else
        θ₋ᵢ = θ[Not(i), :]
        if estimator isa Union{QuantileEstimatorDiscrete, Ensemble{<:QuantileEstimatorDiscrete}}
            set_info = eachcol(θ₋ᵢ)
        else
            # Combine each θ₋ᵢ with the corresponding vector of probability levels, which requires repeating θ₋ᵢ appropriately
            set_info = map(1:K) do k
                θ₋ᵢₖ = repeat(θ₋ᵢ[:, k:k], inner = (1, n_probs))
                vcat(θ₋ᵢₖ, probs')
            end
        end
        θ = θ[i:i, :]
        parameter_names = parameter_names[i:i]
    end

    # Estimates 
    runtime = @elapsed θ̂ = estimate(estimator, Z, set_info, use_gpu = use_gpu)

    # Convert to DataFrame and add information
    d = size(θ, 1)
    runtime = DataFrame(runtime = runtime)
    df = DataFrame(
        parameter = repeat(repeat(parameter_names, inner = n_probs), K),
        truth = repeat(vec(θ), inner = n_probs),
        prob = repeat(repeat(probs, outer = d), K),
        estimate = vec(θ̂),
        m = repeat(m, inner = n_probs*d),
        k = repeat(1:K, inner = n_probs*d),
        j = 1 # just for consistency with other methods
    )

    # Add estimator name if it was provided
    if !isnothing(estimator_names)
        estimator_name = estimator_names
    end # deprecation coercion
    if !isnothing(estimator_name)
        df[!, "estimator"] .= estimator_name
        runtime[!, "estimator"] .= estimator_name
    end

    return Assessment(df, runtime)
end