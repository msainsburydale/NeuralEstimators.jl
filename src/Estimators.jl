"""
	NeuralEstimator
An abstract supertype for all neural estimators.
"""
abstract type NeuralEstimator end

"""
	BayesEstimator <: NeuralEstimator
An abstract supertype for neural Bayes estimators.
"""
abstract type BayesEstimator <: NeuralEstimator end

"""
	PointEstimator <: BayesEstimator
    PointEstimator(network)
A neural point estimator, where the neural `network` is a mapping from the sample space to the parameter space.
"""
struct PointEstimator{N} <: BayesEstimator
    network::N
end
(estimator::PointEstimator)(Z) = estimator.network(Z)

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

@doc raw"""
	PosteriorEstimator <: NeuralEstimator
	PosteriorEstimator(network, q::ApproximateDistribution)
    PosteriorEstimator(network, d::Integer, dstar::Integer = d; q::ApproximateDistribution = NormalisingFlow, kwargs...)
A neural estimator that approximates the posterior distribution $p(\boldsymbol{\theta} \mid \boldsymbol{Z})$, based on a neural `network` and an approximate distribution `q` (see the available in-built [Approximate distributions](@ref)). 

The neural `network` is a mapping from the sample space to a space determined by the chosen approximate distribution `q`. Often, the output space is the space $\mathcal{K}$ of the approximate-distribution parameters $\boldsymbol{\kappa}$. However, for certain distributions (notably, [`NormalisingFlow`](@ref)), the neural network outputs summary statistics of suitable dimension (e.g., the dimension $d$ of the parameter vector), which are then transformed into parameters of the approximate distribution using conventional multilayer perceptrons (see [`NormalisingFlow`](@ref)).

The convenience constructor `PosteriorEstimator(network, d::Integer, dstar::Integer = d)` builds the approximate distribution automatically, with the keyword arguments passed onto the approximate-distribution constructor.  

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d = 2     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 30    # number of independent replicates in each data set
sample(K) = rand32(d, K)
simulate(θ, m) = [ϑ[1] .+ ϑ[2] .* randn32(n, m) for ϑ in eachcol(θ)]

# Distribution used to approximate the posterior 
q = NormalisingFlow(d, d) 

# Neural network (outputs d summary statistics)
w = 128   
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, w, relu), Dense(w, d))
network = DeepSet(ψ, ϕ)

# Initialise the estimator
estimator = PosteriorEstimator(network, q)

# Train the estimator
estimator = train(estimator, sample, simulate, m = m)

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
function PosteriorEstimator(network, q::A) where A <: ApproximateDistribution
    return PosteriorEstimator(q, network)
end




#TODO maybe its better to not have a tuple, and just allow the arguments to be passed as normal... Just have to change DeepSet definition to allow two arguments in some places (this is more natural). Can easily allow backwards compat in this case too. 
@doc raw"""
	RatioEstimator <: NeuralEstimator
	RatioEstimator(network)
A neural estimator that estimates the likelihood-to-evidence ratio,
```math
r(\boldsymbol{Z}, \boldsymbol{\theta}) \equiv p(\boldsymbol{Z} \mid \boldsymbol{\theta})/p(\boldsymbol{Z}),
```
where $p(\boldsymbol{Z} \mid \boldsymbol{\theta})$ is the likelihood and $p(\boldsymbol{Z})$
is the marginal likelihood, also known as the model evidence.

For numerical stability, training is done on the log-scale using the relation 
$\log r(\boldsymbol{Z}, \boldsymbol{\theta}) = \text{logit}(c^*(\boldsymbol{Z}, \boldsymbol{\theta}))$, 
where $c^*(\cdot, \cdot)$ denotes the Bayes classifier as described in the [methodology](@ref "Neural ratio estimators") section. 
Hence, the neural network should be a mapping from $\mathcal{Z} \times \Theta$ to $\mathbb{R}$, 
where $\mathcal{Z}$ and $\Theta$ denote the sample and parameter spaces, respectively. 

!!! note "Network input"
    The neural network must implement a method `network(::Tuple)`, where the first element of the tuple contains the data sets and the second element contains the parameter matrices.  

When the neural network is a [`DeepSet`](@ref) (which implements the above method), two requirements must be met. First, the number of input neurons in the first layer of the outer network must equal $d$ plus the number of output neurons in the final layer of the inner network. Second, the number of output neurons in the final layer of the outer network must be one.

When applying the estimator to data, the log of the likelihood-to-evidence ratio is returned. 
The estimated ratio can then be used in various Bayesian
(e.g., [Hermans et al., 2020](https://proceedings.mlr.press/v119/hermans20a.html))
or frequentist
(e.g., [Walchessen et al., 2024](https://doi.org/10.1016/j.spasta.2024.100848))
inferential algorithms.

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d = 2     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 30    # number of independent replicates in each data set
sample(K) = rand32(d, K)
simulate(θ, m) = [ϑ[1] .+ ϑ[2] .* randn32(n, m) for ϑ in eachcol(θ)]

# Neural network
w = 128 
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w + d, w, relu), Dense(w, w, relu), Dense(w, 1))
network = DeepSet(ψ, ϕ)

# Initialise the estimator
r̂ = RatioEstimator(network)

# Train the estimator
r̂ = train(r̂, sample, simulate, m = m)

# Generate "observed" data 
θ = sample(1)
z = simulate(θ, 200)[1]

# Grid-based optimization and sampling
θ_grid = f32(expandgrid(0:0.01:1, 0:0.01:1))'  # fine gridding of the parameter space
estimate(r̂, z, θ_grid)                         # log of likelihood-to-evidence ratios
posteriormode(r̂, z; θ_grid = θ_grid)           # posterior mode 
sampleposterior(r̂, z; θ_grid = θ_grid)         # posterior samples

# Gradient-based optimization
using Optim
θ₀ = [0.5, 0.5]                                # initial estimate
posteriormode(r̂, z; θ₀ = θ₀)                   # posterior mode 
```
"""
struct RatioEstimator{N} <: NeuralEstimator
    network::N
end
function (estimator::RatioEstimator)(Z, θ; kwargs...)
    estimator((Z, θ); kwargs...) # "Tupleise" the input and pass to Tuple method
end
function (estimator::RatioEstimator)(Zθ::Tuple)
    logr = estimator.network(Zθ)
    if typeof(logr) <: AbstractVector
        logr = reduce(vcat, logr)
    end
    return logr
end

# function (estimator::RatioEstimator)(Zθ::Tuple; classifier::Bool = false)
#     c = σ(estimator.network(Zθ))
#     if typeof(c) <: AbstractVector
#         c = reduce(vcat, c)
#     end
#     classifier ? c : c ./ (1 .- c)
# end

# function (estimator::RatioEstimator)(Zθ::Tuple; return_log_ratio::Bool = true, return_classifier::Bool = false)
#     log_ratio = estimator.network(Zθ)
#     if return_log_ratio
#         return log_ratio
#     end

#     c = σ(log_ratio)
#     if typeof(c) <: AbstractVector
#         c = reduce(vcat, c)
#     end

#     return_classifier ? c : c ./ (1 .- c)
# end

# # Estimate ratio for many data sets and parameter vectors
# θ = sample(1000)
# Z = simulate(θ, m)
# r̂(Z, θ)                                   # log of the likelihood-to-evidence ratios



@doc raw"""
	PiecewiseEstimator <: NeuralEstimator
	PiecewiseEstimator(estimators::Vector{BayesEstimator}, changepoints::Vector{Integer})
Creates a piecewise estimator
([Sainsbury-Dale et al., 2024](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522), Sec. 2.2.2)
from a collection of neural Bayes `estimators` and sample-size `changepoints`.

This allows different estimators to be applied to different ranges of sample sizes. For instance, you may wish to use one estimator for small samples and another for larger ones. Given changepoints $m_1 < m_2 < \dots < m_{l-1}$, the piecewise estimator selects from $l$ trained estimators based on the observed sample size $m$ as follows:
```math
\hat{\boldsymbol{\theta}}(\boldsymbol{Z})
=
\begin{cases}
\hat{\boldsymbol{\theta}}_1(\boldsymbol{Z}) & m \leq m_1,\\
\hat{\boldsymbol{\theta}}_2(\boldsymbol{Z}) & m_1 < m \leq m_2,\\
\quad \vdots \\
\hat{\boldsymbol{\theta}}_l(\boldsymbol{Z}) & m > m_{l-1}.
\end{cases}
```
where $\hat{\boldsymbol{\theta}}_1(\cdot)$ is a neural Bayes estimator trained to be near-optimal over the range of sample sizes in which it is applied. 

Although this strategy requires training multiple neural networks, it is computationally efficient in practice when combined with pre-training (see [Sainsbury-Dale at al., 2024](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522), Sec 2.3.3), which can be automated using [`trainmultiple()`](@ref). 

# Examples
```julia
using NeuralEstimators, Flux

n = 2    # bivariate data
d = 3    # dimension of parameter vector 
w = 128  # width of each hidden layer

# Small-sample estimator
ψ₁ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₁ = Chain(Dense(w, w, relu), Dense(w, d));
θ̂₁ = PointEstimator(DeepSet(ψ₁, ϕ₁))

# Large-sample estimator
ψ₂ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₂ = Chain(Dense(w, w, relu), Dense(w, d));
θ̂₂ = PointEstimator(DeepSet(ψ₂, ϕ₂))

# Piecewise estimator with changepoint m=30
θ̂ = PiecewiseEstimator([θ̂₁, θ̂₂], 30)

# Apply the (untrained) piecewise estimator to data
Z = [rand(n, m) for m ∈ (10, 50)]
estimate(θ̂, Z)
```
"""
struct PiecewiseEstimator{E, C} <: NeuralEstimator
    estimators::E
    changepoints::C
    function PiecewiseEstimator(estimators, changepoints)
        if isa(changepoints, Number)
            changepoints = [changepoints]
        end
        @assert all(isinteger.(changepoints)) "`changepoints` should contain integers"
        if length(changepoints) != length(estimators) - 1
            error("The length of `changepoints` should be one fewer than the number of `estimators`")
        elseif !issorted(changepoints)
            error("`changepoints` should be in ascending order")
        else
            E = typeof(estimators)
            C = typeof(changepoints)
            new{E, C}(estimators, changepoints)
        end
    end
end
function (estimator::PiecewiseEstimator)(Z)
    changepoints = [estimator.changepoints..., Inf]
    m = numberreplicates(Z)
    θ̂ = map(eachindex(Z)) do i
        # find which estimator to use and then apply it
        mᵢ = m[i]
        j = findfirst(mᵢ .<= changepoints)
        estimator.estimators[j](Z[[i]])
    end
    return stackarrays(θ̂)
end
Base.show(io::IO, estimator::PiecewiseEstimator) = print(io, "\nPiecewise estimator with $(length(estimator.estimators)) estimators and sample size change-points: $(estimator.changepoints)")

"""
	Ensemble <: NeuralEstimator
	Ensemble(estimators)
	Ensemble(architecture::Function, J::Integer)
	(ensemble::Ensemble)(Z; aggr = mean)
Defines an ensemble of `estimators` which, when applied to data `Z`, returns the mean (or another summary defined by `aggr`) of the individual estimates (see, e.g., [Sainsbury-Dale et al., 2025, Sec. S5](https://doi.org/10.48550/arXiv.2501.04330)).

The ensemble can be initialised with a collection of trained `estimators` and then
applied immediately to observed data. Alternatively, the ensemble can be
initialised with a collection of untrained `estimators`
(or a function defining the architecture of each estimator, and the number of estimators in the ensemble),
trained with `train()`, and then applied to observed data. In the latter case, where the ensemble is trained directly,
if `savepath` is specified both the ensemble and component estimators will be saved.

Note that `train()` currently acts sequentially on the component estimators, using the `Adam` optimiser.

The ensemble components can be accessed by indexing the ensemble; the number of component estimators can be obtained using `length()`.

See also [`Parallel`](https://fluxml.ai/Flux.jl/stable/reference/models/layers/#Flux.Parallel), which can be used to mimic ensemble methods with an appropriately chosen `connection`. 

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|θ ~ N(θ, 1) with θ ~ N(0, 1)
d = 1     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 30    # number of independent replicates in each data set
sampler(K) = randn32(d, K)
simulator(θ, m) = [μ .+ randn32(n, m) for μ ∈ eachcol(θ)]

# Neural-network architecture of each ensemble component
function architecture()
	ψ = Chain(Dense(n, 64, relu), Dense(64, 64, relu))
	ϕ = Chain(Dense(64, 64, relu), Dense(64, d))
	network = DeepSet(ψ, ϕ)
	PointEstimator(network)
end

# Initialise ensemble with three component estimators 
ensemble = Ensemble(architecture, 3)
ensemble[1]      # access component estimators by indexing
ensemble[1:2]    # indexing with an iterable collection returns the corresponding ensemble 
length(ensemble) # number of component estimators

# Training
ensemble = train(ensemble, sampler, simulator, m = m, epochs = 5)

# Assessment
θ = sampler(1000)
Z = simulator(θ, m)
assessment = assess(ensemble, θ, Z)
rmse(assessment)

# Apply to data
ensemble(Z)
```
"""
struct Ensemble{T <: NeuralEstimator} <: NeuralEstimator
    estimators::Vector{T}
end
Ensemble(architecture::Function, J::Integer) = Ensemble([architecture() for j = 1:J])

function (ensemble::Ensemble)(Z; aggr = mean)
    # Collect each estimator’s output
    θ̂s = [estimator(Z) for estimator in ensemble.estimators]

    # Stack into 3D array (d × n × m) where m = number of estimators
    θ̂ = stackarrays(θ̂s, merge = false)

    # Aggregate elementwise
    if aggr === mean
        θ̂ = mean(θ̂; dims = 3)
    else
        #NB mapslices doesn't work with Zygote, so use mean as the default
        θ̂ = mapslices(aggr, cpu(θ̂); dims = 3)
    end

    return dropdims(θ̂; dims = 3)
end

Base.getindex(e::Ensemble, i::Integer) = e.estimators[i]
Base.getindex(e::Ensemble, indices::AbstractVector{<:Integer}) = Ensemble(e.estimators[indices])
Base.getindex(e::Ensemble, indices::UnitRange{<:Integer}) = Ensemble(e.estimators[indices])
Base.length(e::Ensemble) = length(e.estimators)
Base.eachindex(e::Ensemble) = eachindex(e.estimators)
Base.show(io::IO, ensemble::Ensemble) = print(io, "\nEnsemble with $(length(ensemble.estimators)) component estimators")
