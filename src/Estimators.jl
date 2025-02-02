"""
	NeuralEstimator

An abstract supertype for all neural estimators in `NeuralEstimators.jl`.
"""
abstract type NeuralEstimator end

"""
	BayesEstimator <: NeuralEstimator

An abstract supertype for neural Bayes estimators.
"""
abstract type BayesEstimator <: NeuralEstimator  end

"""
	PointEstimator <: BayesEstimator
    PointEstimator(network)
	(estimator::PointEstimator)(Z)
A point estimator, where the neural `network` is a mapping from the sample space to the parameter space.
"""
struct PointEstimator <: BayesEstimator
	network 
end
(estimator::PointEstimator)(Z) = estimator.network(Z)

#TODO Single shared summary statistic computation for efficiency
#TODO enforce probs ∈ (0, 1)
@doc raw"""
	IntervalEstimator <: BayesEstimator
	IntervalEstimator(u, v = u; probs = [0.025, 0.975], g::Function = exp)
	IntervalEstimator(u, c::Union{Function, Compress}; probs = [0.025, 0.975], g::Function = exp)
	IntervalEstimator(u, v, c::Union{Function, Compress}; probs = [0.025, 0.975], g::Function = exp)
	(estimator::IntervalEstimator)(Z)
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

The functions ``c_i(⋅)`` may be collectively defined by a ``d``-dimensional object of type
[`Compress`](@ref). If these functions are unspecified, they will be set to the
identity function so that the range of the intervals will be unrestricted. 
If only a single neural-network architecture is provided, it will be used for both ``\boldsymbol{u}(⋅)`` and ``\boldsymbol{v}(⋅)``.

The return value when applied to data using [`estimate`()](@ref) is a matrix with ``2d`` rows, where the first and second ``d`` rows correspond to the lower and upper bounds, respectively. The function [`interval()`](@ref) can be used to format this output in a readable ``d`` × 2 matrix.  

See also [`QuantileEstimatorDiscrete`](@ref) and
[`QuantileEstimatorContinuous`](@ref).

# Examples
```
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
struct IntervalEstimator{H} <: BayesEstimator
	u::DeepSet
	v::DeepSet
	c::Union{Function,Compress}
	probs::H
	g::Function
end
IntervalEstimator(u::DeepSet, v::DeepSet = u; probs = [0.025, 0.975], g = exp) = IntervalEstimator(deepcopy(u), deepcopy(v), identity, probs, g)
IntervalEstimator(u::DeepSet, c::Compress; probs = [0.025, 0.975], g = exp) = IntervalEstimator(deepcopy(u), deepcopy(u), c, probs, g)
IntervalEstimator(u::DeepSet, v::DeepSet, c::Compress; probs = [0.025, 0.975], g = exp) = IntervalEstimator(deepcopy(u), deepcopy(v), c, probs, g)
Flux.trainable(est::IntervalEstimator) = (u = est.u, v = est.v)
function (est::IntervalEstimator)(Z)
	bₗ = est.u(Z)                # lower bound
	bᵤ = bₗ .+ est.g.(est.v(Z))  # upper bound
	vcat(est.c(bₗ), est.c(bᵤ))
end

#TODO Single shared summary statistic computation for efficiency
#TODO improve print output
#TODO function for neat output as dxT matrix like interval() 
@doc raw"""
	QuantileEstimatorDiscrete <: BayesEstimator
	QuantileEstimatorDiscrete(v; probs = [0.05, 0.25, 0.5, 0.75, 0.95], g = Flux.softplus, i = nothing)
	(estimator::QuantileEstimatorDiscrete)(Z)
	(estimator::QuantileEstimatorDiscrete)(Z, θ₋ᵢ)

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

See also [`QuantileEstimatorContinuous`](@ref).

# Examples
```
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
τ = [0.05, 0.25, 0.5, 0.75, 0.95]
estimator = QuantileEstimatorDiscrete(v; probs = τ)

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
τ = [0.05, 0.25, 0.5, 0.75, 0.95]
q₁ = QuantileEstimatorDiscrete(v; probs = τ, i = 1)
q₂ = QuantileEstimatorDiscrete(v; probs = τ, i = 2)

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
struct QuantileEstimatorDiscrete{V, P} <: BayesEstimator
	v::V
	probs::P
	g::Union{Function, Nothing}
	i::Union{Integer, Nothing}
end
function QuantileEstimatorDiscrete(v; probs = [0.05, 0.25, 0.5, 0.75, 0.95], g = Flux.softplus, i::Union{Integer, Nothing} = nothing)
	if !isnothing(i) @assert i > 0 end
	QuantileEstimatorDiscrete(deepcopy.(repeat([v], length(probs))), probs, g, i)
end
Flux.trainable(est::QuantileEstimatorDiscrete) = (v = est.v, )
function (est::QuantileEstimatorDiscrete)(input) # input might be Z, or a tuple (Z, θ₋ᵢ)

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
function (est::QuantileEstimatorDiscrete)(Z, θ₋ᵢ::Vector)
	i = est.i
	@assert !isnothing(i) "slot i must be specified when approximating a full conditional"
	if isa(Z, Vector) # repeat θ₋ᵢ to match the number of data sets
		θ₋ᵢ = [θ₋ᵢ for _ in eachindex(Z)]
	end
	est((Z, θ₋ᵢ))  # "Tupleise" the input and apply the estimator
end
(est::QuantileEstimatorDiscrete)(Z, θ₋ᵢ::Number) = est(Z, [θ₋ᵢ])

# Assess the estimators
# using AlgebraOfGraphics, CairoMakie
# θ = sample(1000)
# Z = simulate(θ, m)
# assessment = assess([q₁, q₂], θ, Z, parameter_names = ["μ", "σ"])
# plot(assessment)

# function posterior(Z; μ₀ = 0, σ₀ = 1, σ² = 1)
# 	μ̃ = (1/σ₀^2 + length(Z)/σ²)^-1 * (μ₀/σ₀^2 + sum(Z)/σ²)
# 	σ̃ = sqrt((1/σ₀^2 + length(Z)/σ²)^-1)
# 	Normal(μ̃, σ̃)
# end

#TODO incorporate this into docs somewhere: It's based on the fact that a pair (θᵏ, Zᵏ) sampled as θᵏ ∼ p(θ), Zᵏ ~ p(Z ∣ θᵏ) is also a sample from θᵏ ∼ p(θ ∣ Zᵏ), Zᵏ ~ p(Z).
#TODO clarify output structure when we have multiple probability levels (what is the ordering in this case?)
@doc raw"""
	QuantileEstimatorContinuous <: BayesEstimator
	QuantileEstimatorContinuous(network; i = nothing, num_training_probs::Integer = 1)
	(estimator::QuantileEstimatorContinuous)(Z, τ)
	(estimator::QuantileEstimatorContinuous)(Z, θ₋ᵢ, τ)

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

See also [`QuantileEstimatorDiscrete`](@ref).

# Examples
```
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
struct QuantileEstimatorContinuous <: NeuralEstimator
	deepset::DeepSet #TODO remove ::DeepSet
	i::Union{Integer, Nothing}
end
function QuantileEstimatorContinuous(deepset::DeepSet; i::Union{Integer, Nothing} = nothing)
	if !isnothing(i) @assert i > 0 end
	QuantileEstimatorContinuous(deepset, i)
end
# core method (used internally)
(est::QuantileEstimatorContinuous)(tup::Tuple) = est.deepset(tup)
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
function (est::QuantileEstimatorContinuous)(Z::V, τ::Number) where V <: AbstractVector{A} where A
	est(Z, repeat([[τ]],  length(Z)))
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
	if !isa(q, Vector) q = [q] end
	reduce(hcat, permutedims.(q))
end
(est::QuantileEstimatorContinuous)(Z, θ₋ᵢ::Matrix, τ::Vector) = est(Z, θ₋ᵢ, permutedims(reduce(vcat, τ)))
(est::QuantileEstimatorContinuous)(Z, θ₋ᵢ::Matrix, τ::Number) = est(Z, θ₋ᵢ, repeat([τ], size(θ₋ᵢ, 2)))
(est::QuantileEstimatorContinuous)(Z, θ₋ᵢ::Vector, τ::Vector) = est(Z, reshape(θ₋ᵢ, :, 1), permutedims(τ))
(est::QuantileEstimatorContinuous)(Z, θ₋ᵢ::Vector, τ::Number) = est(Z, θ₋ᵢ, [τ])
(est::QuantileEstimatorContinuous)(Z, θ₋ᵢ::Number, τ::Number) = est(Z, [θ₋ᵢ], τ)
(est::QuantileEstimatorContinuous)(Z, θ₋ᵢ::Number, τ::Vector) = est(Z, [θ₋ᵢ], τ)

# # Closed-form posterior for comparison
# function posterior(Z; μ₀ = 0, σ₀ = 1, σ² = 1)

# 	# Parameters of posterior distribution
# 	μ̃ = (1/σ₀^2 + length(Z)/σ²)^-1 * (μ₀/σ₀^2 + sum(Z)/σ²)
# 	σ̃ = sqrt((1/σ₀^2 + length(Z)/σ²)^-1)

# 	# Posterior
# 	Normal(μ̃, σ̃)
# end

# # Estimate the posterior 0.1-quantile for 1000 test data sets
# τ = 0.1f0
# q̂(Z, τ)                        # neural quantiles
# quantile.(posterior.(Z), τ)'   # true quantiles

# # Estimate several quantiles for a single data set
# z = Z[1]
# τ = f32([0.1, 0.25, 0.5, 0.75, 0.9])
# q̂(z, τ')                     # neural quantiles (note that τ is given as row vector)
# quantile.(posterior(z), τ)   # true quantiles

@doc raw"""
	PosteriorEstimator <: NeuralEstimator
	PosteriorEstimator(q::ApproximateDistribution, network)
	sampleposterior(estimator::PosteriorEstimator, Z, N::Integer)
	posteriormean(estimator::PosteriorEstimator)
A neural estimator that approximates the posterior distribution $p(\boldsymbol{\theta} \mid \boldsymbol{Z})$. 

The neural `network` is a mapping from the sample space to a space that depends on the chosen approximate distribution `q` (see the available in-built [Approximate distributions](@ref)). 
Often, the output space of the neural network is the space $\mathcal{K}$ of approximate-distribution parameters $\boldsymbol{\kappa}$.  
However, for certain approximate distributions (notably, [`NormalisingFlow`](@ref)), the neural network should output summary statistics of some suitable dimension (e.g., the dimension $d$ of the parameter vector). 

# Examples
```
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

## Alternatively, to use a Gaussian approximate distribution: 
# q = GaussianDistribution(d) 
# w = 128
# ψ = Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu))
# ϕ = Chain(Dense(w, w, relu), Dense(w, w, relu), Dense(w, numdistributionalparams(q)))
# network = DeepSet(ψ, ϕ)

# Initialise the estimator
estimator = PosteriorEstimator(q, network)

# Train the estimator
estimator = train(estimator, sample, simulate, m = m)

# Inference with observed data 
θ = [0.8f0; 0.1f0]
Z = simulate(θ, m)
sampleposterior(estimator, Z) # posterior draws 
posteriormean(estimator, Z)   # point estimate
```
"""
struct PosteriorEstimator <: NeuralEstimator
	q::ApproximateDistribution
	network
end
numdistributionalparams(estimator::PosteriorEstimator) = numdistributionalparams(estimator.q)
logdensity(estimator::PosteriorEstimator, θ, Z) = logdensity(estimator.q, θ, estimator.network(Z)) 
(estimator::PosteriorEstimator)(Zθ::Tuple) = logdensity(estimator, Zθ[2], Zθ[1]) # internal method only used for convenience during training # TODO not ideal that we assume an ordering here
sampleposterior(estimator::PosteriorEstimator, Z, N::Integer = 1000) = sampleposterior(estimator.q, estimator.network(Z), N)

# <!-- There are also practical advantages to considering the likelihood-to-evidence ratio: for example, given conditionally (on $\boldsymbol{\theta}$) independent and identically distributed (iid) replicates $\boldsymbol{Z}_1, \dots, \boldsymbol{Z}_m$, the likelihood-to-evidence ratio is of the form $p(\boldsymbol{Z}_1, \dots, \boldsymbol{Z}_m \mid \boldsymbol{\theta}) / p(\boldsymbol{Z}_1, \dots, \boldsymbol{Z}_m) \propto \prod_{i=1}^m r(\boldsymbol{Z}_i, \boldsymbol{\theta})$, that is, a product of single-replicate likelihood-to-evidence ratios.  -->
@doc raw"""
	RatioEstimator <: NeuralEstimator
	RatioEstimator(network)
	(estimator::RatioEstimator)(Z, θ)
	sampleposterior(estimator::RatioEstimator, Z, N::Integer)
A neural estimator that estimates the likelihood-to-evidence ratio,
```math
r(\boldsymbol{Z}, \boldsymbol{\theta}) \equiv p(\boldsymbol{Z} \mid \boldsymbol{\theta})/p(\boldsymbol{Z}),
```
where $p(\boldsymbol{Z} \mid \boldsymbol{\theta})$ is the likelihood and $p(\boldsymbol{Z})$
is the marginal likelihood, also known as the model evidence.

For numerical stability, training is done on the log-scale using the relation 
$\log r(\boldsymbol{Z}, \boldsymbol{\theta}) = \text{logit}(c^*(\boldsymbol{Z}, \boldsymbol{\theta}))$, 
where $c^*(\cdot, \cdot)$ denotes the Bayes classifier as described in the [Methodology](@ref) section. 
Hence, the neural `network` should be a mapping from $\mathcal{Z} \times \Theta$ to $\mathbb{R}$, where $\mathcal{Z}$ and $\Theta$ denote the sample and parameter spaces, respectively. 

When the neural network is a [`DeepSet`](@ref), two requirements must be met. First, the number of input neurons in the first layer of
the outer network must equal $d$ plus the number of output neurons in the final layer of the inner network. 
Second, the number of output neurons in the final layer of the outer network must be one.

When applying the estimator to data `Z`, by default the likelihood-to-evidence ratio
$r(\boldsymbol{Z}, \boldsymbol{\theta})$ is returned (setting the keyword argument
`classifier = true` will yield class probability estimates). The estimated ratio
can then be used in various Bayesian
(e.g., [Hermans et al., 2020](https://proceedings.mlr.press/v119/hermans20a.html))
or frequentist
(e.g., [Walchessen et al., 2024](https://doi.org/10.1016/j.spasta.2024.100848))
inferential algorithms.

See also [`mlestimate`](@ref) and [`mapestimate`](@ref) for obtaining
approximate maximum-likelihood and maximum-a-posteriori estimates, and
[`sampleposterior`](@ref) for obtaining approximate posterior samples.

# Examples
```
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

# Inference with "observed" data (grid-based optimisation and sampling)
θ = sample(1)
z = simulate(θ, m)[1]
θ_grid = f32(expandgrid(0:0.01:1, 0:0.01:1))'  # fine gridding of the parameter space
r̂(z, θ_grid)                                   # likelihood-to-evidence ratios over grid
mlestimate(r̂, z; θ_grid = θ_grid)              # maximum-likelihood estimate
mapestimate(r̂, z; θ_grid = θ_grid)             # maximum-a-posteriori estimate
sampleposterior(r̂, z; θ_grid = θ_grid)         # posterior samples

# Inference with "observed" data (gradient-based optimisation using Optim.jl)
using Optim
θ₀ = [0.5, 0.5]                                # initial estimate
mlestimate(r̂, z; θ₀ = θ₀)                      # maximum-likelihood estimate
mapestimate(r̂, z; θ₀ = θ₀)                     # maximum-a-posteriori estimate
```
"""
struct RatioEstimator <: NeuralEstimator
	deepset::DeepSet #TODO remove ::DeepSet
end
function (estimator::RatioEstimator)(Z, θ; kwargs...)
	estimator((Z, θ); kwargs...) # "Tupleise" the input and pass to Tuple method
end
function (estimator::RatioEstimator)(Zθ::Tuple; classifier::Bool = false)
	c = σ(estimator.deepset(Zθ))
	if typeof(c) <: AbstractVector
		c = reduce(vcat, c)
	end
	classifier ? c : c ./ (1 .- c)
end

# # Estimate ratio for many data sets and parameter vectors
# θ = sample(1000)
# Z = simulate(θ, m)
# r̂(Z, θ)                                   # likelihood-to-evidence ratios
# r̂(Z, θ; classifier = true)                # class probabilities

# # Inference with multiple data sets
# θ = sample(10)
# z = simulate(θ, m)
# r̂(z, θ_grid)                                       # likelihood-to-evidence ratios
# mlestimate(r̂, z; θ_grid = θ_grid)                  # maximum-likelihood estimates
# mlestimate(r̂, z; θ₀ = θ₀)                          # maximum-likelihood estimates
# samples = sampleposterior(r̂, z; θ_grid = θ_grid)   # posterior samples
# θ̄ = reduce(hcat, mean.(samples; dims = 2))         # posterior means
# interval.(samples; probs = [0.05, 0.95])           # posterior credible intervals

@doc raw"""
	PiecewiseEstimator(estimators, changepoints)
Creates a piecewise estimator
([Sainsbury-Dale et al., 2024](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522), sec. 2.2.2)
from a collection of `estimators` and sample-size `changepoints`.

Specifically, with $l$ estimators and sample-size changepoints
$m_1 < m_2 < \dots < m_{l-1}$, the piecewise etimator takes the form,

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

For example, given an estimator ``\hat{\boldsymbol{\theta}}_1(\cdot)`` trained for small
sample sizes (e.g., ``m \leq 30``) and an estimator ``\hat{\boldsymbol{\theta}}_2(\cdot)``
trained for moderate-to-large sample sizes (e.g., ``m > 30``), one may construct a
`PiecewiseEstimator` that dispatches ``\hat{\boldsymbol{\theta}}_1(\cdot)`` if
``m \leq 30`` and ``\hat{\boldsymbol{\theta}}_2(\cdot)`` otherwise.

See also [`trainx()`](@ref) for training estimators for a range of sample sizes.

# Examples
```
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
struct PiecewiseEstimator <: NeuralEstimator
	estimators
	changepoints
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
			new(estimators, changepoints)
		end
	end
end
function (estimator::PiecewiseEstimator)(Z)
	# Note that this is an inefficient implementation, analogous to the inefficient
	# DeepSet implementation. A more efficient approach would be to subset Z based
	# on changepoints, apply the estimators to each block of Z, then combine the estimates.
	changepoints = [estimator.changepoints..., Inf]
	m = numberreplicates(Z)
	θ̂ = map(eachindex(Z)) do i
		# find which estimator to use, and then apply it
		mᵢ = m[i]
		j = findfirst(mᵢ .<= changepoints)
		estimator.estimators[j](Z[[i]])
	end
	return stackarrays(θ̂)
end
Base.show(io::IO, estimator::PiecewiseEstimator) = print(io, "\nPiecewise estimator with $(length(estimator.estimators)) estimators and sample size change-points: $(estimator.changepoints)")


# ---- Helper function for initialising an estimator ----

#TODO this is not very Julian, it would be better to have constructors for each estimator type. 
#     Can do this by splitting initialise_estimator() into a DeepSet constructor that takes `d` .
#     Should have initialise_estimator() as an internal function, and instead have the public API be based on constructors of the various estimator classes. This aligns more with the basic ideas of Julia, where functions returning a certain class should be made as a constructor rather than a separate function.

"""
    initialise_estimator(p::Integer; ...)
Initialise a neural estimator for a statistical model with `p` unknown parameters.

The estimator is couched in the DeepSets framework (see [`DeepSet`](@ref)) so
that it can be applied to data sets containing an arbitrary number of
independent replicates (including the special case of a single replicate).

Note also that the user is free to initialise their neural estimator however
they see fit using arbitrary `Flux` code; see
[here](https://fluxml.ai/Flux.jl/stable/models/layers/) for `Flux`'s API reference.

Finally, the method with positional argument `data_type`is a wrapper that allows
one to specify the type of their data (either "unstructured", "gridded", or
"irregular_spatial").

# Keyword arguments
- `architecture::String`: for unstructured multivariate data, one may use a fully-connected multilayer perceptron (`"MLP"`); for data collected over a grid, a convolutional neural network (`"CNN"`); and for graphical or irregular spatial data, a graphical neural network (`"GNN"`).
- `d::Integer = 1`: for unstructured multivariate data (i.e., when `architecture = "MLP"`), the dimension of the data (e.g., `d = 3` for trivariate data); otherwise, if `architecture ∈ ["CNN", "GNN"]`, the argument `d` controls the number of input channels (e.g., `d = 1` for univariate spatial processes).
- `estimator_type::String = "point"`: the type of estimator; either `"point"` or `"interval"`.
- `depth = 3`: the number of hidden layers; either a single integer or an integer vector of length two specifying the depth of the inner (summary) and outer (inference) network of the DeepSets framework.
- `width = 32`: a single integer or an integer vector of length `sum(depth)` specifying the width (or number of convolutional filters/channels) in each hidden layer.
- `activation::Function = relu`: the (non-linear) activation function of each hidden layer.
- `activation_output::Function = identity`: the activation function of the output layer.
- `variance_stabiliser::Union{Nothing, Function} = nothing`: a function that will be applied directly to the input, usually to stabilise the variance.
- `kernel_size = nothing`: (applicable only to CNNs) a vector of length `depth[1]` containing integer tuples of length `D`, where `D` is the dimension of the convolution (e.g., `D = 2` for two-dimensional convolution).
- `weight_by_distance::Bool = true`: (applicable only to GNNs) flag indicating whether the estimator will weight by spatial distance; if true, a `SpatialGraphConv` layer is used in the propagation module; otherwise, a regular `GraphConv` layer is used.
- `probs = [0.025, 0.975]`: (applicable only if `estimator_type = "interval"`) probability levels defining the lower and upper endpoints of the posterior credible interval.

# Examples
```
## MLP, GNN, 1D CNN, and 2D CNN for a statistical model with two parameters:
p = 2
initialise_estimator(p, architecture = "MLP")
initialise_estimator(p, architecture = "GNN")
initialise_estimator(p, architecture = "CNN", kernel_size = [10, 5, 3])
initialise_estimator(p, architecture = "CNN", kernel_size = [(10, 10), (5, 5), (3, 3)])
```
"""
function initialise_estimator(
    p::Integer;
	architecture::String,
    d::Integer = 1,
    estimator_type::String = "point",
    depth::Union{Integer, Vector{<:Integer}} = 3,
    width::Union{Integer, Vector{<:Integer}} = 32,
	variance_stabiliser::Union{Nothing, Function} = nothing,
    activation::Function = relu,
    activation_output::Function = identity,
    kernel_size = nothing,
	weight_by_distance::Bool = true,
	probs = [0.025, 0.975]
    )

	# "`kernel_size` should be a vector of integer tuples: see the documentation for details"
    @assert p > 0
    @assert d > 0
	@assert architecture ∈ ["MLP", "DNN", "CNN", "GNN"]
	if architecture == "DNN" architecture = "MLP" end # deprecation coercion
    @assert estimator_type ∈ ["point", "interval"]
    @assert all(depth .>= 0)
    @assert length(depth) == 1 || length(depth) == 2
	if isa(depth, Integer) depth = [depth] end
	if length(depth) == 1 depth = repeat(depth, 2) end
    @assert all(width .> 0)
    @assert length(width) == 1 || length(width) == sum(depth)
	if isa(width, Integer) width = [width] end
	if length(width) == 1 width = repeat(width, sum(depth)) end
	# henceforth, depth and width are integer vectors of length 2 and sum(depth), respectively

	if architecture == "CNN"
		@assert !isnothing(kernel_size) "The argument `kernel_size` must be provided when `architecture = 'CNN'`"
		@assert length(kernel_size) == depth[1]
		kernel_size = coercetotuple.(kernel_size)
	end

	L = sum(depth) # total number of hidden layers

	# inference network
	ϕ = []
	if depth[2] >= 1
		push!(ϕ, [Dense(width[l-1] => width[l], activation) for l ∈ (depth[1]+1):L]...)
	end
	push!(ϕ, Dense(width[L] => p, activation_output))
	ϕ = Chain(ϕ...)

	# summary network
	if architecture == "MLP"
		ψ = Chain(
			Dense(d => width[1], activation),
			[Dense(width[l-1] => width[l], activation) for l ∈ 2:depth[1]]...
			)
	elseif architecture == "CNN"
		ψ = Chain(
			Conv(kernel_size[1], d => width[1], activation),
			[Conv(kernel_size[l], width[l-1] => width[l], activation) for l ∈ 2:depth[1]]...,
			Flux.flatten
			)
	elseif architecture == "GNN"
		propagation = weight_by_distance ? SpatialGraphConv : GraphConv
		ψ = GNNChain(
			propagation(d => width[1], activation),
			[propagation(width[l-1] => width[l], activation) for l ∈ 2:depth[1]]...,
			GlobalPool(mean) # readout module
			)
	end

	if !isnothing(variance_stabiliser)
		if architecture ∈ ["MLP", "CNN"]
			ψ = Chain(variance_stabiliser, ψ...)
		elseif architecture == "GNN"
			ψ = GNNChain(variance_stabiliser, ψ...)
		end
	end

	θ̂ = DeepSet(ψ, ϕ)

	#TODO RatioEstimator, QuantileEstimatorDiscrete, QuantileEstimatorContinuous, PosteriorEstimator
	if estimator_type == "point"
		θ̂ = PointEstimator(θ̂)
	elseif estimator_type == "interval"
		θ̂ = IntervalEstimator(θ̂, θ̂; probs = probs)
	end

	return θ̂
end
coercetotuple(x) = (x...,)


# ---- Ensemble of estimators ----

#TODO Think about whether Parallel() might also be useful for ensembles (this might allow for faster computations, and immediate out-of-the-box integration with other parts of the package).

"""
	Ensemble(estimators)
	Ensemble(architecture::Function, J::Integer)
	(ensemble::Ensemble)(Z; aggr = median)

Defines an ensemble based on a collection of `estimators` which,
when applied to data `Z`, returns the median
(or another summary defined by `aggr`) of the estimates.

The ensemble can be initialised with a collection of trained `estimators` and then
applied immediately to observed data. Alternatively, the ensemble can be
initialised with a collection of untrained `estimators`
(or a function defining the architecture of each estimator, and the number of estimators in the ensemble),
trained with `train()`, and then applied to observed data. In the latter case, where the ensemble is trained directly,
if `savepath` is specified both the ensemble and component estimators will be saved.

Note that `train()` currently acts sequentially on the component estimators.

The ensemble components can be accessed by indexing the ensemble; the number
of component estimators can be obtained using `length()`.

# Examples
```
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
struct Ensemble <: NeuralEstimator
	estimators
end
Ensemble(architecture::Function, J::Integer) = Ensemble([architecture() for j in 1:J])

#TODO update savepath behaviour based on new default (with nothing)
function train(ensemble::Ensemble, args...; kwargs...)
	kwargs = (;kwargs...)
	savepath = haskey(kwargs, :savepath) ? kwargs.savepath : ""
	verbose  = haskey(kwargs, :verbose)  ? kwargs.verbose : true
	estimators = map(enumerate(ensemble.estimators)) do (i, estimator)
		verbose && @info "Training estimator $i of $(length(ensemble))"
		if savepath != "" # modify the savepath before passing it onto train
			kwargs = merge(kwargs, (savepath = joinpath(savepath, "estimator$i"),))
		end
		train(estimator, args...; kwargs...)
	end
	ensemble = Ensemble(estimators)

	if savepath != ""
		if !ispath(savepath) mkpath(savepath) end
		model_state = Flux.state(cpu(ensemble)) 
		@save joinpath(savepath, "ensemble.bson") model_state
	end

	return ensemble
end

function (ensemble::Ensemble)(Z; aggr = median)
	# Compute estimate from each estimator, yielding a vector of matrices
	# NB can be done in parallel, but I think the overhead will outweigh the benefit
	θ̂ = [estimator(Z) for estimator in ensemble.estimators]

	# Stack matrices along a new third dimension
	θ̂ = stackarrays(θ̂, merge = false) # equivalent to: θ̂ = cat(θ̂...; dims = 3)
	
	# aggregate elementwise 
	θ̂ = mapslices(aggr, cpu(θ̂); dims = 3) # NB mapslices doesn't work on the GPU, so transfer to CPU 
	θ̂ = dropdims(θ̂; dims = 3)

	return θ̂
end

# Overload Base functions
Base.getindex(e::Ensemble, i::Integer) = e.estimators[i]
Base.getindex(e::Ensemble, indices::AbstractVector{<:Integer}) = Ensemble(e.estimators[indices])
Base.getindex(e::Ensemble, indices::UnitRange{<:Integer}) = Ensemble(e.estimators[indices])
Base.length(e::Ensemble) = length(e.estimators)
Base.eachindex(e::Ensemble) = eachindex(e.estimators)
Base.show(io::IO, ensemble::Ensemble) = print(io, "\nEnsemble with $(length(ensemble.estimators)) component estimators")
