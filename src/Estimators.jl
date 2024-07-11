"""
	NeuralEstimator

An abstract supertype for neural estimators.
"""
abstract type NeuralEstimator end

#TODO I think all architectures below should be DeepSet objects.

# ---- PointEstimator  ----

"""
    PointEstimator(deepset::DeepSet)
A neural point estimator, a mapping from the sample space to the parameter space.

The estimator leverages the [`DeepSet`](@ref) architecture. The only
requirement is that number of output neurons in the final layer of the inference
network (i.e., the outer network) is equal to the number of parameters in the
statistical model.
"""
struct PointEstimator{F} <: NeuralEstimator
	arch::F # NB not sure why I can't replace F with DeepSet
	c::Union{Function,Compress}
	# PointEstimator(arch) = isa(arch, PointEstimator) ? error("Please do not construct PointEstimator objects with another PointEstimator") : new(arch)
end
PointEstimator(arch) = PointEstimator(arch, identity)
@layer PointEstimator
(est::PointEstimator)(Z) = est.c(est.arch(Z))
#NB don't bother documenting c since Compress layer can just be included in deepset

# ---- IntervalEstimator  ----

@doc raw"""
	IntervalEstimator(u, v = u; probs = [0.025, 0.975], g::Function = exp)
	IntervalEstimator(u, c::Union{Function,Compress}; probs = [0.025, 0.975], g::Function = exp)
	IntervalEstimator(u, v, c::Union{Function,Compress}; probs = [0.025, 0.975], g::Function = exp)

A neural interval estimator which, given data ``Z``, jointly estimates marginal
posterior credible intervals based on the probability levels `probs`.

The estimator employs a representation that prevents quantile crossing, namely,
it constructs marginal posterior credible intervals for each parameter
``\theta_i``, ``i = 1, \dots, p,``  of the form,
```math
[c_i(u_i(\boldsymbol{Z})), \;\; c_i(u_i(\boldsymbol{Z})) + g(v_i(\boldsymbol{Z})))],
```
where  ``\boldsymbol{u}(⋅) \equiv (u_1(\cdot), \dots, u_p(\cdot))'`` and
``\boldsymbol{v}(⋅) \equiv (v_1(\cdot), \dots, v_p(\cdot))'`` are neural networks
that transform data into ``p``-dimensional vectors; $g(\cdot)$ is a
monotonically increasing function (e.g., exponential or softplus); and each
``c_i(⋅)`` is a monotonically increasing function that maps its input to the
prior support of ``\theta_i``.

The functions ``c_i(⋅)`` may be defined by a ``p``-dimensional object of type
[`Compress`](@ref). If these functions are unspecified, they will be set to the
identity function so that the range of the intervals will be unrestricted.

If only a single neural-network architecture is provided, it will be used
for both ``\boldsymbol{u}(⋅)`` and ``\boldsymbol{v}(⋅)``.

The return value  when applied to data is a matrix with ``2p`` rows, where the
first and second ``p`` rows correspond to the lower and upper bounds, respectively.

See also [`QuantileEstimatorDiscrete`](@ref) and
[`QuantileEstimatorContinuous`](@ref).

# Examples
```
using NeuralEstimators, Flux

# Generate some toy data
n = 2   # bivariate data
m = 100 # number of independent replicates
Z = rand(n, m)

# prior
p = 3  # number of parameters in the statistical model
min_supp = [25, 0.5, -pi/2]
max_supp = [500, 2.5, 0]
g = Compress(min_supp, max_supp)

# Create an architecture
w = 8  # width of each layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ = Chain(Dense(w, w, relu), Dense(w, p));
u = DeepSet(ψ, ϕ)

# Initialise the interval estimator
estimator = IntervalEstimator(u, g)

# Apply the (untrained) interval estimator
estimator(Z)
interval(estimator, Z)
```
"""
struct IntervalEstimator{F, G, H} <: NeuralEstimator
	u::F
	v::G
	c::Union{Function,Compress}
	probs::H
	g::Function
end
IntervalEstimator(u, v = u; probs = [0.025, 0.975], g = exp) = IntervalEstimator(deepcopy(u), deepcopy(v), identity, probs, g)
IntervalEstimator(u, c::Compress; probs = [0.025, 0.975], g = exp) = IntervalEstimator(deepcopy(u), deepcopy(u), c, probs, g)
IntervalEstimator(u, v, c::Compress; probs = [0.025, 0.975], g = exp) = IntervalEstimator(deepcopy(u), deepcopy(v), c, probs, g)
@layer IntervalEstimator
Flux.trainable(est::IntervalEstimator) = (u = est.u, v = est.v)
function (est::IntervalEstimator)(Z)
	bₗ = est.u(Z)                # lower bound
	bᵤ = bₗ .+ est.g.(est.v(Z))  # upper bound
	vcat(est.c(bₗ), est.c(bᵤ))
end
#NB could rewrite this under-the-hood code in terms of QuantileEstimatorDiscrete to reduce code repetition


# ---- QuantileEstimatorDiscrete  ----

#TODO Single shared summary statistic computation for efficiency
# TODO improve print output

@doc raw"""
	QuantileEstimatorDiscrete(v::DeepSet; probs = [0.05, 0.25, 0.5, 0.75, 0.95], g = Flux.softplus, i = nothing)
	(estimator::QuantileEstimatorDiscrete)(Z)
	(estimator::QuantileEstimatorDiscrete)(Z, θ₋ᵢ)

A neural estimator that jointly estimates a fixed set of marginal posterior
quantiles with probability levels $\{\tau_1, \dots, \tau_T\}$, controlled by the
keyword argument `probs`.

By default, the estimator approximates the marginal quantiles for all parameters in the model, 
that is, the quantiles of 
```math
\theta_i \mid \boldsymbol{Z}
```
for parameters $\boldsymbol{\theta} \equiv (\theta_1, \dots, \theta_p)'$.
Alternatively, if initialised with `i` set to a positive integer, the estimator approximates the quantiles of
the full conditional distribution
```math
\theta_i \mid \boldsymbol{Z}, \boldsymbol{\theta}_{-i},
```
where $\boldsymbol{\theta}_{-i}$ denotes the parameter vector with its $i$th
element removed. For ease of exposition, when targetting marginal
posteriors of the form $\theta_i \mid \boldsymbol{Z}$ (i.e., the default behaviour),
we define $\text{dim}(\boldsymbol{\theta}_{-i}) ≡ 0$.

The estimator leverages the [`DeepSet`](@ref) architecture, subject to two
requirements. First, the number of input neurons in the first layer of the
inference network (i.e., the outer network) must be equal to the number of
neurons in the final layer of the summary network plus
$\text{dim}(\boldsymbol{\theta}_{-i})$. Second, the number of output neurons in
the final layer of the inference network must be equal to
$p - \text{dim}(\boldsymbol{\theta}_{-i})$. 
 The estimator employs a representation that prevents quantile crossing, namely,
```math
\begin{aligned}
\boldsymbol{q}^{(\tau_1)}(\boldsymbol{Z}) &= \boldsymbol{v}^{(\tau_1)}(\boldsymbol{Z}),\\
\boldsymbol{q}^{(\tau_t)}(\boldsymbol{Z}) &= \boldsymbol{v}^{(\tau_1)}(\boldsymbol{Z}) + \sum_{j=2}^t g(\boldsymbol{v}^{(\tau_j)}(\boldsymbol{Z})), \quad t = 2, \dots, T,
\end{aligned}
```
where $\boldsymbol{q}^{(\tau)}(\boldsymbol{Z})$ denotes the vector of $\tau$-quantiles for parameters $\boldsymbol{\theta} \equiv (\theta_1, \dots, \theta_p)'$, 
and $\boldsymbol{v}^{(\tau_t)}(\cdot)$, $t = 1, \dots, T$, are unconstrained neural
networks that transform data into $p$-dimensional vectors, and $g(\cdot)$ is a
non-negative function (e.g., exponential or softplus) applied elementwise to
its arguments. If `g=nothing`, the quantiles are estimated independently through the representation,  
```math
\boldsymbol{q}^{(\tau_t)}(\boldsymbol{Z}) = \boldsymbol{v}^{(\tau_t)}(\boldsymbol{Z}), \quad t = 1, \dots, T. 
```

The return value is a matrix with 
$(p - \text{dim}(\boldsymbol{\theta}_{-i})) \times T$ rows, where the
first set of ``T`` rows corresponds to the estimated quantiles for the first
parameter, the second set of ``T`` rows corresponds to the estimated quantiles
for the second parameter, and so on.

See also [`IntervalEstimator`](@ref) and
[`QuantileEstimatorContinuous`](@ref).

# Examples
```
using NeuralEstimators, Flux, Distributions

# Simple model Z|θ ~ N(θ, 1) with prior θ ~ N(0, 1)
d = 1   # dimension of each independent replicate
p = 1   # number of unknown parameters in the statistical model
m = 30  # number of independent replicates in each data set
prior(K) = randn32(p, K)
simulate(θ, m) = [μ .+ randn32(d, m) for μ ∈ eachcol(θ)]

# Architecture
ψ = Chain(Dense(d, 32, relu), Dense(32, 32, relu))
ϕ = Chain(Dense(32, 32, relu), Dense(32, p))
v = DeepSet(ψ, ϕ)

# Initialise the estimator
τ = [0.05, 0.25, 0.5, 0.75, 0.95]
q̂ = QuantileEstimatorDiscrete(v; probs = τ)

# Train the estimator
q̂ = train(q̂, prior, simulate, m = m)

# Closed-form posterior for comparison
function posterior(Z; μ₀ = 0, σ₀ = 1, σ² = 1)

	# Parameters of posterior distribution
	μ̃ = (1/σ₀^2 + length(Z)/σ²)^-1 * (μ₀/σ₀^2 + sum(Z)/σ²)
	σ̃ = sqrt((1/σ₀^2 + length(Z)/σ²)^-1)

	# Posterior
	Normal(μ̃, σ̃)
end

# Estimate posterior quantiles for 1000 test data sets
θ = prior(1000)
Z = simulate(θ, m)
q̂(Z)                                             # neural quantiles
reduce(hcat, quantile.(posterior.(Z), Ref(τ)))   # true quantiles


# ---- Full conditionals ----


# Simple model Z|μ,σ ~ N(μ, σ²) with μ ~ N(0, 1), σ ∼ IG(3,1)
d = 1         # dimension of each independent replicate
p = 2         # number of unknown parameters in the statistical model
m = 30        # number of independent replicates in each data set
function sample(K)
	μ = randn32(1, K)
	σ = rand(InverseGamma(3, 1), 1, K)
	θ = vcat(μ, σ)
end
simulate(θ, m) = θ[1] .+ θ[2] .* randn32(1, m)
simulate(θ::Matrix, m) = simulate.(eachcol(θ), m)

# Architecture
ψ = Chain(Dense(d, 32, relu), Dense(32, 32, relu))
ϕ = Chain(Dense(32 + 1, 32, relu), Dense(32, 1))
v = DeepSet(ψ, ϕ)

# Initialise estimators respectively targetting quantiles of μ∣Z,σ and σ∣Z,μ 
τ = [0.05, 0.25, 0.5, 0.75, 0.95]
q₁ = QuantileEstimatorDiscrete(v; probs = τ, i = 1)
q₂ = QuantileEstimatorDiscrete(v; probs = τ, i = 2)

# Train the estimators
q₁ = train(q₁, sample, simulate, m = m)
q₂ = train(q₂, sample, simulate, m = m)

# Estimate quantiles of μ∣Z,σ with σ = 0.5 and for 1000 data sets
θ = prior(1000)
Z = simulate(θ, m)    
θ₋ᵢ = 0.5f0 
q₁(Z, θ₋ᵢ)

# Can also apply to a single data set only 
q₁(Z[1], θ₋ᵢ)
```
"""
struct QuantileEstimatorDiscrete{V, P} <: NeuralEstimator
	v::V
	probs::P
	g::Union{Function, Nothing}
	i::Union{Integer, Nothing}
end
function QuantileEstimatorDiscrete(v; probs = [0.05, 0.25, 0.5, 0.75, 0.95], g = Flux.softplus, i::Union{Integer, Nothing} = nothing)
	if !isnothing(i) @assert i > 0 end 
	QuantileEstimatorDiscrete(deepcopy.(repeat([v], length(probs))), probs, g, i)
end
@layer QuantileEstimatorDiscrete
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



#TODO Closed-form posterior for full conditionals for comparison
@doc raw"""
	QuantileEstimatorContinuous(deepset::DeepSet; i = nothing, num_training_probs::Integer = 1)
	(estimator::QuantileEstimatorContinuous)(Z, τ)
	(estimator::QuantileEstimatorContinuous)(Z, θ₋ᵢ, τ)

A neural estimator targetting posterior quantiles.

Given as input data $\boldsymbol{Z}$ and the desired probability level
$\tau ∈ (0, 1)$, by default the estimator approximates the $\tau$-quantile of
```math
\theta_i \mid \boldsymbol{Z}
```
for parameters $\boldsymbol{\theta} \equiv (\theta_1, \dots, \theta_p)'$.
Alternatively, if initialised with `i` set to a positive integer, the estimator 
approximates the $\tau$-quantile of
the full conditional distribution
```math
\theta_i \mid \boldsymbol{Z}, \boldsymbol{\theta}_{-i},
```
where $\boldsymbol{\theta}_{-i}$ denotes the parameter vector with its $i$th
element removed. For ease of exposition, when targetting marginal
posteriors of the form $\theta_i \mid \boldsymbol{Z}$ (i.e., the default behaviour),
we define $\text{dim}(\boldsymbol{\theta}_{-i}) ≡ 0$.

The estimator leverages the [`DeepSet`](@ref) architecture, subject to two
requirements. First, the number of input neurons in the first layer of the
inference network (i.e., the outer network) must be equal to the number of
neurons in the final layer of the summary network plus
$1 + \text{dim}(\boldsymbol{\theta}_{-i})$. Second, the number of output neurons in
the final layer of the inference network must be equal to
$p - \text{dim}(\boldsymbol{\theta}_{-i})$.

Although not a requirement, one may employ a (partially) monotonic neural
network to prevent quantile crossing (i.e., to ensure that the
$\tau_1$-quantile does not exceed the $\tau_2$-quantile for any
$\tau_2 > \tau_1$). There are several ways to construct such a neural network:
one simple yet effective approach is to ensure that all weights associated with
$\tau$ are strictly positive
(see, e.g., [Cannon, 2018](https://link.springer.com/article/10.1007/s00477-018-1573-6)),
and this can be done using the [`DensePositive`](@ref) layer as illustrated in
the examples below.

The return value is a matrix with $p - \text{dim}(\boldsymbol{\theta}_{-i})$ rows,
corresponding to the estimated quantile for each parameter not in $\boldsymbol{\theta}_{-i}$.

See also [`QuantileEstimatorDiscrete`](@ref).

# Examples
```
using NeuralEstimators, Flux, Distributions, InvertedIndices, Statistics

# Simple model Z|θ ~ N(θ, 1) with prior θ ~ N(0, 1)
d = 1         # dimension of each independent replicate
p = 1         # number of unknown parameters in the statistical model
m = 30        # number of independent replicates in each data set
prior(K) = randn32(p, K)
simulateZ(θ, m) = [μ .+ randn32(d, m) for μ ∈ eachcol(θ)]
simulateτ(K)    = [rand32(10) for k in 1:K]  
simulate(θ, m)  = simulateZ(θ, m), simulateτ(size(θ, 2))

# Architecture: partially monotonic network to preclude quantile crossing
w = 64  # width of each hidden layer
ψ = Chain(
	Dense(d, w, relu),
	Dense(w, w, relu),
	Dense(w, w, relu)
	)
ϕ = Chain(
	DensePositive(Dense(w + 1, w, relu); last_only = true),
	DensePositive(Dense(w, w, relu)),
	DensePositive(Dense(w, p))
	)
deepset = DeepSet(ψ, ϕ)

# Initialise the estimator
q̂ = QuantileEstimatorContinuous(deepset)

# Train the estimator
q̂ = train(q̂, prior, simulate, m = m)

# Closed-form posterior for comparison
function posterior(Z; μ₀ = 0, σ₀ = 1, σ² = 1)

	# Parameters of posterior distribution
	μ̃ = (1/σ₀^2 + length(Z)/σ²)^-1 * (μ₀/σ₀^2 + sum(Z)/σ²)
	σ̃ = sqrt((1/σ₀^2 + length(Z)/σ²)^-1)

	# Posterior
	Normal(μ̃, σ̃)
end

# Estimate the posterior 0.1-quantile for 1000 test data sets
θ = prior(1000)
Z = simulateZ(θ, m)
τ = 0.1f0
q̂(Z, τ)                        # neural quantiles
quantile.(posterior.(Z), τ)'   # true quantiles

# Estimate several quantiles for a single data set
z = Z[1]
τ = Float32.([0.1, 0.25, 0.5, 0.75, 0.9])
q̂(z, τ')                     # neural quantiles (note that τ is given as row vector)
quantile.(posterior(z), τ)   # true quantiles


# ---- Full conditionals ----

# Simple model Z|μ,σ ~ N(μ, σ²) with μ ~ N(0, 1), σ ∼ IG(3,1)
d = 1         # dimension of each independent replicate
p = 2         # number of unknown parameters in the statistical model
m = 30        # number of independent replicates in each data set
function prior(K)
	μ = randn32(K)
	σ = rand(InverseGamma(3, 1), K)
	θ = hcat(μ, σ)'
	θ = Float32.(θ)
	return θ
end
simulateZ(θ, m) = θ[1] .+ θ[2] .* randn32(1, m)
simulateZ(θ::Matrix, m) = simulateZ.(eachcol(θ), m)
simulateτ(K)    = [rand32(10) for k in 1:K]
simulate(θ, m)  = simulateZ(θ, m), simulateτ(size(θ, 2))

# Architecture: partially monotonic network to preclude quantile crossing
w = 64  # width of each hidden layer
ψ = Chain(
	Dense(d, w, relu),
	Dense(w, w, relu),
	Dense(w, w, relu)
	)
ϕ = Chain(
	DensePositive(Dense(w + 2, w, relu); last_only = true),
	DensePositive(Dense(w, w, relu)),
	DensePositive(Dense(w, 1))
	)
deepset = DeepSet(ψ, ϕ)

# Initialise the estimator for the first parameter, targetting μ∣Z,σ
i = 1
q̂ = QuantileEstimatorContinuous(deepset; i = i)

# Train the estimator
q̂ = train(q̂, prior, simulate, m = m)

# Estimate quantiles of μ∣Z,σ with σ = 0.5 and for 1000 data sets
θ = prior(1000)
Z = simulateZ(θ, m)
θ₋ᵢ = 0.5f0    # for multiparameter scenarios, use θ[Not(i), :] to determine the order that the conditioned parameters should be given
τ = Float32.([0.1, 0.25, 0.5, 0.75, 0.9])
q̂(Z, θ₋ᵢ, τ)

# Estimate quantiles for a single data set
q̂(Z[1], θ₋ᵢ, τ)
```
"""
struct QuantileEstimatorContinuous <: NeuralEstimator
	deepset::DeepSet
	i::Union{Integer, Nothing}
end
function QuantileEstimatorContinuous(deepset::DeepSet; i::Union{Integer, Nothing} = nothing)
	if !isnothing(i) @assert i > 0 end
	QuantileEstimatorContinuous(deepset, i)
end
@layer QuantileEstimatorContinuous
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



# using NeuralEstimators, Flux, Distributions, InvertedIndices, Statistics
# # Simple model Z|μ,σ ~ N(μ, σ²) with μ ~ N(0, 1), σ ∼ IG(3,1)
# d = 1         # dimension of each independent replicate
# p = 3         # number of unknown parameters in the statistical model
# m = 30        # number of independent replicates in each data set
# function sample(K)
# 	μ = randn32(K)
# 	σ = rand(InverseGamma(3, 1), K)
# 	θ = hcat(μ, σ, σ)'
# 	θ = Float32.(θ)
# 	return θ
# end
# simulateZ(θ, m) = θ[1] .+ θ[2] .* randn32(1, m)
# simulateZ(θ::Matrix, m) = simulateZ.(eachcol(θ), m)
# simulateτ(K)    = [rand32(1) for k in 1:K]
# simulate(θ, m)  = simulateZ(θ, m), simulateτ(size(θ, 2))

# # Architecture: partially monotonic network to preclude quantile crossing
# w = 64  # width of each hidden layer
# ψ = Chain(
# 	Dense(d, w, relu),
# 	Dense(w, w, relu),
# 	Dense(w, w, relu)
# 	)
# ϕ = Chain(
# 	DensePositive(Dense(w + 3, w, relu); last_only = true),
# 	DensePositive(Dense(w, w, relu)),
# 	DensePositive(Dense(w, 1))
# 	)
# deepset = DeepSet(ψ, ϕ)

# # Initialise the estimator for the first parameter, targetting μ∣Z,σ
# i = 1
# q̂ = QuantileEstimatorContinuous(deepset; i = i)

# # Train the estimator
# q̂ = train(q̂, sample, simulate, m = m)

# # Estimate quantiles of μ∣Z,σ with σ = 0.5 and for 1000 data sets
# θ = sample(1000)
# Z = simulateZ(θ, m)
# θ₋ᵢ = [0.5f0, 0.6f0]    # for multiparameter scenarios, use θ[Not(i), :] to determine the order that the conditioned parameters should be given
# τ = Float32.([0.1, 0.25, 0.5, 0.75, 0.9])
# q̂(Z, θ₋ᵢ, τ)

# # Estimate quantiles for a single data set
# q̂(Z[1], θ₋ᵢ, τ)


# ---- RatioEstimator  ----

@doc raw"""
	RatioEstimator(deepset::DeepSet)

A neural estimator that estimates the likelihood-to-evidence ratio,
```math
r(\boldsymbol{Z}, \boldsymbol{\theta}) \equiv p(\boldsymbol{Z} \mid \boldsymbol{\theta})/p(\boldsymbol{Z}),
```
where $p(\boldsymbol{Z} \mid \boldsymbol{\theta})$ is the likelihood and $p(\boldsymbol{Z})$
is the marginal likelihood, also known as the model evidence.

The estimator leverages the [`DeepSet`](@ref) architecture, subject to two
requirements. First, the number of input neurons in the first layer of
the inference network (i.e., the outer network) must equal the number
of output neurons in the final layer of the summary network plus the number of
parameters in the statistical model. Second, the number of output neurons in the
final layer of the inference network must be equal to one.

The ratio estimator is trained by solving a relatively straightforward binary
classification problem. Specifically, consider the problem of distinguishing
dependent parameter--data pairs
${(\boldsymbol{\theta}', \boldsymbol{Z}')' \sim p(\boldsymbol{Z}, \boldsymbol{\theta})}$ with
class labels $Y=1$ from independent parameter--data pairs
${(\tilde{\boldsymbol{\theta}}', \tilde{\boldsymbol{Z}}')' \sim p(\boldsymbol{\theta})p(\boldsymbol{Z})}$
with class labels $Y=0$, and where the classes are balanced. Then the Bayes
classifier under binary cross-entropy loss is given by
```math
c(\boldsymbol{Z}, \boldsymbol{\theta}) = \frac{p(\boldsymbol{Z}, \boldsymbol{\theta})}{p(\boldsymbol{Z}, \boldsymbol{\theta}) + p(\boldsymbol{\theta})p(\boldsymbol{Z})},
```
and hence,
```math
r(\boldsymbol{Z}, \boldsymbol{\theta}) = \frac{c(\boldsymbol{Z}, \boldsymbol{\theta})}{1 - c(\boldsymbol{Z}, \boldsymbol{\theta})}.
```
For numerical stability, training is done on the log-scale using
$\log r(\boldsymbol{Z}, \boldsymbol{\theta}) = \text{logit}(c(\boldsymbol{Z}, \boldsymbol{\theta}))$.

When applying the estimator to data, by default the likelihood-to-evidence ratio
$r(\boldsymbol{Z}, \boldsymbol{\theta})$ is returned (setting the keyword argument
`classifier = true` will yield class probability estimates). The estimated ratio
can then be used in various downstream Bayesian
(e.g., [Hermans et al., 2020](https://proceedings.mlr.press/v119/hermans20a.html))
or Frequentist
(e.g., [Walchessen et al., 2023](https://arxiv.org/abs/2305.04634))
inferential algorithms.

See also [`mlestimate`](@ref) and [`mapestimate`](@ref) for obtaining
approximate maximum-likelihood and maximum-a-posteriori estimates, and
[`sampleposterior`](@ref) for obtaining approximate posterior samples.

# Examples
```
using NeuralEstimators, Flux, Statistics

# Generate data from Z|μ,σ ~ N(μ, σ²) with μ, σ ~ U(0, 1)
p = 2     # number of unknown parameters in the statistical model
d = 1     # dimension of each independent replicate
m = 100   # number of independent replicates

prior(K) = rand32(p, K)
simulate(θ, m) = θ[1] .+ θ[2] .* randn32(d, m)
simulate(θ::AbstractMatrix, m) = simulate.(eachcol(θ), m)

# Architecture
w = 64 # width of each hidden layer
ψ = Chain(
	Dense(d, w, relu),
	Dense(w, w, relu),
	Dense(w, q, relu)
	)
ϕ = Chain(
	Dense(w + p, w, relu),
	Dense(w, w, relu),
	Dense(w, 1)
	)
deepset = DeepSet(ψ, ϕ)

# Initialise the estimator
r̂ = RatioEstimator(deepset)

# Train the estimator
r̂ = train(r̂, prior, simulate, m = m)

# Inference with "observed" data set
θ = prior(1)
z = simulate(θ, m)[1]
θ₀ = [0.5, 0.5]                           # initial estimate
mlestimate(r̂, z;  θ₀ = θ₀)                # maximum-likelihood estimate
mapestimate(r̂, z; θ₀ = θ₀)                # maximum-a-posteriori estimate
θ_grid = expandgrid(0:0.01:1, 0:0.01:1)'  # fine gridding of the parameter space
θ_grid = Float32.(θ_grid)
r̂(z, θ_grid)                              # likelihood-to-evidence ratios over grid
sampleposterior(r̂, z; θ_grid = θ_grid)    # posterior samples
```
"""
struct RatioEstimator <: NeuralEstimator
	deepset::DeepSet
end
@layer RatioEstimator
function (est::RatioEstimator)(Z, θ; kwargs...)
	est((Z, θ); kwargs...) # "Tupleise" the input and pass to Tuple method
end
function (est::RatioEstimator)(Zθ::Tuple; classifier::Bool = false)
	c = σ(est.deepset(Zθ))
	if typeof(c) <: AbstractVector
		c = reduce(vcat, c)
	end
	classifier ? c : c ./ (1 .- c)
end

# # Estimate ratio for many data sets and parameter vectors
# θ = prior(1000)
# Z = simulate(θ, m)
# r̂(Z, θ)                                   # likelihood-to-evidence ratios
# r̂(Z, θ; classifier = true)                # class probabilities

# # Inference with multiple data sets
# θ = prior(10)
# z = simulate(θ, m)
# r̂(z, θ_grid)                                       # likelihood-to-evidence ratios
# mlestimate(r̂, z; θ_grid = θ_grid)                  # maximum-likelihood estimates
# mlestimate(r̂, z; θ₀ = θ₀)                          # maximum-likelihood estimates
# samples = sampleposterior(r̂, z; θ_grid = θ_grid)   # posterior samples
# θ̄ = reduce(hcat, mean.(samples; dims = 2))         # posterior means
# interval.(samples; probs = [0.05, 0.95])           # posterior credible intervals

# ---- PiecewiseEstimator ----

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

For example, given an estimator  ``\hat{\boldsymbol{\theta}}_1(\cdot)`` trained for small
sample sizes (e.g., m ≤ 30) and an estimator ``\hat{\boldsymbol{\theta}}_2(\cdot)``
trained for moderate-to-large sample sizes (e.g., m > 30), we may construct a
`PiecewiseEstimator` that dispatches ``\hat{\boldsymbol{\theta}}_1(\cdot)`` if
m ≤ 30 and ``\hat{\boldsymbol{\theta}}_2(\cdot)`` otherwise.

See also [`trainx()`](@ref) for training estimators for a range of sample sizes.

# Examples
```
using NeuralEstimators, Flux

d = 2  # bivariate data
p = 3  # number of parameters in the statistical model
w = 8  # width of each hidden layer

# Small-sample estimator
ψ₁ = Chain(Dense(d, w, relu), Dense(w, w, relu));
ϕ₁ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂₁ = PointEstimator(DeepSet(ψ₁, ϕ₁))

# Large-sample estimator
ψ₂ = Chain(Dense(d, w, relu), Dense(w, w, relu));
ϕ₂ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂₂ = PointEstimator(DeepSet(ψ₂, ϕ₂))

# Piecewise estimator with changepoint m=30
θ̂ = PiecewiseEstimator([θ̂₁, θ̂₂], 30)

# Apply the (untrained) piecewise estimator to data
Z = [rand(d, 1, m) for m ∈ (10, 50)]
θ̂(Z)
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
@layer PiecewiseEstimator
function (pe::PiecewiseEstimator)(Z)
	# Note that this is an inefficient implementation, analogous to the inefficient
	# DeepSet implementation. A more efficient approach would be to subset Z based
	# on changepoints, apply the estimators to each block of Z, then combine the estimates.
	changepoints = [pe.changepoints..., Inf]
	m = numberreplicates(Z)
	θ̂ = map(eachindex(Z)) do i
		# find which estimator to use, and then apply it
		mᵢ = m[i]
		j = findfirst(mᵢ .<= changepoints)
		pe.estimators[j](Z[[i]])
	end
	return stackarrays(θ̂)
end
Base.show(io::IO, pe::PiecewiseEstimator) = print(io, "\nPiecewise estimator with $(length(pe.estimators)) estimators and sample size change-points: $(pe.changepoints)")


# ---- Helper function for initialising an estimator ----

"""
    initialise_estimator(p::Integer; ...)
	initialise_estimator(p::Integer, data_type::String; ...)
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
- `architecture::String`: for unstructured multivariate data, one may use a densely-connected neural network (`"DNN"`); for data collected over a grid, a convolutional neural network (`"CNN"`); and for graphical or irregular spatial data, a graphical neural network (`"GNN"`).
- `d::Integer = 1`: for unstructured multivariate data (i.e., when `architecture = "DNN"`), the dimension of the data (e.g., `d = 3` for trivariate data); otherwise, if `architecture ∈ ["CNN", "GNN"]`, the argument `d` controls the number of input channels (e.g., `d = 1` for univariate spatial processes).
- `estimator_type::String = "point"`: the type of estimator; either `"point"` or `"interval"`.
- `depth = 3`: the number of hidden layers; either a single integer or an integer vector of length two specifying the depth of the inner (summary) and outer (inference) network of the DeepSets framework.
- `width = 32`: a single integer or an integer vector of length `sum(depth)` specifying the width (or number of convolutional filters/channels) in each hidden layer.
- `activation::Function = relu`: the (non-linear) activation function of each hidden layer.
- `activation_output::Function = identity`: the activation function of the output layer.
- `variance_stabiliser::Union{Nothing, Function} = nothing`: a function that will be applied directly to the input, usually to stabilise the variance.
- `kernel_size = nothing`: (applicable only to CNNs) a vector of length `depth[1]` containing integer tuples of length `D`, where `D` is the dimension of the convolution (e.g., `D = 2` for two-dimensional convolution).
- `weight_by_distance::Bool = false`: (applicable only to GNNs) flag indicating whether the estimator will weight by spatial distance; if true, a `SpatialGraphConv` layer is used in the propagation module; otherwise, a regular `GraphConv` layer is used.

# Examples
```
## DNN, GNN, 1D CNN, and 2D CNN for a statistical model with two parameters:
p = 2
initialise_estimator(p, architecture = "DNN")
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
	weight_by_distance::Bool = false #TODO why should this be false?
    )

	# "`kernel_size` should be a vector of integer tuples: see the documentation for details"
    @assert p > 0
    @assert d > 0
	@assert architecture ∈ ["DNN", "CNN", "GNN"]
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

	# mapping (outer) network
	# ϕ = []
	# if depth[2] > 1
	# 	push!(ϕ, [Dense(width[l-1] => width[l], activation) for l ∈ (depth[1]+1):(L-1)]...)
	# end
	# push!(ϕ, Dense(width[L-1] => p, activation_output))
	# ϕ = Chain(ϕ...)
	ϕ = []
	if depth[2] >= 1
		push!(ϕ, [Dense(width[l-1] => width[l], activation) for l ∈ (depth[1]+1):L]...)
	end
	push!(ϕ, Dense(width[L] => p, activation_output))
	ϕ = Chain(ϕ...)

	# summary (inner) network
	if architecture == "DNN" #TODO change DNN to MLP
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
		# propagation_module = GNNChain(
		# 	propagation(d => width[1], activation),
		# 	[propagation(width[l-1] => width[l], relu) for l ∈ 2:depth[1]]...
		# 	)
		# readout_module = GlobalPool(mean)
		# return GNN(propagation_module, readout_module, ϕ) # return more-efficient GNN object
		ψ = GNNChain(
			propagation(d => width[1], activation, aggr = mean),
			[propagation(width[l-1] => width[l], relu, aggr = mean) for l ∈ 2:depth[1]]...,
			GlobalPool(mean) # readout module
			)
	end

	if variance_stabiliser != nothing
		if architecture ∈ ["DNN", "CNN"]
			ψ = Chain(variance_stabiliser, ψ...)
		elseif architecture == "GNN"
			ψ = GNNChain(variance_stabiliser, ψ...)
		end
	end

	θ̂ = DeepSet(ψ, ϕ)

	if estimator_type == "interval"
		θ̂ = IntervalEstimator(θ̂, θ̂)
	end

	return θ̂
end

function initialise_estimator(p::Integer, data_type::String; args...)
	data_type = lowercase(data_type)
	@assert data_type ∈ ["unstructured", "gridded", "irregular_spatial"]
	architecture = if data_type == "unstructured"
		"DNN"
	elseif data_type == "gridded"
		"CNN"
	elseif data_type == "irregular_spatial"
		"GNN"
	end
	initialise_estimator(p; architecture = architecture, args...)
end

coercetotuple(x) = (x...,)
