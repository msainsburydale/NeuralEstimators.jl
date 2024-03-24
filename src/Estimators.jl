"""
	NeuralEstimator

An abstract supertype for neural estimators.
"""
abstract type NeuralEstimator end

# ---- PointEstimator  ----

#TODO document g (and change it to c)... wait, why do we even need this when Compress can just be added to the network?
"""
    PointEstimator(arch)

A neural point estimator, that is, a mapping from the sample space to the
parameter space, defined by the given neural-network architecture `arch`.
"""
struct PointEstimator{F} <: NeuralEstimator
	arch::F
	g::Union{Function,Compress}
	# PointEstimator(arch) = isa(arch, PointEstimator) ? error("Please do not construct PointEstimator objects with another PointEstimator") : new(arch)
end
PointEstimator(arch) = PointEstimator(arch, identity)
@layer PointEstimator
(est::PointEstimator)(Z) = est.g(est.arch(Z))

# ---- IntervalEstimator  ----

#TODO change documentation to use c instead of g for compress
#TODO allow keyword argument g in the same way as QuantileEstimator
#TODO seealso: QuantileEstimator (and acknowledge that this is a special case).
"""
	IntervalEstimator(u, v = u; probs = [0.025, 0.975])
	IntervalEstimator(u, g::Compress; probs = [0.025, 0.975])
	IntervalEstimator(u, v, g::Compress; probs = [0.025, 0.975])

A neural interval estimator which, given data ``Z``, jointly estimates credible
intervals based on the probability levels `probs` of the form,

```math
[g(u(Z)), 	g(u(Z)) + \\mathrm{exp}(v(Z)))],
```

where

- ``u(⋅)`` and ``v(⋅)`` are neural networks, both of which should transform data into ``p``-dimensional vectors (with ``p`` the number of parameters in the statistical model);
- ``g(⋅)`` is either the identity function or a logistic function that maps its input to the prior support.

The prior support may be defined by a ``p``-dimensional object of type
[`Compress`](@ref). Otherwise, the range of the intervals will be unrestricted
(i.e., ``g(⋅)`` will be the identity function).

Note that, in addition to ensuring that the interval remains in the prior support,
this construction also ensures that the intervals are valid (i.e., it prevents
quantile crossing, in the sense that the upper bound is always greater than the
lower bound).

If only a single neural-network architecture is provided, it will be used
for both `u` and `v`.

The return value is a matrix with ``2p`` rows, where the first and second ``p``
rows correspond to the lower and upper bounds, respectively.

# Examples
```
using NeuralEstimators
using Flux

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
v = deepcopy(u) # use the same architecture for both u and v

# Initialise the interval estimator
estimator = IntervalEstimator(u, v, g)

# Apply the (untrained) interval estimator
estimator(Z)
interval(estimator, Z)
```
"""
struct IntervalEstimator{F, G, H} <: NeuralEstimator
	u::F
	v::G
	g::Union{Function,Compress}
	probs::H
	# note that Flux warns against the use of inner constructors: see https://fluxml.ai/Flux.jl/stable/models/basics/#Flux.@layer
end
IntervalEstimator(u, v = u; probs = [0.025, 0.975]) = IntervalEstimator(deepcopy(u), deepcopy(v), identity, probs)
IntervalEstimator(u, g::Compress; probs = [0.025, 0.975]) = IntervalEstimator(deepcopy(u), deepcopy(u), g, probs)
IntervalEstimator(u, v, g::Compress; probs = [0.025, 0.975]) = IntervalEstimator(deepcopy(u), deepcopy(v), g, probs)
@layer IntervalEstimator
Flux.trainable(est::IntervalEstimator) = (u = est.u, v = est.v)
function (est::IntervalEstimator)(Z)
	bₗ = est.u(Z)              # lower bound
	bᵤ = bₗ .+ exp.(est.v(Z))  # upper bound
	vcat(est.g(bₗ), est.g(bᵤ))
end

# ---- QuantileEstimatorDiscrete  ----

@doc raw"""
	QuantileEstimatorDiscrete(v; probs = [0.05, 0.25, 0.5, 0.75, 0.95], g = Flux.softplus)

A neural estimator that jointly estimates a fixed set of marginal posterior
quantiles with probability levels $\{\tau_1, \dots, \tau_T\}$, controlled by the
keyword argument `probs`.

The estimator employs a representation that prevents quantile crossing, namely,

```math
\begin{aligned}
\hat{\mathbf{q}}^{(\tau_1)}(\mathbf{Z}) &= \mathbf{v}^{(\tau_1)}(\mathbf{Z}),\\
\hat{\mathbf{q}}^{(\tau_t)}(\mathbf{Z}) &= \mathbf{v}^{(\tau_1)}(\mathbf{Z}) + \sum_{j=2}^t g(\mathbf{v}^{(\tau_j)}(\mathbf{Z})), \quad t = 2, \dots, T,
\end{aligned}
```
where $\mathbf{v}^{(\tau_t)}(\cdot)$, $t = 1, \dots, T$, are unconstrained neural
networks that transform data into $p$-dimensional vectors, and $g(\cdot)$ is a
monotonically increasing function (e.g., exponential or softplus) applied
elementwise to its arguments. In this implementation, the same neural-network
architecture `v` is used for each $\mathbf{v}^{(\tau_t)}(\cdot)$, $t = 1, \dots, T$.

The return value is a matrix with ``pT`` rows, where the first set of ``T``
rows corresponds to the estimated quantiles for the first parameter, the second
set of ``T`` rows corresponds to the estimated quantiles for the second
parameter, and so on.

# Examples
```
using NeuralEstimators
using Distributions
using Flux

# Generate data from the model Z|θ ~ N(θ, 1) and θ ~ N(0, 1)
p = 1       # number of unknown parameters in the statistical model
m = 30      # number of independent replicates
d = 1       # dimension of each independent replicate
K = 30000   # number of training samples
prior(K) = randn(Float32, 1, K)
simulate(θ, m) = [μ .+ randn(Float32, 1, m) for μ ∈ eachcol(θ_train)]
θ_train = prior(K)
θ_val   = prior(K)
Z_train = simulate(θ_train, m)
Z_val   = simulate(θ_val, m)

# Architecture
ψ = Chain(Dense(d, 32, relu), Dense(32, 32, relu))
ϕ = Chain(Dense(32, 32, relu), Dense(32, p))
v = DeepSet(ψ, ϕ)

# Initialise the estimator
θ̂ = QuantileEstimatorDiscrete(v)

# Train the estimator
train(θ̂, θ_train, θ_val, Z_train, Z_val)

# Use the estimator with test data
θ = prior(K)
Z = simulate(θ, m)
θ̂(Z)

# Compare to the closed-form posterior
function posterior(Z)
	# Prior hyperparameters
	σ₀  = 1
	σ₀² = σ₀^2
	μ₀  = 0

	# Known variance
    σ  = 1
	σ² = σ^2

	# Posterior parameters
	μ̃ = (1/σ₀² + length(Z)/σ²)^-1 * (μ₀/σ₀² + sum(Z)/σ²)
	σ̃ = sqrt((1/σ₀² + length(Z)/σ²)^-1)

	# Posterior
	Normal(μ̃, σ̃)
end
true_quantiles = quantile.(posterior.(Z), Ref([0.05, 0.25, 0.5, 0.75, 0.95]))
true_quantiles = reduce(hcat, true_quantiles)

# Compare estimates to true values
θ̂(Z) - true_quantiles
```
"""
struct QuantileEstimatorDiscrete{V, P} <: NeuralEstimator
	v::V
	probs::P
	g::Function
	# note that Flux warns against the use of inner constructors: see https://fluxml.ai/Flux.jl/stable/models/basics/#Flux.@layer
end
QuantileEstimatorDiscrete(v; probs = [0.05, 0.25, 0.5, 0.75, 0.95], g = Flux.softplus) = QuantileEstimatorDiscrete(deepcopy.(repeat([v], length(probs))), probs, g)
@layer QuantileEstimatorDiscrete
Flux.trainable(est::QuantileEstimatorDiscrete) = (v = est.v, )
function (est::QuantileEstimatorDiscrete)(Z)

	# Apply each neural network to Z
	v = map(est.v) do v
		v(Z)
	end

	# Simple approach: does not ensure monotonicity
	# return vcat(q...)

	# Monotonic approach:
	# Apply the monotonically increasing transformation to all but the first result
	gv = broadcast.(est.g, v[2:end])
	# Combine
	q = [v[1], gv...]
	reduce(vcat, cumsum(q))
end
#TODO In the docs example, why does the risk increase during training?
#     Is it because I'm computing the risk in different ways before vs. during the training loop?

"""
	DensePositive(layer::Dense, g::Function)
	DensePositive(layer::Dense; g::Function = Flux.relu)
Same as standard
[dense layer](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense)
but ensures positive weights (biases are unconstrained).

# Examples
```
using NeuralEstimators
using Flux

layer = DensePositive(Dense(5 => 2))
x = rand32(5, 64)
layer(x)
```
"""
struct DensePositive
	layer::Dense
	g::Function
	last_only::Bool
end
DensePositive(layer::Dense; g::Function = Flux.relu, last_only::Bool = false) = DensePositive(layer, g, last_only)
@layer DensePositive
# Simple version of forward pass:
# (d::DensePositive)(x) = d.layer.σ.(Flux.softplus(d.layer.weight) * x .+ d.layer.bias)
# Complex version of forward pass based on Flux's Dense code:
function (d::DensePositive)(x::AbstractVecOrMat)
  a = d.layer # extract the underlying fully-connected layer
  _size_check(a, x, 1 => size(a.weight, 2))
  σ = NNlib.fast_act(a.σ, x) # replaces tanh => tanh_fast, etc
  xT = _match_eltype(a, x)   # fixes Float64 input, etc.
  if d.last_only
	  weight = d.g.(hcat(a.weight[:, 1:end-1], a.weight[:, end:end]))
  else
	  weight = d.g.(a.weight)
  end
  σ.(weight * xT .+ a.bias)
end
function (a::DensePositive)(x::AbstractArray)
  a = d.layer # extract the underlying fully-connected layer
  _size_check(a, x, 1 => size(a.weight, 2))
  reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
end

#TODO test with bivariate example (e.g., normal unknown mean and variance)
#TODO add note in docs here and QuantileEstimatorDiscrete that one may also use
#     a simple PointEstimator and the quantileloss to target a specific quantile
#TODO unit testing and documentation
#TODO describe more the architecture (dimension of input, dimension of output, the use of DensePositive)
#TODO add reference to Cannon (2018) and others
#TODO write out the maths:
# The estimator prevents quantile crossing by employing a functional form ensures
# monotonicity with respect to $\tau$, namely,
#
# ```math
# \begin{aligned}
# ...
# \end{aligned}
# ```
# where ...
@doc raw"""
	QuantileEstimator(deepset::DeepSet)

A neural estimator that estimates the marginal posterior quantiles given as
input the desired probability level $\tau ∈ (0, 1)$.

The construction is based on the [`DeepSet`](@ref) architecture. The only
requirement is that number of neurons in the first layer of the inference
network (also known as the outer network) is one greater than the number of
neurons in the final layer of the summary network.

The return value is a matrix with ``p`` rows, corresponding to the estimated
marginal posterior quantiles for each parameter in the statistical model.

# Examples
```
using NeuralEstimators
using Distributions
using Flux

# Generate data from the model Z|θ ~ N(θ, 1) and θ ~ N(0, 1)
d = 1       # dimension of each independent replicate
m = 30      # number of independent replicates
p = 1       # number of unknown parameters in the statistical model
K = 100000  # number of training samples
prior(K) = randn32(1, K)
simulateZ(θ, m) = [μ .+ randn32(1, m) for μ ∈ eachcol(θ_train)]
simulateτ(K) = [rand32(1) for _ in 1:K]
simulate(θ, m) = (simulateZ(θ, m), simulateτ(size(θ, 2)))
θ_train  = prior(K)
θ_val    = prior(K)
Zτ_train = simulate(θ_train, m)
Zτ_val   = simulate(θ_val, m)

# Architecture
w = 128
q = 3
ψ = Chain(
	Dense(d, w, relu),
	Dropout(0.3),
	Dense(w, w, relu),
	Dropout(0.3),
	Dense(w, q, relu)
	)
ϕ = Chain(
	DensePositive(Dense(q + 1, w, relu); last_only = true),
	DensePositive(Dense(w, w, relu)),
	Dropout(0.3),
	DensePositive(Dense(w, p))
	)
deepset = DeepSet(ψ, ϕ)

# Initialise the estimator
θ̂ = QuantileEstimator(deepset)

# Train the estimator
train(θ̂, θ_train, θ_val, Zτ_train, Zτ_val) # training with fixed instances

# Use the estimator with test data
θ = prior(K)
Z = simulateZ(θ, m)
τ = 0.1f0
θ̂(Z[1], τ)
θ̂(Z, τ)

# Compare to closed-form posterior
function posterior(Z)
	# Prior hyperparameters
	σ₀  = 1
	σ₀² = σ₀^2
	μ₀  = 0

	# Known variance
    σ  = 1
	σ² = σ^2

	# Posterior parameters
	μ̃ = (1/σ₀² + length(Z)/σ²)^-1 * (μ₀/σ₀² + sum(Z)/σ²)
	σ̃ = sqrt((1/σ₀² + length(Z)/σ²)^-1)

	# Posterior
	Normal(μ̃, σ̃)
end

# Many data sets, single quantile:
quantile.(posterior.(Z), τ)'
θ̂(Z, τ)

# Single data set, range of quantiles:
τ = [0.1, 0.25, 0.5, 0.75, 0.9]
quantile.(posterior(Z[1]), τ)'
reduce(hcat, θ̂.(Ref(Z[1]), τ))
```
"""
struct QuantileEstimator <: NeuralEstimator
	deepset::DeepSet
end
@layer QuantileEstimator
function (est::QuantileEstimator)(Z::A, τ::Number) where A
	est(Z, [τ])
end
function (est::QuantileEstimator)(Z::V, τ::Number) where V <: AbstractVector{A} where A
	est(Z, repeat([[τ]], length(Z)))
end
(est::QuantileEstimator)(Z, τ) = est((Z, τ)) # "Tupleise" input and pass to Tuple method
(est::QuantileEstimator)(Zτ::Tuple) = est.deepset(Zτ)
#TODO why is the validation risk so large (why is there so much overfitting, at least as measured by validation risk)?
#TODO train(θ̂, prior, simulate, m = m)  # training with "on-the-fly" simulation
#TODO why is the initial training risk different to the previous final training risk?
# train(θ̂, θ_train, θ_train, Zτ_train, Zτ_train)
# train(θ̂, θ_train, θ_train, Zτ_train, Zτ_train)
#TODO add the following to unit testing:
# # Check monotonicty
# using Test
# @test all(θ̂(Z, 0.1f0) .<= θ̂(Z, 0.11f0) .<= θ̂(Z, 0.9f0) .<= θ̂(Z, 0.91f0))

# ---- RatioEstimator  ----

#TODO unit testing
#TODO add more mathematical details from the ARSIA paper. See also the docs for LAMPE for a nice concise way to present the approach: https://github.com/probabilists/lampe/blob/master/lampe/inference/nre.py
#TODO show the example with m = 30 replicates
#TODO add key references (Cranmer, Hermans, Walchessen)
#TODO need to properly describe the learning task, so that "class probability" makes sense
@doc raw"""
	RatioEstimator(deepset::DeepSet)

A neural estimator that estimates the likelihood-to-evidence ratio,

```math
r(\mathbf{Z}, \mathbf{\theta}) \equiv p(\mathbf{Z} \mid \mathbf{\theta})/p(\mathbf{Z}),
```

where $p(\mathbf{Z} \mid \mathbf{\theta})$ is the likelihood and $p(\mathbf{Z})$
is the marginal likelihood, also known as the model evidence.

The construction is based on the [`DeepSet`](@ref) architecture, which is
subject to two requirements. First, the number of neurons in the first layer of
the inference network (also known as the outer network) is equal to the number
of neurons in the final layer of the summary network plus the number of
parameters in the statistical model (i.e., the dimension of $\mathbf{\theta}$).
Second, the number of neurons in the final layer of the inference network must
be equal to one.

For numerical stability, training is done on the log-scale using
$\log r(\mathbf{Z}, \mathbf{\theta}) = \text{logit}(c(\mathbf{Z}, \mathbf{\theta}))$.

Given $K$ data sets, the return value is a $1\times K$ matrix which by default
contains estimates of the likelihood-to-evidence ratio $r(\cdot, \cdot)$.
Alternatively, setting `classifier=true` will yield the corresponding class
probability estimates, $c(\cdot, \cdot) = \frac{r(\cdot, \cdot)}{1 + r(\cdot, \cdot)}$.

# Examples
```
using NeuralEstimators
using Distributions
using Flux

# Generate data from Z|θ ~ N(θ, s²) with θ ~ U(0, 1) and s = 0.2 known
d = 1        # dimension of each independent replicate
m = 1        # number of independent replicates
p = 1        # number of unknown parameters in the statistical model
K = 100000   # number of training samples
s = 0.2f0    # known standard deviation

prior(K) = rand32(p, K)
simulate(θ, m) = [μ .+ s * randn(Float32, 1, m) for μ ∈ eachcol(θ_train)]

# prior(K) = rand32(p, K)
# simulate(θ, m) = θ[1] .+ θ[2] .* randn32(1, m)
# simulate(θ::AbstractMatrix, m) = [simulate(x, m) for x ∈ eachcol(θ)]

θ_train = prior(K)
θ_val   = prior(K)
Z_train = simulate(θ_train, m)
Z_val   = simulate(θ_val, m)

# Architecture
w = 64
ψ = Chain(
	Dense(d, w, relu),
	Dropout(0.3),
	Dense(w, w, relu),
	Dropout(0.3),
	Dense(w, w, relu),
	Dropout(0.3)
	)
ϕ = Chain(
	Dense(w + p, w, relu),
	Dropout(0.3),
	Dense(w, w, relu),
	Dropout(0.3),
	Dense(w, 1)
	)
deepset = DeepSet(ψ, ϕ)

# Initialise the estimator
r̂ = RatioEstimator(deepset)

# Train the estimator
r̂ = train(r̂, θ_train, θ_val, Z_train, Z_val)

# Use the estimator with test data
θ = prior(K)
Z = simulate(θ, m)
r̂(Z, θ)                      # likelihood-to-evidence ratio estimates
r̂(Z, θ; classifier = true)   # class probability estimates

# Compare to closed-form solution
function r(Z, θ; classifier = false)
	# assumes a single independent replicate in Z and θ ~ U(0, 1)
	likelihood = pdf(Normal(θ, s), Z)
	evidence = cdf(Normal(Z, s), 1) - cdf(Normal(Z, s), 0)
	c = likelihood / (likelihood + evidence)
	classifier ? c : c ./ (1 .- c)
end
[r(Z[k][1], θ[k]) for k ∈ 1:K]'                     # true likelihood-to-evidence ratios
[r(Z[k][1], θ[k]; classifier = true) for k ∈ 1:K]'  # true class probabilities

# Sampling
theta_grid = permutedims(range(0.0f0,0.6f0, 750))
sample(r̂, Z[1]; theta_grid = theta_grid)
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
	classifier ? c : c ./ (1 .- c)
end

# NB this will be easily extended to NLE and NPE (whatever methods offer a posterior density)
using StatsBase: wsample
function sample(est::RatioEstimator, Z, N::Integer = 1000; method::String = "IS", theta_grid = nothing)
	# We only need to evaluate the summary-statistic network once (this can lead
	# to enormous savings, e.g., when theta_grid is extremely large)
	# TODO would be better to have a summary(θ̂::DeepSet, Z) that applies the
	# summary network to Z and returns the summary statistics
	t = est.deepset.a(est.deepset.ψ(Z))
	if !isnothing(est.deepset.S)
		s = d.S(Z)
		t = vcat(t, s)
	end

	if method == "IS"
		# Example code:
		# theta_grid = permutedims(expandgrid(range(0.0,0.6, 750), range(0.5, 3.0, 750)))
		# theta_grid = permutedims(range(0.0f0,0.6f0, 750))
		tθ = vcat(repeat(t, 1, size(theta_grid, 2)), theta_grid)
		c = σ(est.deepset.ϕ(tθ))
		r = c ./ (1 .- c) # Santiy check: reduce(vcat, est.(Ref(Z), eachcol(theta_grid)))
		density = vec(r) # TODO In the general case (non-uniform prior), we also need to multiply the density by the prior distribution
		StatsBase.wsample(theta_grid, density, N; replace = true)
	else
		@error "Only `method = 'IS'` (inverse-transform sampling) is currently implemented"
	end
end

# ---- PiecewiseEstimator ----

"""
	PiecewiseEstimator(estimators, breaks)
Creates a piecewise estimator from a collection of `estimators`, based on the
collection of changepoints, `breaks`, which should contain one element fewer
than the number of `estimators`.

Any estimator can be included in `estimators`, including any of the subtypes of
`NeuralEstimator` exported with the package `NeuralEstimators` (e.g., `PointEstimator`,
`QuantileEstimator`, etc.).

# Examples
```
# Suppose that we've trained two neural estimators. The first, θ̂₁, is trained
# for small sample sizes (e.g., m ≤ 30), and the second, `θ̂₂`, is trained for
# moderate-to-large sample sizes (e.g., m > 30). We construct a piecewise
# estimator with a sample-size changepoint of 30, which dispatches θ̂₁ if m ≤ 30
# and θ̂₂ if m > 30.

using NeuralEstimators
using Flux

n = 2  # bivariate data
p = 3  # number of parameters in the statistical model
w = 8  # width of each layer

ψ₁ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₁ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂₁ = DeepSet(ψ₁, ϕ₁)

ψ₂ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₂ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂₂ = DeepSet(ψ₂, ϕ₂)

θ̂ = PiecewiseEstimator([θ̂₁, θ̂₂], [30])
Z = [rand(n, 1, m) for m ∈ (10, 50)]
θ̂(Z)
```
"""
struct PiecewiseEstimator <: NeuralEstimator
	estimators
	breaks
	function PiecewiseEstimator(estimators, breaks)
		if length(breaks) != length(estimators) - 1
			error("The length of `breaks` should be one fewer than the number of `estimators`")
		elseif !issorted(breaks)
			error("`breaks` should be in ascending order")
		else
			new(estimators, breaks)
		end
	end
end
@layer PiecewiseEstimator
function (pe::PiecewiseEstimator)(Z)
	# Note that this is an inefficient implementation, analogous to the inefficient
	# DeepSet implementation. A more efficient approach would be to subset Z based
	# on breaks, apply the estimators to each block of Z, then combine the estimates.
	breaks = [pe.breaks..., Inf]
	m = numberreplicates(Z)
	θ̂ = map(eachindex(Z)) do i
		# find which estimator to use, and then apply it
		mᵢ = m[i]
		j = findfirst(mᵢ .<= breaks)
		pe.estimators[j](Z[[i]])
	end
	return stackarrays(θ̂)
end
Base.show(io::IO, pe::PiecewiseEstimator) = print(io, "\nPiecewise estimator with $(length(pe.estimators)) estimators and sample size change-points: $(pe.breaks)")


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
- `weight_by_distance::Bool = false`: (applicable only to GNNs) flag indicating whether the estimator will weight by spatial distance; if true, a `WeightedGraphConv` layer is used in the propagation module; otherwise, a regular `GraphConv` layer is used.

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
	weight_by_distance::Bool = false
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
	if architecture == "DNN"
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
		propagation = weight_by_distance ? WeightedGraphConv : GraphConv
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
