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
[c_i(u_i(\mathbf{Z})), \;\; c_i(u_i(\mathbf{Z})) + g(v_i(\mathbf{Z})))],
```
where  ``\mathbf{u}(⋅) \equiv (u_1(\cdot), \dots, u_p(\cdot))'`` and
``\mathbf{v}(⋅) \equiv (v_1(\cdot), \dots, v_p(\cdot))'`` are neural networks
that transform data into ``p``-dimensional vectors; $g(\cdot)$ is a
monotonically increasing function (e.g., exponential or softplus); and each
``c_i(⋅)`` is a monotonically increasing function that maps its input to the
prior support of ``\theta_i``.

The functions ``c_i(⋅)`` may be defined by a ``p``-dimensional object of type
[`Compress`](@ref). If these functions are unspecified, they will be set to the
identity function so that the range of the intervals will be unrestricted.

If only a single neural-network architecture is provided, it will be used
for both ``\mathbf{u}(⋅)`` and ``\mathbf{v}(⋅)``.

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

Note that one may use a simple [`PointEstimator`](@ref) and the
[`quantileloss`](@ref) to target a specific quantile.

The return value  when applied to data is a matrix with ``pT`` rows, where the
first set of ``T`` rows corresponds to the estimated quantiles for the first
parameter, the second set of ``T`` rows corresponds to the estimated quantiles
for the second parameter, and so on.

See also [`IntervalEstimator`](@ref) and
[`QuantileEstimatorContinuous`](@ref).

# Examples
```
using NeuralEstimators, Flux, Distributions

# Generate data from the model Z|θ ~ N(θ, 1) and θ ~ N(0, 1)
p = 1       # number of unknown parameters in the statistical model
m = 30      # number of independent replicates
d = 1       # dimension of each independent replicate
K = 30000   # number of training samples
prior(K) = randn32(1, K)
simulate(θ, m) = [μ .+ randn32(1, m) for μ ∈ eachcol(θ_train)]
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
```
"""
struct QuantileEstimatorDiscrete{V, P} <: NeuralEstimator
	v::V
	probs::P
	g::Function
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
#TODO In the docs example, why does the risk increase during training? Is it because I'm computing the risk in different ways before vs. during the training loop?

# # Compare to the closed-form posterior
# function posterior(Z)
# 	# Prior hyperparameters
# 	σ₀  = 1
# 	σ₀² = σ₀^2
# 	μ₀  = 0
#
# 	# Known variance
#     σ  = 1
# 	σ² = σ^2
#
# 	# Posterior parameters
# 	μ̃ = (1/σ₀² + length(Z)/σ²)^-1 * (μ₀/σ₀² + sum(Z)/σ²)
# 	σ̃ = sqrt((1/σ₀² + length(Z)/σ²)^-1)
#
# 	# Posterior
# 	Normal(μ̃, σ̃)
# end
# true_quantiles = quantile.(posterior.(Z), Ref([0.05, 0.25, 0.5, 0.75, 0.95]))
# true_quantiles = reduce(hcat, true_quantiles)
#
# # Compare estimates to true values
# θ̂(Z) - true_quantiles

#TODO describe more the architecture (dimension of input, dimension of output, the use of DensePositive)
#TODO add reference to Cannon (2018) and others
@doc raw"""
	QuantileEstimatorContinuous(deepset::DeepSet)
A neural estimator that estimates marginal posterior quantiles given as
input the desired probability level $\tau ∈ (0, 1)$.

The estimator takes as input the data $\mathbf{Z}$ and the desired probability
level $\tau$, and therefore has the form,

```math
\hat{\mathbf{q}}(\mathbf{Z}, \tau), \quad \tau ∈ (0, 1).
```

The estimator leverages the [`DeepSet`](@ref) architecture. The first
requirement is that number of input neurons in the first layer of the inference
network (i.e., the outer network) is one greater than the number of output
neurons in the final layer of the summary network. The second requirement is
that the number of output neurons in the final layer of the inference network
is equal to the number ``p`` of parameters in the statistical model.

Although not a requirement, one may employ a (partially) monotonic neural
network to prevent quantile crossing (i.e., to ensure that the
$\tau_1$th quantile does not exceed the $\tau_2$th quantile for any
$\tau_2 > \tau_1$). There are several ways to construct such a neural network:
one simple yet effective approach is to ensure that all weights associated with
$\tau$ are strictly positive, and this can be done using the
[`DensePositive`](@ref) layer as illustrated in the examples below.

The return value when applied to data is a matrix with ``p`` rows, corresponding
to the estimated marginal posterior quantiles for each parameter in the
statistical model.

# Examples
```
using NeuralEstimators, Flux

# Generate training data from the model Z|θ ~ N(θ, 1) and θ ~ N(0, 1)
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

# Architecture (monotonic network to preclude the possibility of quantile crossing)
w = 64   # width of each hidden layer
q = 2p   # output dimension of summary network
ψ = Chain(
	Dense(d, w, relu),
	Dense(w, w, relu),
	Dense(w, q, relu)
	)
ϕ = Chain(
	DensePositive(Dense(q + 1, w, relu); last_only = true),
	DensePositive(Dense(w, w, relu)),
	DensePositive(Dense(w, p))
	)
deepset = DeepSet(ψ, ϕ)

# Initialise the estimator
θ̂ = QuantileEstimatorContinuous(deepset)

# Train the estimator
train(θ̂, θ_train, θ_val, Zτ_train, Zτ_val)

# Estimate a single quantile for many test data sets
θ = prior(K)
Z = simulateZ(θ, m)
τ = 0.1f0
θ̂(Z[1], τ)
θ̂(Z, τ)

# Estimate multiple quantiles for a single data set
τ = [0.1, 0.25, 0.5, 0.75, 0.9]
reduce(hcat, θ̂.(Ref(Z[1]), τ))
```
"""
struct QuantileEstimatorContinuous <: NeuralEstimator
	deepset::DeepSet
end
@layer QuantileEstimatorContinuous
function (est::QuantileEstimatorContinuous)(Z::A, τ::Number) where A
	est(Z, [τ])
end
function (est::QuantileEstimatorContinuous)(Z::V, τ::Number) where V <: AbstractVector{A} where A
	est(Z, repeat([[τ]], length(Z)))
end
(est::QuantileEstimatorContinuous)(Z, τ) = est((Z, τ)) # "Tupleise" input and pass to Tuple method
(est::QuantileEstimatorContinuous)(Zτ::Tuple) = est.deepset(Zτ)
#TODO why is the validation risk so large (why is there so much overfitting, at
#     least as measured by validation risk)?
#TODO train(θ̂, prior, simulate, m = m)  # training with "on-the-fly" simulation
#TODO why is the initial training risk different to the previous final training risk?
# train(θ̂, θ_train, θ_train, Zτ_train, Zτ_train)
# train(θ̂, θ_train, θ_train, Zτ_train, Zτ_train)

# # Compare to closed-form posterior
# function posterior(Z)
# 	# Prior hyperparameters
# 	σ₀  = 1
# 	σ₀² = σ₀^2
# 	μ₀  = 0
#
# 	# Known variance
#     σ  = 1
# 	σ² = σ^2
#
# 	# Posterior parameters
# 	μ̃ = (1/σ₀² + length(Z)/σ²)^-1 * (μ₀/σ₀² + sum(Z)/σ²)
# 	σ̃ = sqrt((1/σ₀² + length(Z)/σ²)^-1)
#
# 	# Posterior
# 	Normal(μ̃, σ̃)
# end
#
# # Many data sets, single quantile:
# quantile.(posterior.(Z), τ)'
# θ̂(Z, τ)
#
# # Single data set, range of quantiles:
# τ = [0.1, 0.25, 0.5, 0.75, 0.9]
# quantile.(posterior(Z[1]), τ)'
# reduce(hcat, θ̂.(Ref(Z[1]), τ))

# ---- RatioEstimator  ----

@doc raw"""
	RatioEstimator(deepset::DeepSet)

A neural estimator that estimates the likelihood-to-evidence ratio,
```math
r(\mathbf{Z}, \mathbf{\theta}) \equiv p(\mathbf{Z} \mid \mathbf{\theta})/p(\mathbf{Z}),
```
where $p(\mathbf{Z} \mid \mathbf{\theta})$ is the likelihood and $p(\mathbf{Z})$
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
${(\mathbf{\theta}', \mathbf{Z}')' \sim p(\mathbf{Z}, \mathbf{\theta})}$ with
class labels $Y=1$ from independent parameter--data pairs
${(\tilde{\mathbf{\theta}}', \tilde{\mathbf{Z}}')' \sim p(\mathbf{\theta})p(\mathbf{Z})}$
with class labels $Y=0$, and where the classes are balanced. Then the Bayes
classifier under binary cross-entropy loss is given by
```math
c(\mathbf{Z}, \mathbf{\theta}) = \frac{p(\mathbf{Z}, \mathbf{\theta})}{p(\mathbf{Z}, \mathbf{\theta}) + p(\mathbf{\theta})p(\mathbf{Z})},
```
and hence,
```math
r(\mathbf{Z}, \mathbf{\theta}) = \frac{c(\mathbf{Z}, \mathbf{\theta})}{1 - c(\mathbf{Z}, \mathbf{\theta})}.
```
For numerical stability, training is done on the log-scale using
$\log r(\mathbf{Z}, \mathbf{\theta}) = \text{logit}(c(\mathbf{Z}, \mathbf{\theta}))$.

When applying the estimator to data, by default the likelihood-to-evidence ratio
$r(\mathbf{Z}, \mathbf{\theta})$ is returned (setting the keyword argument
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
p = 2        # number of unknown parameters in the statistical model
d = 1        # dimension of each independent replicate
m = 100      # number of independent replicates
K = 10000    # number of training samples

prior(K) = rand32(p, K)
simulate(θ, m) = θ[1] .+ θ[2] .* randn32(d, m)
simulate(θ::AbstractMatrix, m) = simulate.(eachcol(θ), m)

θ_train = prior(K)
θ_val   = prior(K)
Z_train = simulate(θ_train, m)
Z_val   = simulate(θ_val, m)

# Architecture
w = 64 # width of each hidden layer
q = 2p # number of learned summary statistics
ψ = Chain(
	Dense(d, w, relu),
	Dropout(0.3),
	Dense(w, w, relu),
	Dropout(0.3),
	Dense(w, q, relu),
	Dropout(0.3)
	)
ϕ = Chain(
	Dense(q + p, w, relu),
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

# Inference with "observed" data set
θ = prior(1)
z = simulate(θ, m)[1]
θ₀ = [0.5, 0.5]                           # initial estimate
mlestimate(r̂, z;  θ₀ = θ₀)                # maximum-likelihood estimate
mapestimate(r̂, z; θ₀ = θ₀)                # maximum-a-posteriori estimate
θ_grid = expandgrid(0:0.01:1, 0:0.01:1)'  # fine gridding of the parameter space
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

# References
# - Hermans J, Begy V, Louppe G. 2020. Likelihood-free MCMC with amortized approximate ratio estimators, In Proceedings of the 37th International Conference on Machine Learning (ICML 2020), vol. 119, pp. 4239–4248, PMLR
# - Walchessen J, Lenzi A, Kuusela M. 2023. Neural likelihood surfaces for spatial processes with computationally intensive or intractable likelihoods. arXiv:2305.04634 [stat.ME]

# # Estimate ratio for many data sets and parameter vectors
# θ = prior(K)
# Z = simulate(θ, m)
# r̂(Z, θ)                                   # likelihood-to-evidence ratios
# r̂(Z, θ; classifier = true)                # class probabilities

# # Inference with multiple data sets
# θ = prior(10)
# z = simulate(θ, m)
# r̂(z, grid)                                         # likelihood-to-evidence ratios
# mlestimate(r̂, z; theta_grid = grid)                # maximum-likelihood estimates
# mlestimate(r̂, z; theta_init = [0.5, 0.5])          # maximum-likelihood estimates
# samples = sampleposterior(r̂, z; theta_grid = grid) # posterior samples
# θ̄ = mean.(samples; dims = 2)                       # posterior means
# interval.(samples; probs = [0.05, 0.95])           # posterior credible intervals

# ---- PiecewiseEstimator ----

"""
	PiecewiseEstimator(estimators, breaks)
Creates a piecewise estimator from a collection of `estimators`, based on the
collection of changepoints, `breaks`, which should contain one element fewer
than the number of `estimators`.

# Examples
```
# Suppose that we've trained two neural estimators. The first, θ̂₁, is trained
# for small sample sizes (e.g., m ≤ 30), and the second, `θ̂₂`, is trained for
# moderate-to-large sample sizes (e.g., m > 30). We construct a piecewise
# estimator with a sample-size changepoint of 30, which dispatches θ̂₁ if m ≤ 30
# and θ̂₂ if m > 30.

using NeuralEstimators, Flux

n = 2  # bivariate data
p = 3  # number of parameters in the statistical model
w = 8  # width of each hidden layer

# Small-sample estimator
ψ₁ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₁ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂₁ = PointEstimator(DeepSet(ψ₁, ϕ₁))

# Large-sample estimator
ψ₂ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₂ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂₂ = PointEstimator(DeepSet(ψ₂, ϕ₂))

# Piecewise estimator
θ̂ = PiecewiseEstimator([θ̂₁, θ̂₂], [30])

# Apply the estimator
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
