using Functors: @functor

"""
	NeuralEstimator

An abstract supertype for neural estimators.
"""
abstract type NeuralEstimator end

# ---- PointEstimator  ----

"""
    PointEstimator(arch)

A simple point estimator, that is, a mapping from the sample space to the
parameter space, defined by the given architecture `arch`.
"""
struct PointEstimator{F} <: NeuralEstimator
	arch::F
end
@functor PointEstimator (arch,)
(est::PointEstimator)(Z) = est.arch(Z)


# ---- IntervalEstimator: credible intervals  ----

"""
	IntervalEstimator(arch_lower, arch_upper)
	IntervalEstimator(arch)
A neural interval estimator that jointly estimates credible intervals constructed as,

```math
[l(Z), l(Z) + \\mathrm{exp}(u(Z))],
```

where ``l(⋅)`` and ``u(⋅)`` are the neural networks `arch_lower` and
`arch_upper`, both of which should transform data into ``p``-dimensional vectors,
where ``p`` is the number of parameters in the statistical model. If only a
single neural network architecture `arch` is provided, it will be used for both
`arch_lower` and `arch_upper`.

The returned value is a matrix with ``2p`` rows, where the first and second ``p``
rows correspond to estimates of the lower and upper bound, respectively.

See also [`IntervalEstimatorCompactPrior`](@ref).

# Examples
```
using NeuralEstimators
using Flux

# Generate some toy data
n = 2   # bivariate data
m = 100 # number of independent replicates
Z = rand(n, m)

# Create an architecture
p = 3  # number of parameters in the statistical model
w = 8  # width of each layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ = Chain(Dense(w, w, relu), Dense(w, p));
architecture = DeepSet(ψ, ϕ)

# Initialise the interval estimator
estimator = IntervalEstimator(architecture)

# Apply the interval estimator
estimator(Z)
interval(estimator, Z)
```
"""
struct IntervalEstimator{F, G} <: NeuralEstimator
	l::F
	u::G
end
IntervalEstimator(l) = IntervalEstimator(l, deepcopy(l))
@functor IntervalEstimator
function (est::IntervalEstimator)(Z)
	l = est.l(Z)
	vcat(l, l .+ exp.(est.u(Z)))
end
# Ensure that IntervalEstimator objects are not constructed with PointEstimator:
#TODO find a neater way to do this; don't want to write so many methods, especially for PointIntervalEstimator
IntervalEstimator(l::PointEstimator, u::PointEstimator) = IntervalEstimator(l.arch, u.arch)
IntervalEstimator(l, u::PointEstimator) = IntervalEstimator(l, u.arch)
IntervalEstimator(l::PointEstimator, u) = IntervalEstimator(l.arch, u)


#TODO unit testing
"""
	IntervalEstimatorCompactPrior(u, v, min_supp::Vector, max_supp::Vector)
	IntervalEstimatorCompactPrior(u, v, compress::Compress)
Uses the neural networks `u` and `v` to jointly estimate credible intervals
that are guaranteed to be within the support of the prior distributon. This
support is defined by the ``p``-dimensional vectors `min_supp` and `max_supp`
(or a single ``p``-dimensional object of type `Compress`), where ``p`` is the
number of parameters in the statistical model.

Given data ``Z``, the intervals are constructed as

```math
[g(u(Z)), 	g(u(Z)) + \\mathrm{exp}(v(Z)))],
```

where

- ``u(⋅)`` and ``v(⋅)`` are neural networks, both of which should transform data into ``p``-dimensional vectors;
- ``g(⋅)`` is a logistic function that maps its input to the prior support.

Note that, in addition to ensuring that the interval remains in the prior support,
this constructions also ensures that the intervals are valid (i.e., it prevents
quantile crossing, in the sense that the upper bound is always greater than the
lower bound).

The returned value is a matrix with ``2p`` rows, where the first and second ``p``
rows correspond to estimates of the lower and upper bound, respectively.

See also [`IntervalEstimator`](@ref) and [`Compress`](@ref).

# Examples
```
using NeuralEstimators
using Flux

# prior support
min_supp = [25, 0.5, -pi/2]
max_supp = [500, 2.5, 0]
p = length(min_supp)  # number of parameters in the statistical model

# Generate some toy data
n = 2   # bivariate data
m = 100 # number of independent replicates
Z = rand(n, m)

# Create an architecture
w = 8  # width of each layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ = Chain(Dense(w, w, relu), Dense(w, p));
u = DeepSet(ψ, ϕ)
v = deepcopy(u) # use the same architecture for both u and v

# Initialise the interval estimator
estimator = IntervalEstimatorCompactPrior(u, v, min_supp, max_supp)

# Apply the interval estimator
estimator(Z)
interval(estimator, Z)
```
"""
struct IntervalEstimatorCompactPrior{F, G} <: NeuralEstimator
	u::F
	v::G
	c::Compress
end
IntervalEstimatorCompactPrior(u, v, min_supp, max_supp) = IntervalEstimatorCompactPrior(u, v, Compress(min_supp, max_supp))
@functor IntervalEstimatorCompactPrior
Flux.trainable(est::IntervalEstimatorCompactPrior) = (est.u, est.v)

function (est::IntervalEstimatorCompactPrior)(Z)
	x = est.u(Z)
	y = x .+ exp.(est.v(Z))
	c = est.c
	vcat(c(x), c(y))
end


# NB more complicated version:
# function (est::IntervalEstimatorCompactPrior)(Z)
#
# 	# Extract the compress object that encodes the compact prior support:
# 	c = est.c
#
# 	# Scale the low quantile to the prior support:
# 	u = est.u(Z)
# 	f = c(u)
#
# 	# Scale the high-quantile term to be within u and the maximum of the prior support:
# 	v = est.v(Z)
# 	g = (c.b .- f) ./ (one(eltype(v)) .+ exp.(-c.k .* v))
#
# 	vcat(f, f .+ g)
# end



#TODO unit testing
"""
	PointIntervalEstimator(arch_point, arch_lower, arch_upper)
	PointIntervalEstimator(arch_point, arch_bound)
	PointIntervalEstimator(arch)
A neural estimator that jointly produces point estimates, θ̂(Z), where θ̂(Z) is a
neural point estimator with architecture `arch_point`, and credible intervals constructed as,

```math
[θ̂(Z) - \\mathrm{exp}(l(Z)), θ̂(Z) + \\mathrm{exp}(u(Z))],
```

where ``l(⋅)`` and ``u(⋅)`` are the neural networks `arch_lower` and
`arch_upper`, both of which should transform data into ``p``-dimensional vectors,
where ``p`` is the number of parameters in the statistical model.

If only a single neural network architecture `arch` is provided, it will be used
for all architectures; similarly, if two architectures are provided, the second
will be used for both `arch_lower` and `arch_upper`.

Internally, the point estimates, lower-bound estimates, and upper-bound estimates are concatenated, so
that `PointIntervalEstimator` objects transform data into matrices with ``3p`` rows.

# Examples
```
using NeuralEstimators
using Flux

# Generate some toy data
n = 2   # bivariate data
m = 100 # number of independent replicates
Z = rand(n, m)

# Create an architecture
p = 3  # number of parameters in the statistical model
w = 8  # width of each layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ = Chain(Dense(w, w, relu), Dense(w, p));
architecture = DeepSet(ψ, ϕ)

# Initialise the estimator
estimator = PointIntervalEstimator(architecture)

# Apply the estimator
estimator(Z)
interval(estimator, Z)
```
"""
struct PointIntervalEstimator{H, F, G} <: NeuralEstimator
	θ̂::H
	l::F
	u::G
end
PointIntervalEstimator(θ̂) = PointIntervalEstimator(θ̂, deepcopy(θ̂), deepcopy(θ̂))
PointIntervalEstimator(θ̂, l) = PointIntervalEstimator(θ̂, deepcopy(l), deepcopy(l))
@functor PointIntervalEstimator
function (est::PointIntervalEstimator)(Z)
	θ̂ = est.θ̂(Z)
	vcat(θ̂, θ̂ .- exp.(est.l(Z)), θ̂ .+ exp.(est.u(Z)))
end
# Ensure that IntervalEstimator objects are not constructed with PointEstimator:
#TODO find a neater way to do this; don't want to write so many methods, especially for PointIntervalEstimator
PointIntervalEstimator(θ̂::PointEstimator, l::PointEstimator, u::PointEstimator) = PointIntervalEstimator(θ̂.arch, l.arch, u.arch)
PointIntervalEstimator(θ̂::PointEstimator, l, u) = PointIntervalEstimator(θ̂.arch, l, u)
PointIntervalEstimator(θ̂, l::PointEstimator, u::PointEstimator) = PointIntervalEstimator(θ̂, l.arch, u.arch)


# ---- QuantileEstimator: estimating arbitrary quantiles of the posterior distribution ----

# Should Follow up with this point from Gnieting's paper:
# 9.2 Quantile Estimation
# Koenker and Bassett (1978) proposed quantile regression using an optimum score estimator based on the proper scoring rule (41).


#TODO this is a topic of ongoing research with Jordan
"""
    QuantileEstimator()

Coming soon: this structure will allow for the simultaneous estimation of an
arbitrary number of marginal quantiles of the posterior distribution.
"""
struct QuantileEstimator{F, G} <: NeuralEstimator
	l::F
	u::G
end
# @functor QuantileEstimator
# (c::QuantileEstimator)(Z) = vcat(c.l(Z), c.l(Z) .+ exp.(c.u(Z)))



# ---- PiecewiseEstimator ----

"""
	PiecewiseEstimator(estimators, breaks)
Creates a piecewise estimator from a collection of `estimators`, based on the
collection of changepoints, `breaks`, which should contain one element fewer
than the number of `estimators`.

Any estimator can be included in `estimators`, including any of the subtypes of
`NeuralEstimator` exported with the package `NeuralEstimators` (e.g., `PointEstimator`,
`IntervalEstimator`, etc.).

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
@functor PiecewiseEstimator (estimators,)

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

# Clean printing:
Base.show(io::IO, pe::PiecewiseEstimator) = print(io, "\nPiecewise estimator with $(length(pe.estimators)) estimators and sample size change-points: $(pe.breaks)")
Base.show(io::IO, m::MIME"text/plain", pe::PiecewiseEstimator) = print(io, pe)


# ---- Helper function for initialising an estimator ----

"""
    initialise_estimator(p::Integer; ...)
Initialise a neural estimator for a statistical model with `p` unknown parameters.

The estimator is couched in the DeepSets framework (see [`DeepSet`](@ref)) so
that it can be applied to data sets containing an arbitrary number of
independent replicates (including the special case of a single replicate).

Note also that the user is free to initialise their neural estimator however
they see fit using arbitrary `Flux` code; see
[here](https://fluxml.ai/Flux.jl/stable/models/layers/) for `Flux`'s API reference.

# Keyword arguments
- `architecture::String`: for unstructured data, one may use a densely-connected neural network (`"DNN"`); for data collected over a grid, a convolutional neural network (`"CNN"`); and for graphical or irregular spatial data, a graphical neural network (`"GNN"`).
- `d::Integer = 1`: dimension of the response variable (e.g., `d = 1` for univariate processes).
- `estimator_type::String = "point"`: the type of estimator; either `"point"` or `"interval"`.
- `depth = 3`: the number of hidden layers; either a single integer or an integer vector of length two specifying the depth of the inner (summary) and outer (inference) network of the DeepSets framework.
- `width = 32`: a single integer or an integer vector of length `sum(depth)` specifying the width (or number of convolutional filters/channels) in each hidden layer.
- `activation::Function = relu`: the (non-linear) activation function of each hidden layer.
- `activation_output::Function = identity`: the activation function of the output layer.
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
    activation::Function = relu,
    activation_output::Function = identity,
    kernel_size = nothing,
	weight_by_distance::Bool = false
    )

    @assert p > 0
    @assert d > 0
	@assert architecture ∈ ["DNN", "CNN", "GNN"]
    @assert estimator_type ∈ ["point", "interval"]
    @assert all(depth .> 0)
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
	ϕ = Chain(
		[Dense(width[l-1] => width[l], activation) for l ∈ (depth[1]+1):L]...,
		Dense(width[L] => p, activation_output)
		)

	# summary (inner) network
	if architecture == "DNN"
		ψ = Chain(
			Dense(d => width[1], activation),
			[Dense(width[l] => width[l+1], activation) for l ∈ 1:depth[1]]...
			)
	elseif architecture == "CNN"
		ψ = Chain(
			Conv(kernel_size[1], d => width[1], activation),
			[Conv(kernel_size[l], width[l] => width[l+1], activation) for l ∈ 1:depth[1]]...,
			Flux.flatten
			)
	elseif architecture == "GNN"
		propagation = weight_by_distance ? WeightedGraphConv : GraphConv
		# propagation_module = GNNChain(
		# 	propagation(d => width[1], activation),
		# 	[propagation(width[l] => width[l+1], relu) for l ∈ 1:depth[1]]...
		# 	)
		# readout_module = GlobalPool(mean)
		# return GNN(propagation_module, readout_module, ϕ) # return more-efficient GNN object
		ψ = GNNChain(
			propagation(d => width[1], activation, aggr = mean),
			[propagation(width[l] => width[l+1], relu, aggr = mean) for l ∈ 1:depth[1]]...,
			GlobalPool(mean) # readout module
			)
	end

	θ̂ = DeepSet(ψ, ϕ)

	if estimator_type == "interval"
		θ̂ = IntervalEstimator(θ̂, θ̂)
	end

	return θ̂
end

coercetotuple(x) = (x...,)
