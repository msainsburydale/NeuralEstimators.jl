using Functors: @functor

"""
	NeuralEstimator

An abstract supertype for neural estimators.
"""
abstract type NeuralEstimator end

# ---- PointEstimator  ----

"""
    PointEstimator(arch)

A neural point estimator, that is, a mapping from the sample space to the
parameter space, defined by the given neural-network architecture `arch`.
"""
struct PointEstimator{F} <: NeuralEstimator
	arch::F
	# PointEstimator(arch) = isa(arch, PointEstimator) ? error("Please do not construct PointEstimator objects with another PointEstimator") : new(arch)
end
@functor PointEstimator (arch,)
(est::PointEstimator)(Z) = est.arch(Z)


# ---- IntervalEstimator for amortised credible intervals  ----

"""
	IntervalEstimator(u)
	IntervalEstimator(u, v)
	IntervalEstimator(u, v, g::Compress)
	IntervalEstimator(u, v, min_supp::Vector, max_supp::Vector)

A neural interval estimator which, given data ``Z``, jointly estimates credible
intervals in the form,

```math
[g(u(Z)), 	g(u(Z)) + \\mathrm{exp}(v(Z)))],
```

where

- ``u(⋅)`` and ``v(⋅)`` are neural networks, both of which should transform data into ``p``-dimensional vectors (with ``p`` the number of parameters in the statistical model);
- ``g(⋅)`` is a logistic function that maps its input to the prior support.

The prior support is defined either by the ``p``-dimensional vectors `min_supp`
and `max_supp`, or a single ``p``-dimensional object of type [`Compress`](@ref).
If these objects are not given, the range of the intervals will be unrestricted.

Note that, in addition to ensuring that the interval remains in the prior support,
this construction also ensures that the intervals are valid (i.e., it prevents
quantile crossing, in the sense that the upper bound is always greater than the
lower bound).

If only a single neural-network architecture is provided, it will be used
for both `u` and `v`.

The returned value is a matrix with ``2p`` rows, where the first and second ``p``
rows correspond to estimates of the lower and upper bound, respectively.

# Examples
```
using NeuralEstimators
using Flux

# Generate some toy data
n = 2   # bivariate data
m = 100 # number of independent replicates
Z = rand(n, m)

# prior support
min_supp = [25, 0.5, -pi/2]
max_supp = [500, 2.5, 0]
p = length(min_supp)  # number of parameters in the statistical model

# Create an architecture
w = 8  # width of each layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ = Chain(Dense(w, w, relu), Dense(w, p));
u = DeepSet(ψ, ϕ)
v = deepcopy(u) # use the same architecture for both u and v

# Initialise the interval estimator
estimator = IntervalEstimator(u, v, min_supp, max_supp)

# Apply the interval estimator
estimator(Z)
interval(estimator, Z)
```
"""
struct IntervalEstimator{F, G} <: NeuralEstimator
	u::F
	v::G
	g::Union{Function,Compress}
	# IntervalEstimator(u, v, g) = any(isa.([u, v], PointEstimator)) ? error("Please do not construct IntervalEstimator objects with PointEstimators") : new(u, v, g)
end
IntervalEstimator(u) = IntervalEstimator(u, deepcopy(u), identity)
IntervalEstimator(u, v) = IntervalEstimator(u, v, identity)
IntervalEstimator(u, g::Compress) = IntervalEstimator(u, deepcopy(u), g)
IntervalEstimator(u, min_supp, max_supp) = IntervalEstimator(u, Compress(min_supp, max_supp))
IntervalEstimator(u, v, min_supp, max_supp) = IntervalEstimator(u, v, Compress(min_supp, max_supp))
@functor IntervalEstimator
Flux.trainable(est::IntervalEstimator) = (est.u, est.v)
function (est::IntervalEstimator)(Z)
	bₗ = est.u(Z)              # lower bound
	bᵤ = bₗ .+ exp.(est.v(Z))  # upper bound
	vcat(est.g(bₗ), est.g(bᵤ))
end


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
- `architecture::String`: for unstructured multivariate data, one may use a densely-connected neural network (`"DNN"`); for data collected over a grid, a convolutional neural network (`"CNN"`); and for graphical or irregular spatial data, a graphical neural network (`"GNN"`).
- `d::Integer = 1`: for unstructured multivariate data (i.e., when `architecture = "DNN"`), the dimension of the data (e.g., `d = 3` for trivariate data); otherwise, if `architecture ∈ ["CNN", "GNN"]`, the argument `d` controls the number of input channels (e.g., `d = 1` for univariate spatial processes).
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

	θ̂ = DeepSet(ψ, ϕ)

	if estimator_type == "interval"
		θ̂ = IntervalEstimator(θ̂, θ̂)
	end

	return θ̂
end

coercetotuple(x) = (x...,)
