#NB deprecated because it isn't the recommended way of storing models anymore
"""
	loadbestweights(path::String)

Returns the weights of the neural network saved as 'best_network.bson' in the given `path`.
"""
loadbestweights(path::String) = loadweights(joinpath(path, "best_network.bson"))
loadweights(path::String) = load(path, @__MODULE__)[:weights]


# aliases for backwards compatability
mapestimate = posteriormode; export mapestimate
mlestimate = posteriormode; export mlestimate
WeightedGraphConv = SpatialGraphConv; export WeightedGraphConv 
simulategaussianprocess = simulategaussian; export simulategaussianprocess
estimateinbatches = estimate; export estimateinbatches
trainx = trainmultiple; export trainx
_runondevice(θ̂, z, use_gpu::Bool; batchsize::Integer = 32) = estimate(θ̂, z; batchsize = batchsize, use_gpu = use_gpu)


"""
Generic function that may be overloaded to implicitly define a statistical model.
Specifically, the user should provide a method `simulate(parameters, m)`
that returns `m` simulated replicates for each element in the given set of
`parameters`.
"""
function simulate end

"""
	simulate(parameters, m, J::Integer)

Simulates `J` sets of `m` independent replicates for each parameter vector in
`parameters` by calling `simulate(parameters, m)` a total of `J` times,
where the method `simulate(parameters, m)` is provided by the user via function
overloading.

# Examples
```
import NeuralEstimators: simulate

p = 2
K = 10
m = 15
parameters = rand(p, K)

# Univariate Gaussian model with unknown mean and standard deviation
simulate(parameters, m) = [θ[1] .+ θ[2] .* randn(1, m) for θ ∈ eachcol(parameters)]
simulate(parameters, m)
simulate(parameters, m, 2)
```
"""
function simulate(parameters::P, m, J::Integer; args...) where P <: Union{AbstractMatrix, ParameterConfigurations}
	v = [simulate(parameters, m; args...) for i ∈ 1:J]
	if typeof(v[1]) <: Tuple
		z = vcat([v[i][1] for i ∈ eachindex(v)]...)
		x = vcat([v[i][2] for i ∈ eachindex(v)]...)
		v = (z, x)
	else
		v = vcat(v...)
	end
	return v
end


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
