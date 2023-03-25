# TODO should document the parameter-level covariate functionality, and add
# testing for it. Need to do this for DeepSetExpert too.
# TODO Test that training works with covariates.

using Functors: @functor
using RecursiveArrayTools: VectorOfArray, convert

# ---- Aggregation (pooling) functions ----

meanlastdim(X::A) where {A <: AbstractArray{T, N}} where {T, N} = mean(X, dims = N)
sumlastdim(X::A)  where {A <: AbstractArray{T, N}} where {T, N} = sum(X, dims = N)
LSElastdim(X::A)  where {A <: AbstractArray{T, N}} where {T, N} = logsumexp(X, dims = N)

function _agg(a::String)
	@assert a ∈ ["mean", "sum", "logsumexp"]
	if a == "mean"
		meanlastdim
	elseif a == "sum"
		sumlastdim
	elseif a == "logsumexp"
		LSElastdim
	end
end

# ---- DeepSet Type and constructors ----

# we use unicode characters below to preserve readability of REPL help files
"""
    DeepSet(ψ, ϕ, a)
	DeepSet(ψ, ϕ; a::String = "mean")

A neural estimator in the `DeepSet` representation,

```math
θ̂(𝐙) = ϕ(𝐓(𝐙)),	 	 𝐓(𝐙) = 𝐚(\\{ψ(𝐙ᵢ) : i = 1, …, m\\}),
```

where 𝐙 ≡ (𝐙₁', …, 𝐙ₘ')' are independent replicates from the model, `ψ` and `ϕ`
are neural networks, and `𝐚` is a permutation-invariant aggregation function.

The function `𝐚` must aggregate over the last dimension of an array (i.e., the
replicates dimension). It can be specified as a positional argument of
type `Function`, or as a keyword argument of type `String` with permissible
values `"mean"`, `"sum"`, and `"logsumexp"`.

# Examples
```
using NeuralEstimators
using Flux

n = 10 # number of observations in each realisation
p = 4  # number of parameters in the statistical model

# Construct the neural estimator
w = 32 # width of each layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂ = DeepSet(ψ, ϕ)

# Apply the estimator to a single set of 3 realisations:
Z₁ = rand(n, 3);
θ̂(Z₁)

# Apply the estimator to two sets each containing 3 realisations:
Z₂ = [rand(n, m) for m ∈ (3, 3)];
θ̂(Z₂)

# Apply the estimator to two sets containing 3 and 4 realisations, respectively:
Z₃ = [rand(n, m) for m ∈ (3, 4)];
θ̂(Z₃)

# Repeat the above but with some covariates:
dₓ = 2
ϕₓ = Chain(Dense(w + dₓ, w, relu), Dense(w, p));
θ̂  = DeepSet(ψ, ϕₓ)
x₁ = rand(dₓ)
x₂ = [rand(dₓ), rand(dₓ)]
θ̂((Z₁, x₁))
θ̂((Z₃, x₂))
```
"""
struct DeepSet{T, F, G}
	ψ::T
	ϕ::G
	a::F
end
# 𝐙₁ → ψ() \n
#          ↘ \n
# ⋮     ⋮     a() → ϕ() \n
#          ↗ \n
# 𝐙ₘ → ψ() \n


DeepSet(ψ, ϕ; a::String = "mean") = DeepSet(ψ, ϕ, _agg(a))

@functor DeepSet # allows Flux to optimise the parameters

# Clean printing:
Base.show(io::IO, D::DeepSet) = print(io, "\nDeepSet object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.a)\nOuter network:  $(D.ϕ)")
Base.show(io::IO, m::MIME"text/plain", D::DeepSet) = print(io, D)


# ---- Methods ----

function (d::DeepSet)(Z::A) where {A <: AbstractArray{T, N}} where {T, N}
	d.ϕ(d.a(d.ψ(Z)))
end

function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{A, B}} where {A <: AbstractArray{T, N}, B <: AbstractVector{T}} where {T, N}
	Z = tup[1]
	x = tup[2]
	t = d.a(d.ψ(Z))
	d.ϕ(vcat(t, x))
end


# Simple, intuitive (although inefficient) implementation using broadcasting:

# function (d::DeepSet)(v::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
#   θ̂ = d.ϕ.(d.a.(d.ψ.(v)))
#   θ̂ = stackarrays(θ̂)
#   return θ̂
# end

# Optimised version. This approach ensures that the neural networks ψ and ϕ are
# applied to arrays that are as large as possible, improving efficiency compared
# with the intuitive method above (particularly on the GPU):
function (d::DeepSet)(Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}

	# Convert to a single large Array
	z = stackarrays(Z)

	# Apply the inner neural network
	ψa = d.ψ(z)

	# Compute the indices needed for aggregation and construct a tuple of colons
	# used to subset all but the last dimension of ψa. Note that constructing
	# colons in this manner makes the function agnostic to ndims(ψa).
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)

	# Aggregate each set of transformed features: The resulting vector from the
	# list comprehension is a vector of arrays, where the last dimension of each
	# array is of size 1. Then, stack this vector of arrays into one large array,
	# where the last dimension of this large array has size equal to length(v).
	# Note that we cannot pre-allocate and fill an array, since array mutation
	# is not supported by Zygote (which is needed during training).
	large_aggregated_ψa = [d.a(ψa[colons..., idx]) for idx ∈ indices] |> stackarrays

	# Apply the outer network
	θ̂ = d.ϕ(large_aggregated_ψa)

	return θ̂
end

function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A <: AbstractArray{T, N}, B <: AbstractVector{T}} where {T, N}


	Z = tup[1]
	X = tup[2]

	# Convert to a single large Array
	z = stackarrays(Z)

	# Apply the inner neural network to obtain the neural summary statistics
	ψa = d.ψ(z)

	# Compute the indices needed for aggregation and construct a tuple of colons
	# used to subset all but the last dimension of ψa.
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)

	# concatenate the neural summary statistics with X
	u = map(eachindex(Z)) do i
		idx = indices[i]
		t = d.a(ψa[colons..., idx])
		x = X[i]
		u = vcat(t, x)
		u
	end
	u = stackarrays(u)

	# Apply the outer network
	θ̂ = d.ϕ(u)

	return θ̂
end




# markdown code for documentation in docs/src/workflow/advancedusage.md:
# # Combining neural and expert summary statistics
#
# See [`DeepSetExpert`](@ref).

"""
	samplesize(Z)

Computes the sample size m for a set of independent realisations `Z`, often
useful as an expert summary statistic in `DeepSetExpert` objects.

Note that this function is a simple wrapper around `numberreplicates`, but this
function returns the number of replicates as the eltype of `Z`.
"""
samplesize(Z) = eltype(Z)(numberreplicates(Z))

samplesize(Z::V) where V <: AbstractVector = samplesize.(Z)


# ---- DeepSetExpert Type and constructors ----

# we use unicode characters below to preserve readability of REPL help files
"""
	DeepSetExpert(ψ, ϕ, S, a)
	DeepSetExpert(ψ, ϕ, S; a::String)
	DeepSetExpert(deepset::DeepSet, ϕ, S)


A neural estimator in the `DeepSet` representation with additional expert
summary statistics,

```math
θ̂(𝐙) = ϕ((𝐓(𝐙)', 𝐒(𝐙)')'),	 	 𝐓(𝐙) = 𝐚(\\{ψ(𝐙ᵢ) : i = 1, …, m\\}),
```

where 𝐙 ≡ (𝐙₁', …, 𝐙ₘ')' are independent replicates from the model,
`ψ` and `ϕ` are neural networks, `S` is a function that returns a vector
of expert summary statistics, and `𝐚` is a permutation-invariant
aggregation function.

The dimension of the domain of `ϕ` must be qₜ + qₛ, where qₜ and qₛ are the
dimensions of the ranges of `ψ` and `S`, respectively.

The constructor `DeepSetExpert(deepset::DeepSet, ϕ, S)` inherits `ψ` and `a`
from `deepset`.

See `?DeepSet` for discussion on the aggregation function `𝐚`.

# Examples
```
using NeuralEstimators
using Flux

n = 10 # number of observations in each realisation
p = 4  # number of parameters in the statistical model

# Construct the neural estimator
S = samplesize
qₛ = 1
qₜ = 32
w = 16
ψ = Chain(Dense(n, w, relu), Dense(w, qₜ, relu));
ϕ = Chain(Dense(qₜ + qₛ, w), Dense(w, p));
θ̂ = DeepSetExpert(ψ, ϕ, S)

# Apply the estimator to a single set of 3 realisations:
Z = rand(n, 3);
θ̂(Z)

# Apply the estimator to two sets each containing 3 realisations:
Z = [rand(n, m) for m ∈ (3, 3)];
θ̂(Z)

# Apply the estimator to two sets containing 3 and 4 realisations, respectively:
Z = [rand(n, m) for m ∈ (3, 4)];
θ̂(Z)
```
"""
struct DeepSetExpert{F, G, H, K}
	ψ::G
	ϕ::F
	S::H
	a::K
end
#TODO make this a superclass of DeepSet? Would be better to have a single class
# that dispatches to different methods depending on wether S is present or not.

Flux.@functor DeepSetExpert
Flux.trainable(d::DeepSetExpert) = (d.ψ, d.ϕ)

DeepSetExpert(ψ, ϕ, S; a::String = "mean") = DeepSetExpert(ψ, ϕ, S, _agg(a))
DeepSetExpert(deepset::DeepSet, ϕ, S) = DeepSetExpert(deepset.ψ, ϕ, S, deepset.a)

Base.show(io::IO, D::DeepSetExpert) = print(io, "\nDeepSetExpert object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.a)\nExpert statistics: $(D.S)\nOuter network:  $(D.ϕ)")
Base.show(io::IO, m::MIME"text/plain", D::DeepSetExpert) = print(io, D)


# ---- Methods ----

function (d::DeepSetExpert)(Z::A) where {A <: AbstractArray{T, N}} where {T, N}
	t = d.a(d.ψ(Z))
	s = d.S(Z)
	u = vcat(t, s)
	d.ϕ(u)
end

function (d::DeepSetExpert)(tup::Tup) where {Tup <: Tuple{A, B}} where {A <: AbstractArray{T, N}, B <: AbstractVector{T}} where {T, N}
	Z = tup[1]
	x = tup[2]
	t = d.a(d.ψ(Z))
	s = d.S(Z)
	u = vcat(Z, s, x)
	d.ϕ(u)
end

# # Simple, intuitive (although inefficient) implementation using broadcasting:
# function (d::DeepSetExpert)(v::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
#   stackarrays(d.(Z))
# end

# Optimised version. This approach ensures that the neural networks ϕ and ρ are
# applied to arrays that are as large as possible, improving efficiency compared
# with the intuitive method above (particularly on the GPU):
# Note I can't take the gradient of this function... Might have to open an issue with Zygote.
function (d::DeepSetExpert)(Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}

	# Convert to a single large Array
	z = stackarrays(Z)

	# Apply the inner neural network to obtain the neural summary statistics
	ψa = d.ψ(z)

	# Compute the indices needed for aggregation and construct a tuple of colons
	# used to subset all but the last dimension of ψa.
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)

	# Construct the combined neural and expert summary statistics
	u = map(eachindex(Z)) do i
		idx = indices[i]
		t = d.a(ψa[colons..., idx])
		s = d.S(Z[i])
		u = vcat(t, s)
		u
	end
	u = stackarrays(u)

	# Apply the outer network
	d.ϕ(u)
end

function (d::DeepSetExpert)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A <: AbstractArray{T, N}, B <: AbstractVector{T}} where {T, N}

	Z = tup[1]
	X = tup[2]

	# Convert to a single large Array
	z = stackarrays(Z)

	# Apply the inner neural network to obtain the neural summary statistics
	ψa = d.ψ(z)

	# Compute the indices needed for aggregation and construct a tuple of colons
	# used to subset all but the last dimension of ψa.
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)

	# concatenate the neural summary statistics with X
	u = map(eachindex(Z)) do i
		idx = indices[i]
		t = d.a(ψa[colons..., idx])
		s = d.S(Z[i])
		x = X[i]
		u = vcat(t, s, x)
		u
	end
	u = stackarrays(u)

	# Apply the outer network
	d.ϕ(u)
end


"""
    GNNEstimator(propagation, globalpool, deepset)

A neural estimator based on a graph neural network (GNN). The `propagation`
module transforms graphical input data into a set of hidden feature graphs;
the `globalpool` module aggregates the feature graphs (graph-wise) into a single
hidden feature vector; and the `deepset` module maps the hidden feature vectors
onto the parameter space.

The data should be a `GNNGraph` or `AbstractVector{GNNGraph}`, where each graph
is associated with a single parameter vector. The graphs may contain sub-graphs
corresponding to independent replicates from the model.

# Examples
```
using NeuralEstimators
using Flux
using Flux: batch
using GraphNeuralNetworks
using Statistics: mean

# Create some graphs
d = 1             # dimension of the response variable
n₁, n₂ = 11, 27   # number of nodes
e₁, e₂ = 30, 50   # number of edges
g₁ = rand_graph(n₁, e₁, ndata=rand(d, n₁))
g₂ = rand_graph(n₂, e₂, ndata=rand(d, n₂))
g  = batch([g₁, g₂])

# propagation module
w = 5; o = 7
propagation = GNNChain(GraphConv(d => w), GraphConv(w => w), GraphConv(w => o))

# global pooling module
meanpool = GlobalPool(mean)

# Deep Set module
w = 32
p = 3
ψ₂ = Chain(Dense(o, w, relu), Dense(w, w, relu), Dense(w, w, relu))
ϕ₂ = Chain(Dense(w, w, relu), Dense(w, p))
deepset = DeepSet(ψ₂, ϕ₂)

# GNN estimator
est = GNNEstimator(propagation, meanpool, deepset)

# Apply the estimator to a single graph, a single graph containing sub-graphs,
# and a vector of graphs:
θ̂ = est(g₁)
θ̂ = est(g)
θ̂ = est([g₁, g₂, g])
```
"""
struct GNNEstimator{F, G, H}
	propagation::F      # propagation module
	globalpool::G       # global pooling module
	deepset::H          # Deep Set module to map the learned feature vector to the parameter space
end
@functor GNNEstimator


# The replicates in g are associated with a single parameter.
function (est::GNNEstimator)(g::GNNGraph)

	# Apply the graph-to-graph transformation
	g̃ = est.propagation(g)

	# Global pooling
	ḡ = est.globalpool(g̃)

	# Extract the graph level data (i.e., the pooled features).
	# h is a matrix with
	# 	nrows = number of feature graphs in final propagation layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
	#	ncols = number of original graphs (i.e., number of independent replicates).
	h = ḡ.gdata.u

	# Apply the Deep Set module to map to the parameter space.
	θ̂ = est.deepset(h)
end


# Internally, we combine the graphs when doing mini-batching, to
# fully exploit GPU parallelism. What is slightly different here is that,
# contrary to most applications, we have a multiple graphs associated with each
# label (usually, each graph is associated with a label).
function (est::GNNEstimator)(v::V) where {V <: AbstractVector{G}} where {G <: GNNGraph}

	# Simple, inefficient implementation for sanity checking. Note that this is
	# much slower than the efficient approach below.
	# θ̂ = stackarrays(est.(v))

	# Convert v to a super graph. Since each element of v is itself a super graph
	# (where each sub graph corresponds to an independent replicate), we need to
	# count the number of sub-graphs in each element of v for later use.
	# Specifically, we need to keep track of the indices to determine which
	# independent replicates are grouped together.
	m = numberreplicates(v)
	g = Flux.batch(v)
	# NB batch() causes array mutation, which means that this method
	# cannot be used for computing gradients during training. As a work around,
	# I've added a second method that takes both g and m. The user will not need
	# to use this method, it's only necessary internally during training.

	return est(g, m)
end

function (est::GNNEstimator)(g::GNNGraph, m::AbstractVector{I}) where {I <: Integer}

	# Apply the graph-to-graph transformation
	g̃ = est.propagation(g)

	# Global pooling
	ḡ = est.globalpool(g̃)

	# Extract the graph level features (i.e., the pooled features).
	# h is a matrix with,
	# 	nrows = number of features graphs in final propagation layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
	#	ncols = total number of original graphs (i.e., total number of independent replicates).
	h = ḡ.gdata.u

	# Split the features based on the original grouping.
	ng = length(m)
	cs = cumsum(m)
	indices = [(cs[i] - m[i] + 1):cs[i] for i ∈ 1:ng]
	h̃ = [h[:, idx] for idx ∈ indices]

	# Apply the Deep Set module to map to the parameter space.
	θ̂ = est.deepset(h̃)

	return θ̂
end








# ---- Deep Set pooling (dimension after pooling is greater than 1) ----

# Come back to this later; just get an example with global pooling working first
#
# w = 32
# R = 4
# ψ₁ = Chain(Dense(o, w, relu), Dense(w, w, relu), Dense(w, w, relu)) # NB should the input size just be one? I think so.
# ϕ₁ = Chain(Dense(w, w, relu), Dense(w, R))
# deepsetpool = DeepSet(ψ₁, ϕ₁)
#
# function (est::GNNEstimator)(g::GNNGraph)
#
# 	# Apply the graph-to-graph transformation, and then extract the node-level
# 	# features. This yields a matrix of size (H, N), where H is the number of
# 	# feature graphs in the final layer and N is the total number of nodes in
# 	# all graphs.
# 	x̃ = est.propagation(g).ndata[1] # node-level features
# 	H = size(x̃, 1)
#
# 	# NB: The following is only necessary for more complicated pooling layers.
# 	# Now split x̃ according to which graph it belongs to.
# 	# find the number of nodes in each graph, and construct IntegerRange objects
# 	# to index x̃ appropriately
# 	I = graph_indicator(g)
# 	ng = g.num_graphs
# 	n = [sum(I .== i) for i ∈ 1:ng]
# 	cs  = cumsum(n)
# 	indices = [(cs[i] - n[i] + 1):cs[i] for i ∈ 1:ng]
# 	x̃ = [x̃[:, idx] for idx ∈ indices] # NB maybe I can do this without creating this vector; see what I do for DeepSets (I don't think so, actually).
#
# 	# Apply an abitrary global pooling function to each feature graph
# 	# (i.e., each row of x̃). The pooling function should return a vector of length
# 	# equal to the number of graphs, and where each element is a vector of length RH,
# 	# where R is the number of elements in each graph after pooling.
# 	h = est.globalpool(x̃)
#
# 	# Apply the Deep Set module to map the learned feature vector to the
# 	# parameter space
# 	θ̂ = est.deepset(h)
#
# 	return θ̂
# end
#
# # # Assumes y is an Array{T, 2}, where the number of rows is H and the number of
# # # columns is equal to the number of nodes for the current graph
# # function DeepSetPool(deepset::DeepSet, y::M) where {M <: AbstractMatrix{T}} where {T}
# # 	y = [y[j, :] for j ∈ 1:size(y, 1)]
# # 	y = reshape.(y, 1, 1, :)
# # 	h = deepset(y)
# # 	vec(h)
# # end





# ---- Functions assuming that the propagation and globalpool layers have been wrapped in WithGraph() ----

# NB this is a low priority optimisation that is only useful if we are training
# with a fixed set of locations.

# function (est::GNNEstimator)(a::A) where {A <: AbstractArray{T, N}} where {T, N}
#
# 	# Apply the graph-to-graph transformation
# 	g̃ = est.propagation(a)
#
# 	# Global pooling
# 	# h is a matrix with,
# 	# 	nrows = number of features graphs in final propagation layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
# 	#	ncols = number of original graphs (i.e., number of independent replicates).
# 	h = est.globalpool(g̃)
#
# 	# Reshape matrix to three-dimensional arrays for compatibility with Flux
# 	o = size(h, 1)
# 	h = reshape(h, o, 1, :)
#
# 	# Apply the Deep Set module to map to the parameter space.
# 	θ̂ = est.deepset(h)
# end
#
#
# function (est::GNNEstimator)(v::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
#
# 	# Simple, less efficient implementation for sanity checking:
# 	θ̂ = stackarrays(est.(v))
#
# 	# # Convert v to a super graph. Since each element of v is itself a super graph
# 	# # (where each sub graph corresponds to an independent replicate), we need to
# 	# # count the number of sub-graphs in each element of v for later use.
# 	# # Specifically, we need to keep track of the indices to determine which
# 	# # independent replicates are grouped together.
# 	# m = est.propagation.g.num_graphs
# 	# m = repeat([m], length(v))
# 	#
# 	# g = Flux.batch(repeat([est.propagation.g], length(v)))
# 	# g = GNNGraph(g, ndata = (Z = stackarrays(v)))
# 	#
# 	# # Apply the graph-to-graph transformation
# 	# g̃ = est.propagation.model(g)
# 	#
# 	# # Global pooling
# 	# ḡ = est.globalpool(g̃)
# 	#
# 	# # Extract the graph level data (i.e., the pooled features).
# 	# # h is a matrix with,
# 	# # 	nrows = number of features graphs in final propagation layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
# 	# #	ncols = total number of original graphs (i.e., total number of independent replicates).
# 	# h = ḡ.gdata[1]
# 	#
# 	# # Split the data based on the original grouping
# 	# ng = length(v)
# 	# cs = cumsum(m)
# 	# indices = [(cs[i] - m[i] + 1):cs[i] for i ∈ 1:length(v)]
# 	# h = [h[:, idx] for idx ∈ indices]
# 	#
# 	# # Reshape matrices to three-dimensional arrays for compatibility with Flux
# 	# o = size(h[1], 1)
# 	# h = reshape.(h, o, 1, :)
# 	#
# 	# # Apply the Deep Set module to map to the parameter space.
# 	# θ̂ = est.deepset(h)
#
# 	return θ̂
# end




"""
    Compress(a, b)

Uses the scaled logistic function to compress the output of a neural network to
be between `a` and `b`.

The elements of `a` should be less than the corresponding element of `b`.

# Examples
```
using NeuralEstimators
using Flux

p = 3
a = [0.1, 4, 2]
b = [0.9, 9, 3]
l = Compress(a, b)
K = 10
θ = rand(p, K)
l(θ)

n = 20
Z = rand(n, K)
θ̂ = Chain(Dense(n, 15), Dense(15, p), l)
θ̂(Z)
```
"""
struct Compress{T}
  a::T
  b::T
  m::T
end
Compress(a, b) = Compress(a, b, (b + a) / 2)

(l::Compress)(θ) = l.a .+ (l.b - l.a) ./ (one(eltype(θ)) .+ exp.(-(θ .- l.m)))

Flux.@functor Compress
Flux.trainable(l::Compress) =  ()
