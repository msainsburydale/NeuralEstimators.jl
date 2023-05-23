using Functors: @functor
using RecursiveArrayTools: VectorOfArray, convert

# ---- Aggregation (pooling) and misc functions ----

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

"""
	samplesize(Z)

Computes the sample size m for a set of independent realisations `Z`, often
useful as an expert summary statistic in `DeepSetExpert` objects.

Note that this function is a simple wrapper around `numberreplicates`, but this
function returns the number of replicates as the eltype of `Z`.
"""
samplesize(Z) = eltype(Z)(numberreplicates(Z))
samplesize(Z::V) where V <: AbstractVector = samplesize.(Z)

# ---- DeepSet ----

"""
    DeepSet(ψ, ϕ, a)
	DeepSet(ψ, ϕ; a::String = "mean")

The Deep Set representation,

```math
θ̂(𝐙) = ϕ(𝐓(𝐙)),	 	 𝐓(𝐙) = 𝐚(\\{ψ(𝐙ᵢ) : i = 1, …, m\\}),
```

where 𝐙 ≡ (𝐙₁', …, 𝐙ₘ')' are independent replicates from the model, `ψ` and `ϕ`
are neural networks, and `a` is a permutation-invariant aggregation function.

To make the architecture agnostic to the sample size ``m``, the aggregation
function `a` must aggregate over the replicates. It can be specified as a
positional argument of type `Function`, or as a keyword argument with
permissible values `"mean"`, `"sum"`, and `"logsumexp"`.

`DeepSet` objects act on data stored as `Vector{A}`, where each
element of the vector is associated with one parameter vector (i.e., one set of
independent replicates), and where `A` depends on the form of the data and the
chosen architecture for `ψ`. As a rule of thumb, when the data are stored as an
array, the replicates are stored in the final dimension of the array. (This is
usually the 'batch' dimension, but batching with `DeepSets` is done at the set
level, i.e., sets of replicates are batched together.) For example, with
gridded spatial data and `ψ` a CNN, `A` should be
a 4-dimensional array, with the replicates stored in the 4ᵗʰ dimension.

Note that, internally, data stored as `Vector{Arrays}` are first
concatenated along the replicates dimension before being passed into the inner
neural network `ψ`; this means that `ψ` is applied to a single large array
rather than many small arrays, which can substantially improve computational
efficiency, particularly on the GPU.

Set-level information, ``𝐱``, that is not a function of the data can be passed
directly into the outer network `ϕ` in the following manner,

```math
θ̂(𝐙) = ϕ((𝐓(𝐙)', 𝐱')'),	 	 𝐓(𝐙) = 𝐚(\\{ψ(𝐙ᵢ) : i = 1, …, m\\}),
```

This is done by providing a `Tuple{Vector{A}, Vector{B}}`, where
the first element of the tuple contains the vector of data sets and the second
element contains the vector of set-level information.

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

# Apply the estimator
Z₁ = rand(n, 3);                  # single set of 3 realisations
Z₂ = [rand(n, m) for m ∈ (3, 3)]; # two sets each containing 3 realisations
Z₃ = [rand(n, m) for m ∈ (3, 4)]; # two sets containing 3 and 4 realisations
θ̂(Z₁)
θ̂(Z₂)
θ̂(Z₃)

# Repeat the above but with set-level information:
qₓ = 2
ϕ  = Chain(Dense(w + qₓ, w, relu), Dense(w, p));
θ̂  = DeepSet(ψ, ϕ)
x₁ = rand(qₓ)
x₂ = [rand(qₓ) for _ ∈ eachindex(Z₂)]
θ̂((Z₁, x₁))
θ̂((Z₂, x₂))
θ̂((Z₃, x₂))
```
"""
struct DeepSet{T, F, G}
	ψ::T
	ϕ::G
	a::F
end
DeepSet(ψ, ϕ; a::String = "mean") = DeepSet(ψ, ϕ, _agg(a))
@functor DeepSet
Base.show(io::IO, D::DeepSet) = print(io, "\nDeepSet object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.a)\nOuter network:  $(D.ϕ)")
Base.show(io::IO, m::MIME"text/plain", D::DeepSet) = print(io, D)


# Single data set
function (d::DeepSet)(Z::A) where A
	d.ϕ(d.a(d.ψ(Z)))
end

# Single data set with set-level covariates
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{A, B}} where {A, B <: AbstractVector{T}} where T
	Z = tup[1]
	x = tup[2]
	t = d.a(d.ψ(Z))
	d.ϕ(vcat(t, x))
end

# Multiple data sets: simple fallback method using broadcasting
function (d::DeepSet)(Z::V) where {V <: AbstractVector{A}} where A
  	stackarrays(d.(Z))
end

# Multiple data sets: optimised version for array data.
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
	t = stackarrays([d.a(ψa[colons..., idx]) for idx ∈ indices])

	# Apply the outer network
	θ̂ = d.ϕ(t)

	return θ̂
end

# Multiple data sets with set-level covariates
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B <: AbstractVector{T}} where {T}
	Z = tup[1]
	x = tup[2]
	t = d.a.(d.ψ.(Z))
	u = vcat.(t, x)
	stackarrays(d.ϕ.(u))
end

# Multiple data sets: optimised version for array data + vector set-level covariates.
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A <: AbstractArray{T, N}, B <: AbstractVector{T}} where {T, N}

	Z = tup[1]
	X = tup[2]

	# Almost exactly the same code as the method defined above, but here we also
	# concatenate the covariates X before passing them into the outer network
	z = stackarrays(Z)
	ψa = d.ψ(z)
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)
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


# ---- DeepSetExpert: DeepSet with expert summary statistics ----

# Note that this struct is necessary because the Vector{Array} method of
# `DeepSet` concatenates the arrays into a single large array before passing
# the data into ψ.
"""
	DeepSetExpert(ψ, ϕ, S, a)
	DeepSetExpert(ψ, ϕ, S; a::String = "mean")
	DeepSetExpert(deepset::DeepSet, ϕ, S)

Identical to `DeepSet`, but with additional expert summary statistics,

```math
θ̂(𝐙) = ϕ((𝐓(𝐙)', 𝐒(𝐙)')'),	 	 𝐓(𝐙) = 𝐚(\\{ψ(𝐙ᵢ) : i = 1, …, m\\}),
```

where `S` is a function that returns a vector of expert summary statistics.

The constructor `DeepSetExpert(deepset::DeepSet, ϕ, S)` inherits `ψ` and `a`
from `deepset`.

Similarly to `DeepSet`, set-level information can be incorporated by passing a
`Tuple`, in which case we have

```math
θ̂(𝐙) = ϕ((𝐓(𝐙)', 𝐒(𝐙)', 𝐱')'),	 	 𝐓(𝐙) = 𝐚(\\{ψ(𝐙ᵢ) : i = 1, …, m\\}).
```

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

# Apply the estimator
Z₁ = rand(n, 3);                  # single set
Z₂ = [rand(n, m) for m ∈ (3, 4)]; # two sets
θ̂(Z₁)
θ̂(Z₂)

# Repeat the above but with set-level information:
qₓ = 2
ϕ  = Chain(Dense(qₜ + qₛ + qₓ, w, relu), Dense(w, p));
θ̂  = DeepSetExpert(ψ, ϕ, S)
x₁ = rand(qₓ)
x₂ = [rand(qₓ) for _ ∈ eachindex(Z₂)]
θ̂((Z₁, x₁))
θ̂((Z₂, x₂))
```
"""
struct DeepSetExpert{F, G, H, K}
	ψ::G
	ϕ::F
	S::H
	a::K
end
Flux.@functor DeepSetExpert
Flux.trainable(d::DeepSetExpert) = (d.ψ, d.ϕ)
DeepSetExpert(ψ, ϕ, S; a::String = "mean") = DeepSetExpert(ψ, ϕ, S, _agg(a))
DeepSetExpert(deepset::DeepSet, ϕ, S) = DeepSetExpert(deepset.ψ, ϕ, S, deepset.a)
Base.show(io::IO, D::DeepSetExpert) = print(io, "\nDeepSetExpert object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.a)\nExpert statistics: $(D.S)\nOuter network:  $(D.ϕ)")
Base.show(io::IO, m::MIME"text/plain", D::DeepSetExpert) = print(io, D)

# Single data set
function (d::DeepSetExpert)(Z::A) where {A <: AbstractArray{T, N}} where {T, N}
	t = d.a(d.ψ(Z))
	s = d.S(Z)
	u = vcat(t, s)
	d.ϕ(u)
end

# Single data set with set-level covariates
function (d::DeepSetExpert)(tup::Tup) where {Tup <: Tuple{A, B}} where {A, B <: AbstractVector{T}} where T
	Z = tup[1]
	x = tup[2]
	t = d.a(d.ψ(Z))
	s = d.S(Z)
	u = vcat(t, s, x)
	d.ϕ(u)
end

# Multiple data sets: simple fallback method using broadcasting
function (d::DeepSetExpert)(Z::V) where {V <: AbstractVector{A}} where A
  	stackarrays(d.(Z))
end


# Multiple data sets: optimised version for array data.
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

# Multiple data sets with set-level covariates
function (d::DeepSetExpert)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B <: AbstractVector{T}} where {T}
	Z = tup[1]
	x = tup[2]
	t = d.a.(d.ψ.(Z))
	s = d.S.(Z)
	u = vcat.(t, s, x)
	stackarrays(d.ϕ.(u))
end


# Multiple data sets with set-level covariates: optimised version for array data.
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


# ---- GNN ----

# Note that this architecture is currently more efficient than using
# `PropagateReadout` as the inner network of a `DeepSet`, because here we are
# able to invoke the efficient `array`-method of `DeepSet`.

"""
    GNN(propagation, readout, deepset)

A graph neural network (GNN) designed for parameter estimation.

The `propagation` module transforms graphical input data into a set of
hidden-feature graphs; the `readout` module aggregates these feature graphs
(graph-wise) into a single hidden feature vector of fixed length; and the
`deepset` module maps the hidden feature vector onto the output space.

The data should be a `GNNGraph` or `AbstractVector{GNNGraph}`, where each graph
is associated with a single parameter vector. The graphs may contain sub-graphs
corresponding to independent replicates from the model. In cases where the
independent replicates are stored over a fixed set of nodes, one
may store the replicated data in the `ndata` field of a graph as a
three-dimensional array with dimensions d × m × n, where d is the dimension of
the response variable (i.e, d = 1 for univariate data), m is the
number of replicates of the graph, and n is the number of nodes in the graph.

# Examples
```
using NeuralEstimators
using Flux
using Flux: batch
using GraphNeuralNetworks
using Statistics: mean
using Test

# propagation and readout modules
d = 1; w = 5; o = 7
propagation = GNNChain(GraphConv(d => w), GraphConv(w => w), GraphConv(w => o))
readout     = GlobalPool(mean)

# DeepSet module
w = 32
p = 3
ψ = Chain(Dense(o, w, relu), Dense(w, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, p))
deepset = DeepSet(ψ, ϕ)

# GNN estimator
θ̂ = GNN(propagation, readout, deepset)

# Apply the estimator to a single graph, a single graph containing sub-graphs,
# and a vector of graphs:
n₁, n₂ = 11, 27                             # number of nodes
e₁, e₂ = 30, 50                             # number of edges
g₁ = rand_graph(n₁, e₁, ndata=rand(d, n₁))
g₂ = rand_graph(n₂, e₂, ndata=rand(d, n₂))
g₃ = batch([g₁, g₂])
θ̂(g₁)
θ̂(g₃)
θ̂([g₁, g₂, g₃])

@test size(θ̂(g₁)) == (p, 1)
@test size(θ̂(g₃)) == (p, 1)
@test size(θ̂([g₁, g₂, g₃])) == (p, 3)

# Efficient storage approach when the nodes do not vary between replicates:
n = 100                     # number of nodes in the graph
e = 200                     # number of edges in the graph
m = 30                      # number of replicates of the graph
g = rand_graph(n, e)        # fixed structure for all graphs
x = rand(d, m, n)
g₁ = Flux.batch([GNNGraph(g; ndata = x[:, i, :]) for i ∈ 1:m])
g₂ = GNNGraph(g; ndata = x)
θ₁ = θ̂(g₁)
θ₂ = θ̂(g₂)
@test size(θ₁) == (p, 1)
@test size(θ₂) == (p, 1)
@test all(θ₁ .≈ θ₂)

v₁ = [g₁, g₁]
v₂ = [g₂, g₂]
θ₁ = θ̂(v₁)
θ₂ = θ̂(v₂)
@test size(θ₁) == (p, 2)
@test size(θ₂) == (p, 2)
@test all(θ₁ .≈ θ₂)
```
"""
struct GNN{F, G, H}
	propagation::F      # propagation module
	readout::G       # global pooling module
	deepset::H          # Deep Set module to map the learned feature vector to the parameter space
end
@functor GNN


dropsingleton(x::AbstractMatrix) = x
dropsingleton(x::A) where A <: AbstractArray{T, 3} where T = dropdims(x, dims = 3)

# Single data set (replicates in g are associated with a single parameter).
function (est::GNN)(g::GNNGraph)

	# Apply the graph-to-graph transformation
	g̃ = est.propagation(g)

	# Global pooling
	ḡ = est.readout(g̃)

	# Extract the graph level data (i.e., the pooled features).
	# h is a matrix with
	# 	nrows = number of feature graphs in final propagation layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
	#	ncols = number of original graphs (i.e., number of independent replicates).
	h = ḡ.gdata.u
	h = dropsingleton(h) # drops the redundant third dimension in the "efficient" storage approach

	# Apply the Deep Set module to map to the parameter space.
	θ̂ = est.deepset(h)
end

# Multiple data sets
# (see also the Union{GNN, PropagateReadout} method defined below)
function (est::GNN)(g::GNNGraph, m::AbstractVector{I}) where {I <: Integer}

	# Apply the graph-to-graph transformation and global pooling
	ḡ = est.readout(est.propagation(g))

	# Extract the graph level features (i.e., pooled features), a matrix with:
	# 	nrows = number of features graphs in final propagation layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
	#	ncols = total number of original graphs (i.e., total number of independent replicates).
	h = ḡ.gdata.u

	# Split the features based on the original grouping
	if ndims(h) == 2
		ng = length(m)
		cs = cumsum(m)
		indices = [(cs[i] - m[i] + 1):cs[i] for i ∈ 1:ng]
		h̃ = [h[:, idx] for idx ∈ indices]
	elseif ndims(h) == 3
		h̃ = [h[:, :, i] for i ∈ 1:size(h, 3)]
	end

	# Apply the DeepSet module to map to the parameter space
	return est.deepset(h̃)
end


# ---- PropagateReadout ----

"""
    PropagateReadout(propagation, readout)

A module intended to act as the inner network `ψ` in a `DeepSet` or `DeepSetExpert`
architecture, performing the `propagation` and `readout` (global pooling)
transformations of a GNN.

The graphical data should be stored as a `GNNGraph` or `AbstractVector{GNNGraph}`,
where each graph is associated with a single parameter vector. The graphs may
contain sub-graphs corresponding to independent replicates from the model.

This approach is less efficient than [`GNN`](@ref) but *currently*
more flexible, as it allows us to exploit the `DeepSetExpert` architecture and
set-level covariate methods for `DeepSet`. It may be possible to improve the
efficiency of this approach by carefully defining specialised methods, or I
could make `GNN` more flexible, again by carefully defining specialised methods.

# Examples
```
using NeuralEstimators
using Flux
using Flux: batch
using GraphNeuralNetworks
using Statistics: mean

# Create some graph data
d = 1                                        # dimension of response variable
n₁, n₂ = 11, 27                              # number of nodes
e₁, e₂ = 30, 50                              # number of edges
g₁ = rand_graph(n₁, e₁, ndata = rand(d, n₁))
g₂ = rand_graph(n₂, e₂, ndata = rand(d, n₂))
g₃ = batch([g₁, g₂])

# propagation module and readout modules
w = 5; o = 7
propagation = GNNChain(GraphConv(d => w), GraphConv(w => w), GraphConv(w => o))
readout = GlobalPool(mean)

# DeepSet estimator with GNN for the inner network ψ
w = 32
p = 3
ψ = PropagateReadout(propagation, readout)
ϕ = Chain(Dense(o, w, relu), Dense(w, p))
θ̂ = DeepSet(ψ, ϕ)

# Apply the estimator to a single graph, a single graph containing sub-graphs,
# and a vector of graphs:
θ̂(g₁)
θ̂(g₃)
θ̂([g₁, g₂, g₃])

# Repeat the above but with set-level information:
qₓ = 2
ϕ = Chain(Dense(o + qₓ, w, relu), Dense(w, p))
θ̂ = DeepSet(ψ, ϕ)
x₁ = rand(qₓ)
x₂ = [rand(qₓ) for _ ∈ eachindex([g₁, g₂, g₃])]
θ̂((g₁, x₁))
θ̂((g₃, x₁))
θ̂(([g₁, g₂, g₃], x₂))

# Repeat the above but with expert statistics:
S = samplesize
qₛ = 1
ϕ = Chain(Dense(o + qₓ + qₛ, w, relu), Dense(w, p))
θ̂ = DeepSetExpert(ψ, ϕ, S)
θ̂((g₁, x₁))
θ̂((g₃, x₁))
θ̂(([g₁, g₂, g₃], x₂))
```
"""
struct PropagateReadout{F, G}
	propagation::F      # propagation module
	readout::G       # global pooling module
end
@functor PropagateReadout


# Single data set
function (est::PropagateReadout)(g::GNNGraph)

	# Apply the graph-to-graph transformation and global pooling
	ḡ = est.readout(est.propagation(g))

	# Extract the graph level data (i.e., pooled features), a matrix with:
	# 	nrows = number of feature graphs in final propagation layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
	#	ncols = number of original graphs (i.e., number of independent replicates).
	h = ḡ.gdata.u
	h = dropsingleton(h) # drops the redundant third dimension in the "efficient" storage approach

	return h
end


# Multiple data sets
# Internally, we combine the graphs when doing mini-batching to
# fully exploit GPU parallelism. What is slightly different here is that,
# contrary to most applications, we have a multiple graphs associated with each
# label (usually, each graph is associated with a label).
function (est::Union{GNN, PropagateReadout})(v::V) where {V <: AbstractVector{G}} where {G <: GNNGraph}

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


function (est::PropagateReadout)(g::GNNGraph, m::AbstractVector{I}) where {I <: Integer}

	# Apply the graph-to-graph transformation and global pooling
	ḡ = est.readout(est.propagation(g))

	# Extract the graph level features (i.e., pooled features), a matrix with:
	# 	nrows = number of features graphs in final propagation layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
	#	ncols = total number of original graphs (i.e., total number of independent replicates).
	h = ḡ.gdata.u

	# Split the features based on the original grouping
	if ndims(h) == 2
		ng = length(m)
		cs = cumsum(m)
		indices = [(cs[i] - m[i] + 1):cs[i] for i ∈ 1:ng]
		h̃ = [h[:, idx] for idx ∈ indices]
	elseif ndims(h) == 3
		h̃ = [h[:, :, i] for i ∈ 1:size(h, 3)]
	end

	# Return the hidden feature vector associated with each group of replicates
	return h̃
end

# ---- GraphConv ----

using Flux: batched_mul, ⊠
using GraphNeuralNetworks: check_num_nodes
import GraphNeuralNetworks: GraphConv
export GraphConv


"""
	(l::GraphConv)(g::GNNGraph, x::A) where A <: AbstractArray{T, 3} where {T}

Given an array `x` with dimensions d × m × n, where m is the
number of replicates of the graph and n is the number of nodes in the graph,
this method yields an array with dimensions `out` × m × n, where `out` is the
number of output channels for the given layer.

After global pooling, the pooled features are a three-dimenisonal array of size
`out` × m × 1, which is close to the format of the pooled features one would
obtain when "batching" the graph replicates into a single supergraph (in that
case, the the pooled features are a matrix of size `out` × m).

# Examples
```
using GraphNeuralNetworks
d = 2                       # dimension of response variable
n = 100                     # number of nodes in the graph
e = 200                     # number of edges in the graph
m = 30                      # number of replicates of the graph
g = rand_graph(n, e)        # fixed structure for all graphs
g.ndata.x = rand(d, m, n)   # node data varies between graphs

# One layer example:
out = 16
l = GraphConv(d => out)
l(g)
size(l(g)) # (16, 30, 100)

# Propagation and global-pooling modules:
gnn = GNNChain(
	GraphConv(d => out),
	GraphConv(out => out),
	GlobalPool(+)
)
gnn(g)
u = gnn(g).gdata.u
size(u)    # (16, 30, 1)

# check that gnn(g) == gnn(all_graphs)
using GraphNeuralNetworks
using Flux
using Test
d = 2                       # dimension of response variable
n = 100                     # number of nodes in the graph
e = 200                     # number of edges in the graph
m = 30                      # number of replicates of the graph
g = rand_graph(n, e)        # fixed structure for all graphs
out = 16
x = rand(d, m, n)
gnn = GNNChain(
	GraphConv(d => out),
	GraphConv(out => out),
	GlobalPool(+)
)
g₁ = Flux.batch([GNNGraph(g; ndata = x[:, i, :]) for i ∈ 1:m])
g₂ = GNNGraph(g; ndata = x)
gnn(g₁)
gnn(g₂)
u₁ = gnn(g₁).gdata.u
u₂ = gnn(g₂).gdata.u
y = gnn(g₂)
dropsingleton(y.gdata.u)

@test size(u₁)[1:2] == size(u₂)[1:2]
@test size(u₂, 3) == 1
@test all(u₁ .≈ u₂)
```
"""
function (l::GraphConv)(g::GNNGraph, x::A) where A <: AbstractArray{T, 3} where {T}
    check_num_nodes(g, x)
    m = GraphNeuralNetworks.propagate(copy_xj, g, l.aggr, xj = x)
    x = l.σ.(l.weight1 ⊠ x .+ l.weight2 ⊠ m .+ l.bias) # ⊠ is shorthand for batched_mul
	return x
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
# function (est::PropagateReadout)(g::GNNGraph)
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
# 	h = est.readout(x̃)
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





# ---- Functions assuming that the propagation and readout layers have been wrapped in WithGraph() ----

# NB this is a low priority optimisation that is only useful if we are training
# with a fixed set of locations.

# function (est::PropagateReadout)(a::A) where {A <: AbstractArray{T, N}} where {T, N}
#
# 	# Apply the graph-to-graph transformation
# 	g̃ = est.propagation(a)
#
# 	# Global pooling
# 	# h is a matrix with,
# 	# 	nrows = number of features graphs in final propagation layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
# 	#	ncols = number of original graphs (i.e., number of independent replicates).
# 	h = est.readout(g̃)
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
# function (est::PropagateReadout)(v::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
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
# 	# ḡ = est.readout(g̃)
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


@doc raw"""
    Compress(a, b, k = 1)
Layer that compresses its input to be within the range `a` and `b`, where each
element of `a` is less than the corresponding element of `b`.

The layer uses a logistic function,

```math
l(θ) = a + \frac{b - a}{1 + e^{-kθ}},
```

where the arguments `a` and `b` together combine to shift and scale the logistic
function to the desired range, and the growth rate `k` controls the steepness
of the curve.

The logistic function given [here](https://en.wikipedia.org/wiki/Logistic_function)
contains an additional parameter, θ₀, which is the input value corresponding to
the functions midpoint. In `Compress`, we fix θ₀ = 0, since the output of a
randomly initialised neural network is typically around zero.

# Examples
```
using NeuralEstimators
using Flux

a = [25, 0.5, -pi/2]
b = [500, 2.5, 0]
p = length(a)
K = 100
θ = randn(p, K)
l = Compress(a, b)
l(θ)

n = 20
θ̂ = Chain(Dense(n, p), l)
Z = randn(n, K)
θ̂(Z)
```
"""
struct Compress{T}
  a::T
  b::T
  k::T
end
Compress(a, b) = Compress(a, b, ones(eltype(a), length(a)))

(l::Compress)(θ) = l.a .+ (l.b - l.a) ./ (one(eltype(θ)) .+ exp.(-l.k .* θ))

Flux.@functor Compress
Flux.trainable(l::Compress) =  ()

# ---- SplitApply ----

"""
	SplitApply(layers, indices)
Splits an array into multiple sub-arrays by subsetting the rows using
the collection of `indices`, and then applies each layer in `layers` to the
corresponding sub-array.

Specifically, for each `i` = 1, …, ``n``, with ``n`` the number of `layers`,
`SplitApply(x)` performs `layers[i](x[indices[i], :])`, and then vertically
concatenates the resulting transformed arrays.

# Examples
```
using NeuralEstimators

d = 4
K = 50
p₁ = 2          # number of non-covariance matrix parameters
p₂ = d*(d+1)÷2  # number of covariance matrix parameters
p = p₁ + p₂

a = [0.1, 4]
b = [0.9, 9]
l₁ = Compress(a, b)
l₂ = CovarianceMatrix(d)
l = SplitApply([l₁, l₂], [1:p₁, p₁+1:p])

θ = randn(p, K)
l(θ)
```
"""
struct SplitApply{T,G}
  layers::T
  indices::G
end
Flux.@functor SplitApply (layers, )
Flux.trainable(l::SplitApply) = ()
function (l::SplitApply)(x::AbstractArray)
	vcat([layer(x[idx, :]) for (layer, idx) in zip(l.layers, l.indices)]...)
end


# ---- Cholesky, Covariance, and Correlation matrices ----

@doc raw"""
	CorrelationMatrix(d)
Layer for constructing the parameters of an unconstrained `d`×`d` correlation matrix.

The layer transforms a `Matrix` with `d`(`d`-1)÷2 rows into a `Matrix` with
the same dimension.

Internally, the layers uses the algorithm
described [here](https://mc-stan.org/docs/reference-manual/cholesky-factors-of-correlation-matrices-1.html#cholesky-factor-of-correlation-matrix-inverse-transform)
and [here](https://mc-stan.org/docs/reference-manual/correlation-matrix-transform.html#correlation-matrix-transform.section)
to construct a valid Cholesky factor 𝐋, and then extracts the strict lower
triangle from the positive-definite correlation matrix 𝐑 = 𝐋𝐋'. The strict lower
triangle is extracted and vectorised in line with Julia's column-major ordering.
For example, when modelling the correlation matrix,

```math
\begin{bmatrix}
1   & R₁₂ &  R₁₃ \\
R₂₁ & 1   &  R₂₃\\
R₃₁ & R₃₂ & 1\\
\end{bmatrix},
```

the rows of the matrix returned by a `CorrelationMatrix` layer will
be ordered as

```math
R₂₁, R₃₁, R₃₂,
```

which means that the output can easily be transformed into the implied
correlation matrices using the strict variant of [`vectotril`](@ref) and `Symmetric`.

# Examples
```
using NeuralEstimators
using LinearAlgebra

d = 4
p = d*(d-1)÷2
l = CorrelationMatrix(d)
θ = randn(p, 50)

# returns a matrix of parameters
θ = l(θ)

# convert matrix of parameters to implied correlation matrices
R = map(eachcol(θ)) do y
	R = Symmetric(cpu(vectotril(y, strict = true)), :L)
	R[diagind(R)] .= 1
	R
end
```
"""
struct CorrelationMatrix{T <: Integer, Q}
  d::T
  idx::Q
end
function CorrelationMatrix(d::Integer)
	idx = tril(trues(d, d), -1)
	idx = findall(vec(idx)) # convert to scalar indices
	return CorrelationMatrix(d, idx)
end
function (l::CorrelationMatrix)(x)
	p, K = size(x)
	L = [vectocorrelationcholesky(x[:, k]) for k ∈ 1:K]
	R = broadcast(x -> x*permutedims(x), L) # note that I replaced x' with permutedims(x) because Transpose/Adjoints don't work well with Zygote
	θ = broadcast(x -> x[l.idx], R)
	return hcat(θ...)
end
function vectocorrelationcholesky(v)
	ArrayType = containertype(v)
	v = cpu(v)
	z = tanh.(vectotril(v; strict=true))
	n = length(v)
	d = (-1 + isqrt(1 + 8n)) ÷ 2 + 1

	L = [ correlationcholeskyterm(i, j, z)  for i ∈ 1:d, j ∈ 1:d ]
	return convert(ArrayType, L)
end
function correlationcholeskyterm(i, j, z)
	T = eltype(z)
	if i < j
		zero(T)
	elseif 1 == i == j
		one(T)
	elseif 1 == j < i
		z[i, j]
	elseif 1 < j == i
		prod(sqrt.(one(T) .- z[i, 1:j-i].^2))
	else
		z[i, j] * prod(sqrt.(one(T) .- z[i, 1:j-i].^2))
	end
end



@doc raw"""
	CholeskyCovariance(d)
Layer for constructing the parameters of the lower Cholesky factor associated
with an unconstrained `d`×`d` covariance matrix.

The layer transforms a `Matrix` with `d`(`d`+1)÷2 rows into a `Matrix` of the
same dimension, but with `d` rows constrained to be positive (corresponding to
the diagonal elements of the Cholesky factor) and the remaining rows
unconstrained.

The ordering of the transformed `Matrix` aligns with Julia's column-major
ordering. For example, when modelling the Cholesky factor,

```math
\begin{bmatrix}
L₁₁ &     &     \\
L₂₁ & L₂₂ &     \\
L₃₁ & L₃₂ & L₃₃ \\
\end{bmatrix},
```

the rows of the matrix returned by a `CholeskyCovariance` layer will
be ordered as

```math
L₁₁, L₂₁, L₃₁, L₂₂, L₃₂, L₃₃,
```

which means that the output can easily be transformed into the implied
Cholesky factors using [`vectotril`](@ref).

# Examples
```
using NeuralEstimators

d = 4
p = d*(d+1)÷2
θ = randn(p, 50)
l = CholeskyCovariance(d)
θ = l(θ)                              # returns matrix (used for Flux networks)
L = [vectotril(y) for y ∈ eachcol(θ)] # convert matrix to Cholesky factors
```
"""
struct CholeskyCovariance{T <: Integer, G}
  d::T
  diag_idx::G
end
function CholeskyCovariance(d::Integer)
	diag_idx = [1]
	for i ∈ 1:(d-1)
		push!(diag_idx, diag_idx[i] + d-i+1)
	end
	CholeskyCovariance(d, diag_idx)
end
function (l::CholeskyCovariance)(x)
	p, K = size(x)
	y = [i ∈ l.diag_idx ? exp.(x[i, :]) : x[i, :] for i ∈ 1:p]
	permutedims(reshape(vcat(y...), K, p))
end

@doc raw"""
    CovarianceMatrix(d)
Layer for constructing the parameters of an unconstrained `d`×`d` covariance matrix.

The layer transforms a `Matrix` with `d`(`d`+1)÷2 rows into a `Matrix` of the
same dimension.

Internally, it uses a `CholeskyCovariance` layer to construct a
valid Cholesky factor 𝐋, and then extracts the lower triangle from the
positive-definite covariance matrix 𝚺 = 𝐋𝐋'. The lower triangle is extracted
and vectorised in line with Julia's column-major ordering. For example, when
modelling the covariance matrix,

```math
\begin{bmatrix}
Σ₁₁ & Σ₁₂ & Σ₁₃ \\
Σ₂₁ & Σ₂₂ & Σ₂₃ \\
Σ₃₁ & Σ₃₂ & Σ₃₃ \\
\end{bmatrix},
```

the rows of the matrix returned by a `CovarianceMatrix` layer will
be ordered as

```math
Σ₁₁, Σ₂₁, Σ₃₁, Σ₂₂, Σ₃₂, Σ₃₃,
```

which means that the output can easily be transformed into the implied
covariance matrices using [`vectotril`](@ref) and `Symmetric`.

# Examples
```
using NeuralEstimators
using LinearAlgebra

d = 4
p = d*(d+1)÷2
θ = randn(p, 50)

l = CovarianceMatrix(d)
θ = l(θ)
Σ = [Symmetric(cpu(vectotril(y)), :L) for y ∈ eachcol(θ)]
```
"""
struct CovarianceMatrix{T <: Integer, G}
  d::T
  idx::G
  choleskyparameters::CholeskyCovariance
end
function CovarianceMatrix(d::Integer)
	idx = tril(trues(d, d))
	idx = findall(vec(idx)) # convert to scalar indices
	return CovarianceMatrix(d, idx, CholeskyCovariance(d))
end

function (l::CovarianceMatrix)(x)
	L = _constructL(l.choleskyparameters, x)
	Σ = broadcast(x -> x*permutedims(x), L) # note that I replaced x' with permutedims(x) because Transpose/Adjoints don't work well with Zygote
	θ = broadcast(x -> x[l.idx], Σ)
	return hcat(θ...)
end

function _constructL(l::CholeskyCovariance, x)
	Lθ = l(x)
	K = size(Lθ, 2)
	L = [vectotril(view(Lθ, :, i)) for i ∈ 1:K]
	L
end

function _constructL(l::CholeskyCovariance, x::Array)
	Lθ = l(x)
	K = size(Lθ, 2)
	L = [vectotril(collect(view(Lθ, :, i))) for i ∈ 1:K]
	L
end

(l::CholeskyCovariance)(x::AbstractVector) = l(reshape(x, :, 1))
(l::CovarianceMatrix)(x::AbstractVector) = l(reshape(x, :, 1))
(l::CorrelationMatrix)(x::AbstractVector) = l(reshape(x, :, 1))


# ---- Withheld layers ----

# The following layers are withheld for now because the determinant constraint
# can cause exploding gradients during training. I may make these available
# in the future if I ever come up with a more stable way to implement the
# constraint.



# """
# `CholeskyCovarianceConstrained` constrains the `determinant` of the Cholesky
# factor. Since the determinant of a triangular matrix is equal to the product of
# its diagonal elements, the determinant is constrained by setting the final
# diagonal element equal to `determinant`/``(Π Lᵢᵢ)`` where the product is over
# ``i < d``.
# """
# struct CholeskyCovarianceConstrained{T <: Integer, G}
#   d::T
#   determinant::G
#   choleskyparameters::CholeskyCovariance
# end
# function CholeskyCovarianceConstrained(d, determinant = 1f0)
# 	CholeskyCovarianceConstrained(d, determinant, CholeskyCovariance(d))
# end
# function (l::CholeskyCovarianceConstrained)(x)
# 	y = l.choleskyparameters(x)
# 	u = y[l.choleskyparameters.diag_idx[1:end-1], :]
# 	v = l.determinant ./ prod(u, dims = 1)
# 	vcat(y[1:end-1, :], v)
# end
#
# """
# `CovarianceMatrixConstrained` constrains the `determinant` of the
# covariance matrix to `determinant`.
# """
# struct CovarianceMatrixConstrained{T <: Integer, G}
#   d::T
#   idx::G
#   choleskyparameters::CholeskyCovarianceConstrained
# end
# function CovarianceMatrixConstrained(d::Integer, determinant = 1f0)
# 	idx = tril(trues(d, d))
# 	idx = findall(vec(idx)) # convert to scalar indices
# 	return CovarianceMatrixConstrained(d, idx, CholeskyCovarianceConstrained(d, sqrt(determinant)))
# end
#
# (l::CholeskyCovarianceConstrained)(x::AbstractVector) = l(reshape(x, :, 1))
# (l::CovarianceMatrixConstrained)(x::AbstractVector) = l(reshape(x, :, 1))

# function _constructL(l::Union{CholeskyCovariance, CholeskyCovarianceConstrained}, x::Array)
# function (l::Union{CovarianceMatrix, CovarianceMatrixConstrained})(x)
# function _constructL(l::Union{CholeskyCovariance, CholeskyCovarianceConstrained}, x)

# @testset "CholeskyCovarianceConstrained" begin
# 	l = CholeskyCovarianceConstrained(d, 2f0) |> dvc
# 	θ̂ = l(θ)
# 	@test size(θ̂) == (p, K)
# 	@test all(θ̂[l.choleskyparameters.diag_idx, :] .> 0)
# 	@test typeof(θ̂) == typeof(θ)
# 	L = [vectotril(x) for x ∈ eachcol(θ̂)]
# 	@test all(det.(L) .≈ 2)
# 	testbackprop(l, dvc, p, K, d)
# end

# @testset "CovarianceMatrixConstrained" begin
# 	l = CovarianceMatrixConstrained(d, 4f0) |> dvc
# 	θ̂ = l(θ)
# 	@test size(θ̂) == (p, K)
# 	@test all(θ̂[l.choleskyparameters.choleskyparameters.diag_idx, :] .> 0)
# 	@test typeof(θ̂) == typeof(θ)
# 	testbackprop(l, dvc, p, K, d)
#
# 	Σ = [Symmetric(cpu(vectotril(y)), :L) for y ∈ eachcol(θ̂)]
# 	Σ = convert.(Matrix, Σ);
# 	@test all(isposdef.(Σ))
# 	@test all(det.(Σ) .≈ 4)
# end



# NB efficient version but not differentiable because it mutates arrays.
# I also couldn't find a way to adapt this approach (i.e., using calculations
# from previous columns) to make it differentiable.
# function vectocorrelationcholesky_nondifferentiable(v)
# 	ArrayType = containertype(v)
# 	v = cpu(v)
# 	z = tanh.(vectotril(v; strict=true))
# 	T = eltype(z)
# 	n = length(v)
# 	d = (-1 + isqrt(1 + 8n)) ÷ 2 + 1
#
# 	L = Matrix{T}(undef, d, d)
# 	for i ∈ 1:d
# 		for j ∈ 1:d
# 			if i < j
# 				L[i, j] = zero(T)
# 			elseif i == j
# 				if i == 1
# 					L[i, j] = one(T)
# 				else
# 					L[i, j] = sqrt(one(T) - sum(L[i, 1:j-1].^2))
# 				end
# 			else
# 				if j == 1
# 					L[i, j] = z[i, j]
# 				else
# 					L[i, j] = z[i, j] * sqrt(one(T) - sum(L[i, 1:j-1].^2))
# 				end
# 			end
# 		end
# 	end
#
# 	return convert(ArrayType, L)
# end

# function vectocorrelationcholesky_upper(v)
# 	ArrayType = containertype(v)
# 	v = cpu(v)
# 	z = tanh.(vectotriu(v; strict=true))
# 	n = length(v)
# 	d = (-1 + isqrt(1 + 8n)) ÷ 2 + 1
#
# 	U = [ uppercorrelationcholeskyterm_upper(i, j, z)  for i ∈ 1:d, j ∈ 1:d ]
# 	return convert(ArrayType, U)
# end
#
# function correlationcholeskyterm_upper(i, j, z)
# 	T = eltype(z)
# 	if i > j
# 		zero(T)
# 	elseif 1 == i == j
# 		one(T)
# 	elseif 1 == i < j
# 		z[i, j]
# 	elseif 1 < i == j
# 		prod(sqrt.(one(T) .- z[1:i-1, j].^2))
# 	else
# 		z[i, j] * prod(sqrt.(one(T) .- z[1:i-1, j].^2))
# 	end
# end
