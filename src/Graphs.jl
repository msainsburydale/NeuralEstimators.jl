# ---- Fixed-location graph representations ----

#TODO How will I get batching to work in this situation?

"""
	GNNGraphFixedStructure
A type used to efficiently store graphical data in situations where the graph
structure is fixed for all instances, which allows for important efficiency
optimisations that can substantially reduce training time. The type has two fields:

- `graph::GNNGraph`
- `group_indicator::AbstractVector{<:UnitRange}`

The field `graph` stores the graph structure and `ndata` as an array with
dimensions d × M × n,  where d is the dimension of the input data, M is the
total number of replicates of the graph, and n is the number of nodes in the graph.
The field `group_indicator` groups the replicates stored in `graph`, and hence
should be a partion of 1:M.

The convenience constructors take either:

- a single `GNNGraph` and a vector of 3D arrays,
- or a vector of `GNNGraph`s, all of which have the same graph structure and where `ndata` in each graph is a 3D array.

# Examples
```
using NeuralEstimators
using GraphNeuralNetworks

# Graph structure
n = 100                     # number of nodes in the graph
e = 200                     # number of edges in the graph
graph = rand_graph(n, e)    # fixed structure for all graphs

# Generate data
d     = 2                 # dimension of response variable
m     = rand(30:90, 5)    # number of replicates of the graph (5 groups of replicates)
ndata = [rand(d, m̃, n) for m̃ ∈ m]

# Fixed graph representation
g = GNNGraphFixedStructure(graph, ndata)
```
"""
struct GNNGraphFixedStructure
    graph::GNNGraph
	group_indicator::AbstractVector{<:UnitRange}
end
@functor GNNGraphFixedStructure

function GNNGraphFixedStructure(graph::GNNGraph, ndata::A, group_indicator::AbstractVector{<:UnitRange}) where A <: AbstractArray{T, 3} where T
	@assert graph.num_graphs == 1
	@assert maximum(maximum(group_indicator)) == size(ndata, 2) == sum(length.(group_indicator))
	@assert minimum(minimum(group_indicator)) == 1
	graph = GNNGraph(graph; ndata = ndata)
	GNNGraphFixedStructure(graph, group_indicator)
end
function GNNGraphFixedStructure(graph::GNNGraph, ndata::V) where V <: AbstractVector{A} where A <: AbstractArray{T, 3} where T
	@assert graph.num_graphs == 1 "`graph` should contain a single graph only"
	@assert length(unique(size.(ndata, 1))) == 1 "The first dimension of the arrays in `ndata` should all be of the same size"
	@assert all(size.(ndata, 3) .== graph.num_nodes) "The third dimension of the arrays in `ndata` should have size equal to the number of nodes in `graph`"

	# Create group_indicator
	K  = length(ndata)
	m  = size.(ndata, 2) # number of independent replicates for every element in ndata
	cs = cumsum(m)
	group_indicator = [(cs[k] - m[k] + 1):cs[k] for k ∈ 1:K]

	# Convert vector of arrays into a single large array
	# Note that the below code is equivalent to: ndata = cat(ndata...; dims = 2)
	# We use stackarrays() since it is lazy and cat() can be slow for large K
	perm = [1, 3, 2]
	ndata = permutedims(stackarrays(permutedims.(ndata, Ref(perm))), perm)
	@assert size(ndata, 2) == sum(m)

	GNNGraphFixedStructure(graph, ndata, group_indicator)
end
function GNNGraphFixedStructure(graphs::V) where V <: AbstractVector{GNNGraph}
	graph = graphs[1].graph
	@assert all(broadcast(x -> x.graph == graph, graphs)) "All graphs must have the same graph structure"
	@assert all(broadcast(x -> ndims(x.ndata) == 3, graphs)) "`All graphs must have `ndata` as a 3-dimensional array"
	ndata = broadcast(x -> x.ndata, graphs)
	GNNGraphFixedStructure(graph, ndata)
end

# ---- GNN ----

#TODO Change this to GNNEstimator

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
g₁ = Flux.batch([GNNGraph(g; ndata = x[:, i, :]) for i ∈ 1:m]) # regular storage
g₂ = GNNGraph(g; ndata = x)                                    # efficient storage
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
	propagation::F   # propagation module
	readout::G       # global pooling module
	deepset::H       # Deep Set module to map the learned feature vector to the parameter space
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
	u = ḡ.gdata.u
	h = dropsingleton(u) # drops the redundant third dimension in the "efficient" storage approach

	# Apply the Deep Set module to map to the parameter space.
	θ̂ = est.deepset(u)
end


function (est::GNN)(g::GNNGraphFixedStructure)
	g̃ = est.readout(est.propagation(g.graph))
	u = g̃.gdata.u
	u = [u[:, indices] for indices ∈ g.group_indicator]
	est.deepset(u)
end

# Multiple data sets
function (est::GNN)(v::V) where {V <: AbstractVector{G}} where {G <: GNNGraph}

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


# TODO remove the if statements (note that there are also some in numberreplicates()).

# Multiple data sets
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

# Note that `GNN` is currently more efficient than using
# `PropagateReadout` as the inner network of a `DeepSet`, because here we are
# able to invoke the efficient `array`-method of `DeepSet`.

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

#NB this is identical to the method for GNN
# Multiple data sets
# Internally, we combine the graphs when doing mini-batching to
# fully exploit GPU parallelism. What is slightly different here is that,
# contrary to most applications, we have a multiple graphs associated with each
# label (usually, each graph is associated with a label).
function (est::PropagateReadout)(v::V) where {V <: AbstractVector{G}} where {G <: GNNGraph}

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
