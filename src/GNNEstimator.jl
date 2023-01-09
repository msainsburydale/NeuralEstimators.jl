"""
    GNNEstimator(graphtograph, globalpool, deepset)

A neural estimator based on a graph neural network (GNN). The `graphtograph`
module transforms graphical input data into a set of hidden feature graphs;
the `globalpool` module aggregates the feature graphs (graph-wise); and the
`deepset` module maps the aggregated feature vectors onto the parameter space.

Data structure: The data should be a `GNNGraph` or `AbstractVector{GNNGraph}`,
where each graph is associated with a single parameter vector. The graphs may
contain sub-graphs corresponding to independent replicates of the data
generating process.

# Examples
```
using Flux
using GraphNeuralNetworks
using Statistics: mean
n₁, n₂ = 11, 27
m₁, m₂ = 30, 50
d = 1
g₁ = rand_graph(n₁, m₁, ndata=rand(Float32, d, n₁))
g₂ = rand_graph(n₂, m₂, ndata=rand(Float32, d, n₂))
g = Flux.batch([g₁, g₂])

# graph-to-graph propagation module
w = 5
o = 7
graphtograph = GNNChain(GraphConv(d => w), GraphConv(w => w), GraphConv(w => o))

# global pooling module
meanpool = GlobalPool(mean)

# Deep Set module
w = 32
p = 3
ψ₂ = Chain(Dense(o, w, relu), Dense(w, w, relu), Dense(w, w, relu))
ϕ₂ = Chain(Dense(w, w, relu), Dense(w, p))
deepset = DeepSet(ψ₂, ϕ₂)

# Full estimator
est = GNNEstimator(graphtograph, meanpool, deepset)

# A single graph containing sub-graphs
θ̂ = est(g)

# A vector of graphs
v = [g₁, g₂, Flux.batch([g₁, g₂])]
θ̂ = est(v)
```
"""
struct GNNEstimator{F, G, H}
	graphtograph::F     # graph-to-graph propagation module
	globalpool::G       # global pooling module
	deepset::H          # Deep Set module to map the learned feature vector to the parameter space
end
@functor GNNEstimator


# The replicates in g are associated with a single parameter.
function (est::GNNEstimator)(g::GNNGraph)

	# Apply the graph-to-graph transformation
	g̃ = est.graphtograph(g)

	# Global pooling
	ḡ = est.globalpool(g̃)

	# Extract the graph level data (i.e., the pooled features).
	# h is a matrix with
	# 	nrows = number of features graphs in final graphtograph layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
	#	ncols = number of original graphs (i.e., number of independent replicates).
	h = ḡ.gdata[1]

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
	m = _numberreplicates(v)
	g = Flux.batch(v)
	# NB batch() causes array mutation, which means that this method
	# cannot be used for computing gradients during training. As a work around,
	# I've added a second method that takes both g and m. The user will not need
	# to use this method, it's only necessary internally during training.

	return est(g, m)
end

function (est::GNNEstimator)(g::GNNGraph, m::AbstractVector{I}) where {I <: Integer}

	# Apply the graph-to-graph transformation
	g̃ = est.graphtograph(g)

	# Global pooling
	ḡ = est.globalpool(g̃)

	# Extract the graph level features (i.e., the pooled features).
	# h is a matrix with,
	# 	nrows = number of features graphs in final graphtograph layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
	#	ncols = total number of original graphs (i.e., total number of independent replicates).
	h = ḡ.gdata[1]

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
# 	x̃ = est.graphtograph(g).ndata[1] # node-level features
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
# 	x̃ = [x̃[:, idx] for idx ∈ indices] # TODO maybe I can do this without creating this vector; see what I do for DeepSets (I don't think so, actually).
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





# ---- Functions assuming that the graphtograph and globalpool layers have been wrapped in WithGraph() ----

# NB this is a low priority optimisation that is only useful if we are training
# with a fixed set of locations.

# function (est::GNNEstimator)(a::A) where {A <: AbstractArray{T, N}} where {T, N}
#
# 	# Apply the graph-to-graph transformation
# 	g̃ = est.graphtograph(a)
#
# 	# Global pooling
# 	# h is a matrix with,
# 	# 	nrows = number of features graphs in final graphtograph layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
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
# 	# m = est.graphtograph.g.num_graphs
# 	# m = repeat([m], length(v))
# 	#
# 	# g = Flux.batch(repeat([est.graphtograph.g], length(v)))
# 	# g = GNNGraph(g, ndata = (Z = stackarrays(v)))
# 	#
# 	# # Apply the graph-to-graph transformation
# 	# g̃ = est.graphtograph.model(g)
# 	#
# 	# # Global pooling
# 	# ḡ = est.globalpool(g̃)
# 	#
# 	# # Extract the graph level data (i.e., the pooled features).
# 	# # h is a matrix with,
# 	# # 	nrows = number of features graphs in final graphtograph layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
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
