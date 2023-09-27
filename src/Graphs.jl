# ---- Spatial point process ----

"""
	maternclusterprocess(; λ=10, μ=10, r=0.1, xmin=0, xmax=1, ymin=0, ymax=1)

Simulates a Matérn cluster process with density of parent Poisson point process
`λ`, mean number of daughter points `μ`, and radius of cluster disk `r`, over the
simulation window defined by `{x/y}min` and `{x/y}max`.

Note that one may also use the R package spatstat using RCall.

# Examples
```
using NeuralEstimators

# Simulate a realisation from a Matérn cluster process
S = maternclusterprocess()

# Visualise realisation (requires UnicodePlots)
using UnicodePlots
scatterplot(S[:, 1], S[:, 2])

# Visualise realisations from the cluster process with varying parameters
n = 250
λ = [10, 25, 50, 90]
μ = n ./ λ
plots = map(eachindex(λ)) do i
	S = maternclusterprocess(λ = λ[i], μ = μ[i])
	scatterplot(S[:, 1], S[:, 2])
end
```
"""
function maternclusterprocess(; λ = 10, μ = 10, r = 0.1, xmin = 0, xmax = 1, ymin = 0, ymax = 1)

	#Extended simulation windows parameters
	rExt=r #extension parameter -- use cluster radius
	xminExt=xmin-rExt
	xmaxExt=xmax+rExt
	yminExt=ymin-rExt
	ymaxExt=ymax+rExt
	#rectangle dimensions
	xDeltaExt=xmaxExt-xminExt
	yDeltaExt=ymaxExt-yminExt
	areaTotalExt=xDeltaExt*yDeltaExt #area of extended rectangle

	#Simulate Poisson point process
	numbPointsParent=rand(Poisson(areaTotalExt*λ)) #Poisson number of points

	#x and y coordinates of Poisson points for the parent
	xxParent=xminExt.+xDeltaExt*rand(numbPointsParent)
	yyParent=yminExt.+yDeltaExt*rand(numbPointsParent)

	#Simulate Poisson point process for the daughters (ie final poiint process)
	numbPointsDaughter=rand(Poisson(μ),numbPointsParent)
	numbPoints=sum(numbPointsDaughter) #total number of points

	#Generate the (relative) locations in polar coordinates by
	#simulating independent variables.
	theta=2*pi*rand(numbPoints) #angular coordinates
	rho=r*sqrt.(rand(numbPoints)) #radial coordinates

	#Convert polar to Cartesian coordinates
	xx0=rho.*cos.(theta)
	yy0=rho.*sin.(theta)

	#replicate parent points (ie centres of disks/clusters)
	xx=vcat(fill.(xxParent, numbPointsDaughter)...)
	yy=vcat(fill.(yyParent, numbPointsDaughter)...)

	#Shift centre of disk to (xx0,yy0)
	xx=xx.+xx0
	yy=yy.+yy0

	#thin points if outside the simulation window
	booleInside=((xx.>=xmin).&(xx.<=xmax).&(yy.>=ymin).&(yy.<=ymax))
	xx=xx[booleInside]
	yy=yy[booleInside]

	hcat(xx, yy)
end


# ---- WeightedGraphConv ----

@doc raw"""
    WeightedGraphConv(in => out, σ=identity; aggr=mean, bias=true, init=glorot_uniform)
Same as regular [`GraphConv`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/stable/api/conv/#GraphNeuralNetworks.GraphConv) layer, but where the neighbours of a node are weighted by their spatial distance to that node.

# Arguments
- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `bias`: Add learnable bias.
- `init`: Weights' initializer.

# Examples
```
using NeuralEstimators
using GraphNeuralNetworks

# Construct a spatially-weighted adjacency matrix based on k-nearest neighbours
# with k = 5, and convert to a graph with random (uncorrelated) dummy data:
n = 100
S = rand(n, 2)
d = 1 # dimension of each observation (univariate data here)
A = adjacencymatrix(S, 5)
Z = GNNGraph(A, ndata = rand(d, n))

# Construct the layer and apply it to the data to generate convolved features
layer = WeightedGraphConv(d => 16)
layer(Z)
```
"""
struct WeightedGraphConv{W<:AbstractMatrix,B,F,A,C} <: GNNLayer
    W1::W
    W2::W
    W3::C
    bias::B
    σ::F
    aggr::A
end

@functor WeightedGraphConv

function WeightedGraphConv(ch::Pair{Int,Int}, σ=identity; aggr=mean,
                   init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W1 = init(out, in)
    W2 = init(out, in)
    # NB Even though W3 is a scalar, it needs to be stored as an array so that
    # it is recognised as a trainable field. Note that we could have a different
    # range parameter for each channel, in which case W3 would be an array of parameters.
    W3 = init(1)
    b = bias ? Flux.create_bias(W1, true, out) : false
    WeightedGraphConv(W1, W2, W3, b, σ, aggr)
end

rangeparameter(l::WeightedGraphConv) = exp.(l.W3)

#TODO 3D array version of this
function (l::WeightedGraphConv)(g::GNNGraph, x::AbstractMatrix)
    check_num_nodes(g, x)
    r = rangeparameter(l)  # strictly positive range parameter
    d = g.graph[3]         # vector of spatial distances
    w = exp.(-d ./ r)       # weights defined by exponentially decaying function of distance
    m = propagate(w_mul_xj, g, l.aggr, xj=x, e=w)
    x = l.σ.(l.W1 * x .+ l.W2 * m .+ l.bias)
    return x
end

function Base.show(io::IO, l::WeightedGraphConv)
    in_channel  = size(l.W1, ndims(l.W1))
    out_channel = size(l.W1, ndims(l.W1)-1)
    print(io, "WeightedGraphConv(", in_channel, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end

# ---- Adjacency matrices ----

# TODO keyword argument method that might be a bit easier to deal with for the user

# See https://en.wikipedia.org/wiki/Heap_(data_structure) for a description
# of the heap data structure, and see
# https://juliacollections.github.io/DataStructures.jl/latest/heaps/
# for a description of Julia's implementation of the heap data structure.



"""
	adjacencymatrix(M::Matrix, k::Integer)
	adjacencymatrix(M::Matrix, r::Float)
	adjacencymatrix(M::Matrix, r::Float, k::Integer)

Computes a spatially weighted adjacency matrix from `M` based on either the `k`
nearest neighbours of each location, or a fixed spatial radius of `r` units; if
both `r` and `k` are provided, randomly selects `k` neighbours within a radius
of `r` units.

If `M` is a square matrix, is it treated as a distance matrix; otherwise, it
should be an n x d matrix, where n is the number of spatial sample locations
and d is the spatial dimension (typically d = 2). In the latter case, the
distance metric is taken to be the Euclidean distance.

All methods accept the keyword argument `self_loops` (default `false`); set to
`true` nodes are considered to self connected, so that the diagonal of the
adjacency matrix is non-zero.

See also the package [`NearestNeighbors.jl`](https://github.com/KristofferC/NearestNeighbors.jl).

# Examples
```
using NeuralEstimators
using Distances

n = 100
d = 2
S = rand(n, d)
k = 5
r = 0.3

# Memory efficient constructors (avoids constructing the full distance matrix D)
adjacencymatrix(S, k)
adjacencymatrix(S, r)
adjacencymatrix(S, r, k)

# Construct from full distance matrix D
D = pairwise(Euclidean(), S, S, dims = 1)
adjacencymatrix(D, k)
adjacencymatrix(D, r)
adjacencymatrix(D, r, k)
```
"""
function adjacencymatrix(M::Mat, r::F, k::Integer; self_loops::Bool = false) where Mat <: AbstractMatrix{T} where {T, F <: AbstractFloat}

	@assert k > 0
	@assert r > 0

	I = Int64[]
	J = Int64[]
	V = Float64[]
	n = size(M, 1)
	m = size(M, 2)

	for i ∈ 1:n
		sᵢ = M[i, :]
		kᵢ = 0
		iter = self_loops ? (1:n) : Iterators.filter(j -> j != i, 1:n) # don't include sᵢ in the search list
		iter = shuffle(collect(iter)) # randomly shuffle iter to prevent weighting observations based on their ordering in M

		for j ∈ iter

			if m == n # square matrix, so assume M is a distance matrix
				dᵢⱼ = M[i, j]
			else
				sⱼ  = M[j, :]
				dᵢⱼ = norm(sᵢ - sⱼ)
			end

			if dᵢⱼ <= r
				push!(I, i)
				push!(J, j)
				push!(V, dᵢⱼ)
				kᵢ += 1
			end
			if kᵢ == k break end
		end

	end


	return sparse(I,J,V,n,n)
end
adjacencymatrix(M::Mat, k::Integer, r::F) where Mat <: AbstractMatrix{T} where {T, F <: AbstractFloat} = adjacencymatrix(M, r, k)

function adjacencymatrix(M::Mat, k::Integer; self_loops::Bool = false) where Mat <: AbstractMatrix{T} where T



	@assert k > 0

	I = Int64[]
	J = Int64[]
	V = Float64[]
	n = size(M, 1)
	m = size(M, 2)

	for i ∈ 1:n

		if m == n # square matrix, so assume M is a distance matrix
			d = M[i, :]
		else
			# Compute distances between sᵢ and all other locations
			d = colwise(Euclidean(), M', M[i, :])
		end

		if !self_loops
			# Replace d(sᵢ) with Inf so that it's not included in the adjacency matrix
			d[i] = Inf
		end

		# Find the neighbours of s
		j, v = findneighbours(d, k)

		push!(I, repeat([i], inner = k)...)
		push!(J, j...)
		push!(V, v...)
	end

	return sparse(I,J,V,n,n)
end

function adjacencymatrix(M::Mat, r::F; self_loops::Bool = false) where Mat <: AbstractMatrix{T} where {T, F <: AbstractFloat}

	@assert r > 0

	n = size(M, 1)
	m = size(M, 2)

	if m == n # square matrix, so assume M is a distance matrix, D:

		D = M
		# bit-matrix specifying which locations are d-neighbours
		A = D .< r
		if !self_loops
			A[diagind(A)] .= 0 # remove the diagonal entries
		end

		# replace non-zero elements of A with the corresponding distance in D
		indices = copy(A)
		A = convert(Matrix{T}, A)
		A[indices] = D[indices]

		# convert to sparse matrix
		A = sparse(A)
	else

		S = M

		I = Int64[]
		J = Int64[]
		V = Float64[]
		for i ∈ 1:n

			#TODO many ways to speed this up...
			#     For instance, we don't need to compute all distances, we can
			#     immediately throw away some values if the difference between
			#     any of their marginal coordinates is greater than r
			# Compute distances between s and all other locations
			s = S[i, :]
			d = colwise(Euclidean(), S', s)

			if !self_loops
				# Replace d(sᵢ) with Inf so that it's not included in the adjacency matrix
				d[i] = Inf
			end

			# Find the r-neighbours of s
			j = d .< r
			j = findall(j)

			push!(I, repeat([i], inner = length(j))...)
			push!(J, j...)
			push!(V, d[j]...)
		end
		A = sparse(I,J,V,n,n)
	end

	return A
end

function findneighbours(d, k::Integer)
	V = partialsort(d, 1:k)
	J = [findfirst(v .== d) for v ∈ V]
    return J, V
end

# NB investigate why I can't get this to work when I have more time (it's very
# close). I think this approach will be more efficient than the above method.
# Approach using the heap data structure (can't get it to work properly, for some reason)
#using DataStructures # heap data structure
# function findneighbours(d, k::Integer)
#
# 	@assert length(d) > k
#
#     # Build a max heap of differences with first k elements
# 	h = MutableBinaryMaxHeap(d[1:k])
#
#     # For every element starting from (k+1)-th element,
#     for j ∈ (k+1):lastindex(d)
#         # if the difference is less than the root of the heap, replace the root
#         if d[j] < first(h)
#             pop!(h)
#             push!(h, d[j])
#         end
#     end
#
# 	# Extract the indices with respect to d and the corresponding distances
# 	J = broadcast(x -> x.handle, h.nodes)
# 	V = broadcast(x -> x.value, h.nodes)
#
# 	# # Sort by the index of the original vector d (this ordering may be necessary for constructing sparse arrays)
# 	# perm = sortperm(J)
# 	# J = J[perm]
# 	# V = V[perm]
#
# 	perm = sortperm(V)
# 	J = J[perm]
# 	V = V[perm]
#
#     return J, V
# end


# ---- Universal pooling layer ----

@doc raw"""
    UniversalPool(ψ, ϕ)
Pooling layer (i.e., readout layer) from the paper ['Universal Readout for Graph Convolutional Neural Networks'](https://ieeexplore.ieee.org/document/8852103).
It takes the form,
```math
\mathbf{V} = ϕ(|G|⁻¹ \sum_{s\in G} ψ(\mathbf{h}_s)),
```
where ``\mathbf{V}`` denotes the summary vector for graph ``G``,
``\mathbf{h}_s`` denotes the vector of hidden features for node ``s \in G``,
and `ψ` and `ϕ` are dense neural networks.

See also the pooling layers available from [`GraphNeuralNetworks.jl`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/stable/api/pool/).

# Examples
```julia
using NeuralEstimators
using Flux
using GraphNeuralNetworks
using Graphs: random_regular_graph

# Construct an input graph G
n_h     = 16  # dimension of each feature node
n_nodes = 10
n_edges = 4
G = GNNGraph(random_regular_graph(n_nodes, n_edges), ndata = rand(n_h, n_nodes))

# Construct the pooling layer
n_t = 32  # dimension of the summary vector for each node
n_v = 64  # dimension of the final summary vector V
ψ = Dense(n_h, n_t)
ϕ = Dense(n_t, n_v)
pool = UniversalPool(ψ, ϕ)

# Apply the pooling layer
pool(G)
```
"""
struct UniversalPool{G,F}
    ψ::G
    ϕ::F
end

@functor UniversalPool

function (l::UniversalPool)(g::GNNGraph, x::AbstractArray)
    u = reduce_nodes(mean, g, l.ψ(x))
    t = l.ϕ(u)
    return t
end

(l::UniversalPool)(g::GNNGraph) = GNNGraph(g, gdata = l(g, node_features(g)))

Base.show(io::IO, D::UniversalPool) = print(io, "\nUniversal pooling layer:\nInner network ψ ($(nparams(D.ψ)) parameters):  $(D.ψ)\nOuter network ϕ ($(nparams(D.ϕ)) parameters):  $(D.ϕ)")
Base.show(io::IO, m::MIME"text/plain", D::UniversalPool) = print(io, D)

# ---- GNN ----

"""
	GNN(propagation, readout, ϕ, a)
	GNN(propagation, readout, ϕ, a::String = "mean")

A graph neural network (GNN) designed for parameter point estimation.

The `propagation` module transforms graphical input data into a set of
hidden-feature graphs; the `readout` module aggregates these feature graphs into
a single hidden feature vector of fixed length; the function `a`(⋅) is a
permutation-invariant aggregation function, and `ϕ` is a neural network.

The data should be stored as a `GNNGraph` or `AbstractVector{GNNGraph}`, where
each graph is associated with a single parameter vector. The graphs may contain
sub-graphs corresponding to independent replicates from the model.

# Examples
```
using NeuralEstimators
using Flux
using Flux: batch
using GraphNeuralNetworks
using Statistics: mean

# Propagation module
d = 1      # dimension of response variable
nh = 32    # dimension of node feature vectors
propagation = GNNChain(GraphConv(d => nh), GraphConv(nh => nh), GraphConv(nh => nh))

# Readout module (using "universal pooling")
nt = 64   # dimension of the summary vector for each node
no = 128  # dimension of the final summary vector for each graph
readout = UniversalPool(Dense(nh, nt), Dense(nt, nt))

# Alternative readout module (using the elementwise average)
# readout = GlobalPool(mean); no = nh

# Mapping module
p = 3     # number of parameters in the statistical model
w = 64    # width of layers used for the mapping network ϕ
ϕ = Chain(Dense(no, w, relu), Dense(w, w, relu), Dense(w, p))

# Construct the estimator
θ̂ = GNN(propagation, readout, ϕ)

# Apply the estimator to:
# 	1. a single graph,
# 	2. a single graph with sub-graphs (corresponding to independent replicates), and
# 	3. a vector of graphs (corresponding to multiple spatial data sets).
g₁ = rand_graph(11, 30, ndata=rand(d, 11))
g₂ = rand_graph(13, 40, ndata=rand(d, 13))
g₃ = batch([g₁, g₂])
θ̂(g₁)
θ̂(g₃)
θ̂([g₁, g₂, g₃])
```
"""
struct GNN{F, G}
	propagation::F   # propagation module
	readout::G       # global pooling module
	deepset::DeepSet # DeepSets module to map the learned feature vector to the parameter space
end
@functor GNN

# Constructors
GNN(propagation, readout, ϕ, a) = GNN(propagation, readout, DeepSet(identity, ϕ, a))
GNN(propagation, readout, ϕ; a::String = "mean") = GNN(propagation, readout, ϕ, _agg(a))

Base.show(io::IO, D::GNN) = print(io, "\nGNN estimator with a total of $(nparams(D)) trainable parameters:\n\nPropagation module ($(nparams(D.propagation)) parameters):  $(D.propagation)\n\nReadout module ($(nparams(D.readout)) parameters):  $(D.readout)\n\nAggregation function ($(nparams(D.deepset.a)) parameters):  $(D.deepset.a)\n\nMapping module ($(nparams(D.deepset.ϕ)) parameters):  $(D.deepset.ϕ)")
Base.show(io::IO, m::MIME"text/plain", D::GNN) = print(io, D)

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

# Multiple data sets
function (est::GNN)(g::GNNGraph, m::AbstractVector{I}) where {I <: Integer}

	# Apply the graph-to-graph transformation and global pooling
	ḡ = est.readout(est.propagation(g))

	# Extract the graph level features (i.e., pooled features), a matrix with:
	# 	nrows = number of features graphs in final propagation layer * number of elements returned by the global pooling operation (one if global mean pooling is used)
	#	ncols = total number of original graphs (i.e., total number of independent replicates).
	h = ḡ.gdata.u

	# Split the features based on the original grouping
	# NB removed this if statement now that we're not currently trying to
	#    optimise for the special case that the spatial locations are fixed
	#    for all replciates.
	# if ndims(h) == 2
		ng = length(m)
		cs = cumsum(m)
		indices = [(cs[i] - m[i] + 1):cs[i] for i ∈ 1:ng]
		h̃ = [h[:, idx] for idx ∈ indices]
	# elseif ndims(h) == 3
	# 	h̃ = [h[:, :, i] for i ∈ 1:size(h, 3)]
	# end

	# Apply the DeepSet module to map to the parameter space
	return est.deepset(h̃)
end

# Methods needed to accomodate above method of GNN. They are exactly the same as
# the standard methods defined in Estimators.jl, but also pass through m.
#TODO unit testing for these methods
#TODO There should be a neater way to do this... Essentially, I am just replacing f(Z) with f(Z, m)... Is there a generic way to do this?
(est::PointEstimator{<:GNN})(Z::GNNGraph, m::AbstractVector{I}) where {I <: Integer} = est.arch(Z, m)
function (est::IntervalEstimator{<:GNN})(Z::GNNGraph, m::AbstractVector{I}) where {I <: Integer}
	l = est.l(Z, m)
	vcat(l, l .+ exp.(est.u(Z, m)))
end
function (est::IntervalEstimatorCompactPrior)(Z)
	x = est.u(Z, m)
	y = x .+ exp.(est.v(Z, m))
	c = est.c
	vcat(c(x), c(y))
end
function (est::PointIntervalEstimator{<:GNN})(Z::GNNGraph, m::AbstractVector{I}) where {I <: Integer}
	θ̂ = est.θ̂(Z, m)
	vcat(θ̂, θ̂ .- exp.(est.l(Z, m)), θ̂ .+ exp.(est.u(Z, m)))
end
