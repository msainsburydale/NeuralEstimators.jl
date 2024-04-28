#
# For a concrete example, we consider the spatial Gaussian-process model,
# ```math
# Z_{j} = Y(\boldsymbol{s}_{j}) + \epsilon_{j}, \quad j = 1, \dots, n,
# ```
# where $\boldsymbol{Z} \equiv (Z_{1}, \dots, Z_{n})'$ are data collected at locations $\{\boldsymbol{s}_{1}, \dots, \boldsymbol{s}_{n}\}$ in a spatial domain $\mathcal{D}$, $Y(\cdot)$ is a spatially-correlated mean-zero Gaussian process, and $\epsilon_j \sim N(0, \tau^2)$, $j = 1, \dots, n$ is Gaussian white noise with standard deviation $\tau > 0$. Here, we use the popular isotropic Matérn covariance function,
# ```math
# \text{cov}\big(Y(\boldsymbol{s}), Y(\boldsymbol{u})\big)
# =
# \sigma^2 \frac{2^{1 - \nu}}{\Gamma(\nu)} \Big(\frac{\|\boldsymbol{s} - \boldsymbol{u}\|}{\rho}\Big)^\nu K_\nu\Big(\frac{\|\boldsymbol{s} - \boldsymbol{u}\|}{\rho}\Big),
# \quad \boldsymbol{s}, \boldsymbol{u} \in \mathcal{D},
# ```
# where $\sigma^2$ is the marginal variance, $\Gamma(\cdot)$ is the gamma function, $K_\nu(\cdot)$ is the Bessel function of the second kind of order $\nu$, and $\rho > 0$ and $\nu > 0$ are range and smoothness parameters, respectively. For ease of illustration, we fix $\sigma^2 = 1$ and $\nu = 1$, which leaves two unknown parameters that need to be estimated, $\boldsymbol{\theta} \equiv (\tau, \rho)'$.

#NB I'm now set up to implement "local pooling", which might improve time and memory efficiency if implemented well

# ---- Spatial Graph ----

#TODO q is not currently being used in the example... think I need to allow Z to be a three-dimensional array
#TODO elsewhere in the package, I think I use d to denote the dimension of the response variable... this will cause confusion, so get this right (check with the papers)
#TODO documentation (keyword arguments: pyramid_pool, k = 10, maxmin = true)
@doc raw"""
	spatialgraph(S)
	spatialgraph(S, Z)
	spatialgraph(g::GNNGraph, Z)
Given data `Z` and spatial locations `S`, constructs a
[`GNNGraph`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/stable/api/gnngraph/#GNNGraph-type)
ready for use in a graph neural network.

Let $\mathcal{D} \subset \mathbb{R}^d$ denote the spatial domain of interest.
In the case that $m$ independent replicates are collected over the same set of
$n$ spatial locations,
```math
\{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset \mathcal{D},
```
`Z` and `S` should be given as $n \times m$ and $n \times d$ matrices,
respectively. Otherwise, in the case that $m$ independent replicates
are collected over differing sets of spatial locations,
```math
\{\mathbf{s}_{ij}, \dots, \mathbf{s}_{in_i}\} \subset \mathcal{D}, \quad i = 1, \dots, m,
```
`Z` should be given as an $m$-dimensional vector of $n_i$-dimensional vectors,
and `S` should be given as an $m$-dimensional vector of $n_i \times d$ matrices.

# Examples
```
using NeuralEstimators, BenchmarkTools

# Dimension of the response, number of replicates, and spatial dimension
q = 1  # dimension of response (here, univariate data)
m = 5  # number of replicates
d = 2  # spatial dimension

# Spatial locations fixed for all replicates
n = 100
S = rand(n, d)
Z = rand(n, m)
g = spatialgraph(S)
g = spatialgraph(g, Z)
g = spatialgraph(S, Z)
@btime spatialgraph($S)
@btime spatialgraph($S, pyramid_pool = false)

# Spatial locations varying between replicates
n = rand(50:100, m)
S = rand.(n, d)
Z = rand.(n)
g = spatialgraph(S)
g = spatialgraph(g, Z)
g = spatialgraph(S, Z)
@btime spatialgraph($S)
@btime spatialgraph($S, pyramid_pool = false)
```
"""
function spatialgraph(S::AbstractMatrix; store_S::Bool = false, pyramid_pool::Bool = false, k = 10, maxmin = true)
	ndata = DataStore()
	S = Float32.(S)
	A = adjacencymatrix(S, k; maxmin = maxmin)
	S = permutedims(S) # need final dimension to be n-dimensional
	if store_S
		ndata = (ndata..., S = S)
	end
	if pyramid_pool
		clusterings = computeclusters(S)
		ndata = (ndata..., clusterings = clusterings)
	end
	g = GNNGraph(A, ndata = ndata)

end
spatialgraph(S::AbstractVector; kwargs...) = batch(spatialgraph.(S)) # spatial locations varying between replicates
#TODO multivariate data with spatial locations varying between replicates

# Wrappers that allow data to be passed into an already-constructed graph
# (useful for partial simulation on the fly, with the parameters held fixed)
spatialgraph(g::GNNGraph, Z) = GNNGraph(g, ndata = (g.ndata..., Z = reshapeZ(Z)))
reshapeZ(Z::V) where V <: AbstractVector{A} where A <: AbstractArray = stackarrays(reshapeZ.(Z))
reshapeZ(Z::AbstractVector) = reshapeZ(reshape(Z, length(Z), 1))
reshapeZ(Z::AbstractMatrix) = reshapeZ(reshape(Z, 1, size(Z)...))
function reshapeZ(Z::A) where A <: AbstractArray{T, 3} where {T}
	# Z is given as a three-dimensional array, with
	# Dimension 1: q, dimension of the response variable (e.g., singleton with univariate data)
	# Dimension 2: n, number of spatial locations
	# Dimension 3: m, number of replicates
	# Permute dimensions 2 and 3 since GNNGraph requires final dimension to be n-dimensional
	permutedims(Float32.(Z), (1, 3, 2))
end

# Wrapper that allows Z to be included at construction time
function spatialgraph(S, Z; kwargs...)
	g = spatialgraph(S; kwargs...)
	spatialgraph(g, Z)
end

# ---- SpatialGraphConv ----

# import Flux: Bilinear
# # function (b::Bilinear)(Z::A) where A <: AbstractArray{T, 3} where T
# # 	@assert size(Z, 2) == 2
# # 	x = Z[:, 1, :]
# # 	y = Z[:, 2, :]
# # 	b(x, y)
# # end

@doc raw"""
    SpatialGraphConv(in => out, g=relu; aggr=mean, bias=true, init=glorot_uniform, width=16, f=relu, c=1)

Implements the graph convolution
```math
 \mathbf{h}^{(l)}_{j} =
 g\Big(
 \mathbf{\Gamma}_{\!1}^{(l)} \mathbf{h}^{(l-1)}_{j}
 +
 \mathbf{\Gamma}_{\!2}^{(l)} \bar{\mathbf{h}}^{(l)}_{j}
 +
 \mathbf{\gamma}^{(l)}
 \Big),
 \quad
 \bar{\mathbf{h}}^{(l)}_{j} = \sum_{j' \in \mathcal{N}(j)}\mathbf{w}(\mathbf{s}_j, \mathbf{s}_{j'}; \mathbf{\beta}^{(l)}) \odot \mathbf{h}^{(l-1)}_{j'},
```
where $\mathbf{h}^{(l)}_{j}$ is the feature vector at location
$\mathbf{s}_j$ at layer $l$, $g(\cdot)$ is a non-linear activation function
applied elementwise, $\mathbf{\Gamma}_{\!1}^{(l)}$ and
$\mathbf{\Gamma}_{\!2}^{(l)}$ are trainable parameter matrices,
$\mathbf{\gamma}^{(l)}$ is a trainable bias vector, $\mathcal{N}(j)$ denotes the
indices of neighbours of $\mathbf{s}_j$ and $j$ itself,
$\mathbf{w}(\cdot, \cdot)$ is a learnable weight function parameterised by
$\mathbf{\beta}^{(l)}$, and $\odot$ denotes elementwise multiplication. Note
that summation over $\mathcal{N}(j)$ may be replaced by another aggregation
function, such as the elementwise mean or maximum.

The function $\mathbf{w}(\cdot, \cdot)$ is modelled using a multilayer
perceptron with a single hidden layer. When modelling stationary processes, it
can be made a function of spatial displacement, so that
$\mathbf{w}(\mathbf{s}_j, \mathbf{s}_{j'}) \equiv \mathbf{w}(\mathbf{s}_{j'} - \mathbf{s}_j)$.
Similarly, when modelling isotropic processes, it can be made a
function of spatial distance, so that
$\mathbf{w}(\mathbf{s}_j, \mathbf{s}_{j'}) \equiv \mathbf{w}(\|\mathbf{s}_{j'} - \mathbf{s}_j\|)$.

The function $\mathbf{w}(\cdot, \cdot)$ returns a vector that is of the same
dimension as the feature vectors of the previous layer. At the first layer,
the "feature" vectors correspond to the spatial data and, for univariate
spatial processes, the dimension of $\mathbf{w}(\cdot, \cdot)$ will be equal to
one, which may be a source of inflexibility. To increase flexibility, one may
construct several "channels" in an analogous manner to conventional
convolution, specifically, by constructing the intermediate representation as

```math
\bar{\mathbf{h}}^{(l)}_{j} =
\sum_{j' \in \mathcal{N}(j)}
\Big(
\mathbf{w}(\mathbf{s}_j, \mathbf{s}_{j'}; \mathbf{\beta}_{1}^{(l)})
\oplus
\dots
\oplus
\mathbf{w}(\mathbf{s}_j, \mathbf{s}_{j'}; \mathbf{\beta}_{c}^{(l)})
\Big)
\odot
\Big(
 \mathbf{h}^{(l-1)}_{j'}
\oplus
\dots
\oplus
 \mathbf{h}^{(l-1)}_{j'}
\Big),
```
where $c$ denotes the number of channels and $\oplus$ denotes vector concatentation.

# Arguments
- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `g`: Activation function.
- `aggr`: Aggregation operator $\mathbf{a}(\cdot)$ (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `bias`: Add learnable bias?
- `init`: Weights' initializer.
- `width`: Width of the hidden layer of $\mathbf{w}(\cdot, \cdot)$.
- `f`: Activation function used in $\mathbf{w}(\cdot, \cdot)$.
- `c`: The number of "channels" of $\mathbf{w}(\cdot, \cdot)$.

# Examples
```
using NeuralEstimators, Flux, GraphNeuralNetworks

# Generate some toy data
m = 5            # number of replicates
d = 2            # spatial dimension
n = 100          # number of spatial locations
S = rand(n, d)   # spatial locations
Z = rand(n, m)   # toy data

# Construct the graph
g = spatialgraph(S, Z)

# Construct and apply spatial graph convolution layers
layer1 = SpatialGraphConv(1 => 16, c = 8)
layer2 = SpatialGraphConv(16 => 32)
layer2(layer1(g))

# With a skip connection
GNN = GNNChain(
	GraphSkipConnection(SpatialGraphConv(1 => 16)),
	SpatialGraphConv(16 + 1 => 32) # one extra input dimension corresponding to the input data
)
GNN(g)
```
"""
struct SpatialGraphConv{W<:AbstractMatrix,NN,B,F,A} <: GNNLayer
    Γ1::W
    Γ2::W
	b::B
	w::NN
    g::F
    a::A
	# d::Integer # TODO spatial dimension of the process (needed to define the input size of the weight function)
	# isotropic::Bool  # TODO if true, we just use the distances (should this be stored in the edge features)?
	# stationary::Bool # TODO if true, we just use the spatial displacements (should this be stored in the edge features)?
end
@layer SpatialGraphConv
WeightedGraphConv = SpatialGraphConv; export WeightedGraphConv # alias for backwards compatability
function SpatialGraphConv(
	ch::Pair{Int,Int},
	g = relu;
	aggr = mean,
	init = glorot_uniform,
	bias::Bool = true,
	width::Integer = 16,
	f = relu,
	c::Integer = 1)

	# Weight matrix
	in, out = ch
    Γ1 = init(out, in)
    Γ2 = init(out, in*c)

	# Bias vector
	b = bias ? Flux.create_bias(Γ1, true, out) : false

	# Spatial weighting function
	w_init = rand32 # initialise w with positive weights to prevent zero outputs
	w = map(1:c) do _
		Chain( # isotropic process
			Dense(1 => width, f, init = w_init),
			Dense(width => in, f, init = w_init)
			)
		#d = 2 # spatial dimension
		#w = Dense(d => in, f)          # stationary process
		#w = Bilinear((d, d) => in, f)  # nonstationary process (or no assumptions)
	end
	w = c == 1 ? w[1] : Parallel(vcat, w...)

    SpatialGraphConv(Γ1, Γ2, b, w, g, aggr)
end
function (l::SpatialGraphConv)(g::GNNGraph)
	Z = :Z ∈ keys(g.ndata) ? g.ndata.Z : first(values(g.ndata))
	h = l(g, Z)
	@ignore_derivatives GNNGraph(g, ndata = (g.ndata..., Z = h))
end
function (l::SpatialGraphConv)(g::GNNGraph, x::A) where A <: AbstractArray{T, 3} where {T}
    check_num_nodes(g, x)
	d = permutedims(g.graph[3]) # spatial distances
    e = l.w(d) # spatial weighting
	# repeat e to match the number of independent replicates
	# NB if e were a vector, could do h = propagate(w_mul_xj, g, l.a, xj=x, e=e)
	m = size(x, 2)
	e = permutedims(stackarrays([e for _ in 1:m], merge = false), (1, 3, 2))
	h = propagate(e_mul_xj, g, l.a, xj=x, e=e)
	l.g.(l.Γ1 ⊠ x .+ l.Γ2 ⊠ h .+ l.b) # ⊠ is shorthand for batched_mul
end
function Base.show(io::IO, l::SpatialGraphConv)
    in_channel  = size(l.Γ1, ndims(l.Γ1))
    out_channel = size(l.Γ1, ndims(l.Γ1)-1)
    print(io, "SpatialGraphConv(", in_channel, " => ", out_channel)
    l.g == identity || print(io, ", ", l.g)
    print(io, ", aggr=", l.a)
    print(io, ")")
end
# m = 5            # number of replicates
# d = 2            # spatial dimension
# n = 100          # number of spatial locations
# S = rand(n, d)   # spatial locations
# Z = rand(n, m)   # toy data
# g = spatialgraph(S, Z)
# x = g.ndata.Z
# l = SpatialGraphConv(1 => 16)
# d = permutedims(g.graph[3]) # spatial distances
# e = l.w(d)
# l(g)
# l(g).ndata.Z
# l = SpatialGraphConv(1 => 16, c = 5)
# d = permutedims(g.graph[3]) # spatial distances
# e = l.w(d)
# l(g)
# l(g).ndata.Z

#TODO document
struct GraphSkipConnection{T} <: GNNLayer
	layers::T
end
@layer GraphSkipConnection
function (skip::GraphSkipConnection)(g::GNNGraph)
  h = skip.layers(g)
  x = cat(h.ndata.Z, g.ndata.Z; dims = 1)
  @ignore_derivatives GNNGraph(g, ndata = (g.ndata..., Z = x))
end
function Base.show(io::IO, b::GraphSkipConnection)
  print(io, "GraphSkipConnection(", b.layers, ")")
end


# ---- Clustering ----

# TODO suppress warning "clustering cost increased at iteration 1"
"""
	computeclusters(S::Matrix)
Computes hierarchical clusters based on K-means.

# Examples
```
# random set of locations
d = 2
n = 5000
S = rand(d, n)
computeclusters(S)
```
"""
function computeclusters(S::Matrix)

	# Note that if we just want random initial values, we can simply do:
	# K = [16, 4, 1]
	# permutedims(reduce(hcat, assignments.(kmeans.(Ref(S), K))))

	# To construct a grid of initial points, we needsquare numbers when d = 1,
	# cubic numbers when d=3, quartic numbers when d=4, etc.
	d = size(S, 1)
	# K = d ∈ [1, 2] ? [16, 4, 1] : (1:3).^d #TODO try with just one cluster layer
	K = d ∈ [1, 2] ? [16, 4, 1] : (1:3).^d #TODO try with just one cluster layer
	clusterings = map(K) do k
		# Compute initial seeds
		# Partition the domain in a consistent way, so that the spatial
		# relationship is predictable/consistent. Do this using the keyword
		# argument "init", which allows an integer vector of length kc that provides
		# the indices of points to use as initial seeds. So, we just provide the
		# inital points as a grid based on S, where the grid ordering is consistent.
		# The resulting clusters should then roughly align with this grid each time
		# a new set of locations is given.
		if k == 1
			τ = [0.5]
		else
			τ = (0:isqrt(k)-1)/(isqrt(k)-1)
		end
		S_quantiles = quantile.(eachrow(S), Ref(τ))
		init_points = permutedims(expandgrid(S_quantiles...))
		init = map(eachcol(init_points)) do s
			partialsortperm(vec(sum(abs.(S .- s), dims = 1)), 1)
		end
		# S[:, init] # points that will be used as initial cluster points
		@suppress_err assignments(kmeans(S, k; init = init))
	end
	permutedims(reduce(hcat, clusterings))
end


# TODO documentation
@doc raw"""
	SpatialPyramidPool(aggr)

Clusterings are stored as a matrix with $n$ columns, where each row
corresponds to a clustering at a different resolutions (each spatial
location belongs to a single cluster in a given resolution). The clusterings
can be stored in the graph object (so that the clustering algorithm need
only be called once for a given set of locations); if clusterings is not
present in the graph object, the layer will compute the clusterings
automatically (this is less efficient since the clusterings cannot be stored
for later use).

# Examples
```
using NeuralEstimators, Statistics

# Constants across the examples
q = 1   # univariate data
m = 5   # number of independent replicates
d = 2   # spatial dimension, D ⊂ ℜ²
layer = SpatialGraphConv(q => 16)
pool  = SpatialPyramidPool(mean)

# Spatial locations fixed for all replicates
n = 100
S = rand(n, d)
Z = rand(n, m)
g = spatialgraph(S, Z)
h = layer(g)
r = pool(h)

# Spatial locations varying between replicates
n = rand(50:100, m)
S = rand.(n, d)
Z = rand.(n)
g = spatialgraph(S, Z)
h = layer(g)
r = pool(h)
```
"""
struct SpatialPyramidPool{F} <: GNNLayer
    aggr::F
end
@layer SpatialPyramidPool
function (l::SpatialPyramidPool)(g::GNNGraph)

	@assert :clusterings ∈ keys(g.ndata) # could compute clusterings here, but more efficient not to plus it makes things more complicated

	# Input Z is an nₕ x m x n array, where nₕ is the number of hidden features of
	# each node in the final propagation layer
	Z = :Z ∈ keys(g.ndata) ? g.ndata.Z : first(values(g.ndata))

	# Extract clusterings, a cxn matrix with cᵣ the number of cluster resolutions
	clusterings = g.ndata.clusterings
	# TODO now that we hardcode the cluster sizes, can compute clusterings here

	# Pool the features over the clusterings
	if g.num_graphs == 1
		R = poolfeatures(Z, clusterings, l.aggr)
	else
		R = map(1:g.num_graphs) do i
			# NB getgraph() is very slow, don't use it here
			node_idx = findall(i .== g.graph_indicator)
			poolfeatures(Z[:, :, node_idx], clusterings[:, node_idx], l.aggr)
		end
		R = reduce(hcat, R)
	end

	# R is a Cnₕ x m matrix where C is the total number of clusters across all
	# clustering resolutions. It is now ready to be passed to the aggregation
	# function of the DeepSets architecture. Note that we cannot store R in
	# g.gdata, since it must have last dimension equal to the number of graphs
	# (we can leave the singleton dimension if we want to do this).
    return R
end
Base.show(io::IO, l::SpatialPyramidPool) = print(io, "\nSpatialPyramidPool with aggregation function $(l.aggr)")

function poolfeatures(Z, clusterings, aggr)
	R = map(eachrow(clusterings)) do clustering
		K = maximum(clustering)
		r = map(1:K) do k
			idx = findall(k .== clustering)
			h = Z[:, :, idx]
			aggr(h, dims = 3)
		end
		reduce(vcat, r)
	end
	R = reduce(vcat, R)
	R = dropdims(R; dims = 3)
end

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
using NeuralEstimators, Flux, GraphNeuralNetworks
using Graphs: random_regular_graph

# Construct an input graph G
n_h     = 16  # dimension of each feature node
n_nodes = 10
n_edges = 4
G = GNNGraph(random_regular_graph(n_nodes, n_edges), ndata = rand(Float32, n_h, n_nodes))

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
@layer UniversalPool
function (l::UniversalPool)(g::GNNGraph, x::AbstractArray)
    u = reduce_nodes(mean, g, l.ψ(x))
    t = l.ϕ(u)
    return t
end
(l::UniversalPool)(g::GNNGraph) = GNNGraph(g, gdata = l(g, node_features(g)))
Base.show(io::IO, D::UniversalPool) = print(io, "\nUniversal pooling layer:\nInner network ψ ($(nparams(D.ψ)) parameters):  $(D.ψ)\nOuter network ϕ ($(nparams(D.ϕ)) parameters):  $(D.ϕ)")


# ---- GNNSummary ----

#TODO can this be used with RatioEstimator, QuantileEstimatorDiscrete, and
# QuantileEstimatorContinuous? If so, state that in the example.
#TODO improve and update documentation
"""
	GNNSummary(propagation, readout)

A graph neural network (GNN) designed for parameter point estimation.

The `propagation` module transforms graphical input data into a set of
hidden-feature graphs; the `readout` module aggregates these feature graphs into
a single hidden feature vector of fixed length; the function `a`(⋅) is a
permutation-invariant aggregation function, and `ϕ` is a neural network. Expert,
user-defined summary statistics `S` can also be utilised, as described in [`DeepSet`](@ref).

The data should be stored as a `GNNGraph` or `Vector{GNNGraph}`, where
each graph is associated with a single parameter vector. The graphs may contain
subgraphs corresponding to independent replicates from the model.

# Examples
```
using NeuralEstimators, Flux, GraphNeuralNetworks
using Flux: batch
using Statistics: mean

# Propagation module
d = 1      # dimension of response variable
nh = 32    # dimension of node feature vectors
propagation = GNNChain(GraphConv(d => nh), GraphConv(nh => nh), GraphConv(nh => nh))

# Simple readout module, the elementwise average
readout = GlobalPool(mean)
no = nh # dimension of the final summary vector for each graph

# Summary network is the composition of the propagation and readout module
ψ = GNNSummary(propagation, readout)

# Mapping module
p = 3     # number of parameters in the statistical model
w = 64    # width of layers used for the mapping network ϕ
ϕ = Chain(Dense(no, w, relu), Dense(w, w, relu), Dense(w, p))

# Construct the estimator
θ̂ = DeepSet(ψ, ϕ)

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
struct GNNSummary{F, G}
	propagation::F   # propagation module
	readout::G       # readout module
end
@layer GNNSummary
Base.show(io::IO, D::GNNSummary) = print(io, "\nThe propagation and readout modules of a graph neural network (GNN), with a total of $(nparams(D)) trainable parameters:\n\nPropagation module ($(nparams(D.propagation)) parameters):  $(D.propagation)\n\nReadout module ($(nparams(D.readout)) parameters):  $(D.readout)")
#TODO Really don't like this function; it's name is not reflective of what it
# really does. Note also that the data
dropsingleton(x::AbstractMatrix) = x
dropsingleton(x::A) where A <: AbstractArray{T, 3} where T = dropdims(x, dims = 3)


function (pr::GNNSummary)(g::GNNGraph)

	# Propagation module, a graph-to-graph transformation
	h = pr.propagation(g)

	# Readout module, computes a fixed-length vector for each replicate
	# R is a matrix with:
	# nrows = number of readout summary statistics
	# ncols = number of independent replicates
	if isa(pr.readout, SpatialPyramidPool)
		R = pr.readout(h)
	else
		# Ensure that we can still use the standard pooling layers
		Z = :Z ∈ keys(h.ndata) ? h.ndata.Z : first(values(h.ndata))
		R = GNNGraph(h, gdata = pr.readout(h, Z))
		#dropsingleton(R.gdata.u) # drops the redundant third dimension
		R = R.gdata.u
		reshape(R, size(R, 1), :)
	end
end
# Code from GNN example:
# θ = sample(1)
# g = simulate(θ, 7)[1]
# ψ(g)
# θ = sample(2)
# # g = simulate(θ, 1:10) # TODO errors! Currently not allowed to have data sets with differing number of independent replicates
# g = simulate(θ, 5)
# g = Flux.batch(g)
# ψ(g)

# ---- Adjacency matrices ----

"""
	adjacencymatrix(M::Matrix, k::Integer; maxmin::Bool = false)
	adjacencymatrix(M::Matrix, r::Float)
	adjacencymatrix(M::Matrix, r::Float, k::Integer)

Computes a spatially weighted adjacency matrix from `M` based on either the
`k`-nearest neighbours of each location; all nodes within a disc of radius `r`;
or, if both `r` and `k` are provided, a random set of `k` neighbours with a disc
of radius `r`.

If `maxmin=false` (default) the `k`-nearest neighbours are chosen based on all points in
the graph. If `maxmin=true`, a so-called maxmin ordering is applied,
whereby an initial point is selected, and each subsequent point is selected to
maximise the minimum distance to those points that have already been selected.
Then, the neighbours of each point are defined as the `k`-nearest neighbours
amongst the points that have already appeared in the ordering.

If `M` is a square matrix, it is treated as a distance matrix; otherwise, it
should be an ``n`` x d matrix, where ``n`` is the number of spatial locations
and ``d`` is the spatial dimension (typically ``d`` = 2). In the latter case,
the distance metric is taken to be the Euclidean distance. Note that the maxmin
ordering currently requires a set of spatial locations (not a distance matrix).

By convention, we consider a location to neighbour itself and, hence,
`k`-neighbour methods will yield `k`+1 neighbours for each location. Note that
one may use `dropzeros!()` to remove these self-loops from the constructed
adjacency matrix (see below).

# Examples
```
using NeuralEstimators, Distances, SparseArrays

n = 100
d = 2
S = rand(n, d)
k = 10
r = 0.1

# Memory efficient constructors (avoids constructing the full distance matrix D)
adjacencymatrix(S, k)
adjacencymatrix(S, k; maxmin = true)
adjacencymatrix(S, r)
adjacencymatrix(S, r, k)

# Construct from full distance matrix D
D = pairwise(Euclidean(), S, S, dims = 1)
adjacencymatrix(D, k)
adjacencymatrix(D, r)
adjacencymatrix(D, r, k)

# Removing self-loops so that a location is not its own neighbour
adjacencymatrix(S, k) |> dropzeros!
```
"""
function adjacencymatrix(M::Mat, r::F, k::Integer) where Mat <: AbstractMatrix{T} where {T, F <: AbstractFloat}

	@assert k > 0
	@assert r > 0

	I = Int64[]
	J = Int64[]
	V = T[]
	n = size(M, 1)
	m = size(M, 2)

	for i ∈ 1:n
		sᵢ = M[i, :]
		kᵢ = 0
		iter = shuffle(collect(1:n)) # shuffle to prevent weighting observations based on their ordering in M

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

	A = sparse(I,J,V,n,n)


	return A
end
adjacencymatrix(M::Mat, k::Integer, r::F) where Mat <: AbstractMatrix{T} where {T, F <: AbstractFloat} = adjacencymatrix(M, r, k)

#NB would be good to add the keyword argument initialise_centre::Bool = true that makes the starting point fixed to the centre of the spatial domain. This point could just be the closest point to the average of the spatial coordinates.
function adjacencymatrix(M::Mat, k::Integer; maxmin::Bool = false, moralise::Bool = false) where Mat <: AbstractMatrix{T} where T

	@assert k > 0

	I = Int64[]
	J = Int64[]
	V = T[]
	n = size(M, 1)
	m = size(M, 2)

	if m == n # square matrix, so assume M is a distance matrix
		D = M
	else      # otherwise, M is a matrix of spatial locations
		S = M
	end

	if k >= n # more neighbours than observations: return a dense adjacency matrix
		if m != n
			D = pairwise(Euclidean(), S')
		end
		A = sparse(D)
	elseif !maxmin
		k += 1 # each location neighbours itself, so increase k by 1
		for i ∈ 1:n

			if m == n
				d = D[i, :]
			else
				# Compute distances between sᵢ and all other locations
				d = colwise(Euclidean(), S', S[i, :])
			end

			# Find the neighbours of s
			j, v = findneighbours(d, k)

			push!(I, repeat([i], inner = k)...)
			push!(J, j...)
			push!(V, v...)
		end
		A = sparse(I,J,V,n,n)
	else
		@assert m != n "`adjacencymatrix` with maxmin-ordering requires a matrix of spatial locations, not a distance matrix"
		ord     = ordermaxmin(S)          # calculate ordering
		Sord    = S[ord, :]               # re-order locations
		NNarray = findorderednn(Sord, k)  # find k nearest neighbours/"parents"
		R = builddag(NNarray, T)          # build DAG
		A = moralise ?  R' * R : R        # moralise

		# Add distances to A
		# TODO This is memory inefficient, especially for large n;
		# only optimise if we find that this approach works well and this is a bottleneck
		D = pairwise(Euclidean(), Sord')
		I, J, V = findnz(A)
		indices = collect(zip(I,J))
		indices = CartesianIndex.(indices)
		A.nzval .= D[indices]

		# "unorder" back to the original ordering
		# Sanity check: Sord[sortperm(ord), :] == S
		# Sanity check: D[sortperm(ord), sortperm(ord)] == pairwise(Euclidean(), S')
		A = A[sortperm(ord), sortperm(ord)]
	end

	return A
end

# using NeuralEstimators, Distances, SparseArrays
# import NeuralEstimators: adjacencymatrix, ordermaxmin, findorderednn, builddag, findneighbours
# n = 5000
# d = 2
# S = rand(Float32, n, d)
# k = 10
# @elapsed adjacencymatrix(S, k; maxmin = true) # 10 seconds
# @elapsed adjacencymatrix(S, k) # 0.3 seconds
#
# @elapsed ord = ordermaxmin(S) # 0.57 seconds
# Sord    = S[ord, :]
# @elapsed NNarray = findorderednn(Sord, k) # 9 seconds... this is the bottleneck
# @elapsed R = builddag(NNarray)  # 0.02 seconds

function adjacencymatrix(M::Mat, r::F) where Mat <: AbstractMatrix{T} where {T, F <: AbstractFloat}

	@assert r > 0

	n = size(M, 1)
	m = size(M, 2)

	if m == n # square matrix, so assume M is a distance matrix, D:
		D = M
		A = D .< r # bit-matrix specifying which locations are d-neighbours

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
		V = T[]
		for i ∈ 1:n
			# Compute distances between s and all other locations
			s = S[i, :]
			d = colwise(Euclidean(), S', s)

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

# TODO this function is much, much slower than the R version... need to optimise
function getknn(S, s, k; args...)
  tree = KDTree(S; args...)
  nn_index, nn_dist = knn(tree, s, k, true)
  nn_index = hcat(nn_index...) |> permutedims # nn_index = stackarrays(nn_index, merge = false)'
  nn_dist  = hcat(nn_dist...)  |> permutedims # nn_dist  = stackarrays(nn_dist, merge = false)'
  nn_index, nn_dist
end

function ordermaxmin(S)

  # get number of locs
  n = size(S, 1)
  k = isqrt(n)
  # k is number of neighbors to search over
  # get the past and future nearest neighbors
  NNall = getknn(S', S', k)[1]
  # pick a random ordering
  index_in_position = [sample(1:n, n, replace = false)..., repeat([missing],1*n)...]
  position_of_index = sortperm(index_in_position[1:n])
  # loop over the first n/4 locations
  # move an index to the end if it is a
  # near neighbor of a previous location
  curlen = n
  nmoved = 0
  for j ∈ 2:2n
	nneigh = round(min(k, n /(j-nmoved+1)))
    nneigh = Int(nneigh)
   if !ismissing(index_in_position[j])
      neighbors = NNall[index_in_position[j], 1:nneigh]
      if minimum(skipmissing(position_of_index[neighbors])) < j
        nmoved += 1
        curlen += 1
        position_of_index[ index_in_position[j] ] = curlen
        rassign(index_in_position, curlen, index_in_position[j])
        index_in_position[j] = missing
    	end
  	end
  end
  ord = collect(skipmissing(index_in_position))

  return ord
end

# rowMins(X) = vec(mapslices(minimum, X, dims = 2))
# colMeans(X) = vec(mapslices(mean, X, dims = 1))
# function ordermaxmin_slow(S)
# 	n = size(S, 1)
# 	D = pairwise(Euclidean(), S')
# 	## Vecchia sequence based on max-min ordering: start with most central location
#   	vecchia_seq = [argmin(D[argmin(colMeans(D)), :])]
#   	for j in 2:n
#     	vecchia_seq_new = (1:n)[Not(vecchia_seq)][argmax(rowMins(D[Not(vecchia_seq), vecchia_seq, :]))]
# 		rassign(vecchia_seq, j, vecchia_seq_new)
# 	end
#   return vecchia_seq
# end

function rassign(v::AbstractVector, index::Integer, x)
	@assert index > 0
	if index <= length(v)
		v[index] = x
	elseif index == length(v)+1
		push!(v, x)
	else
		v = [v..., fill(missing, index - length(v) - 1)..., x]
	end
	return v
end

function findorderednnbrute(S, k::Integer)
  # find the k+1 nearest neighbors to S[j,] in S[1:j,]
  # by convention, this includes S[j,], which is distance 0
  n = size(S, 1)
  k = min(k,n-1)
  NNarray = Matrix{Union{Integer, Missing}}(missing, n, k+1)
  for j ∈ 1:n
	d = colwise(Euclidean(), S[1:j, :]', S[j, :])
    NNarray[j, 1:min(k+1,j)] = sortperm(d)[1:min(k+1,j)]
  end
  return NNarray
end

function findorderednn(S, k::Integer)

  # number of locations
  n = size(S, 1)
  k = min(k,n-1)
  mult = 2

  # to store the nearest neighbor indices
  NNarray = Matrix{Union{Integer, Missing}}(missing, n, k+1)

  # find neighbours of first mult*k+1 locations by brute force
  maxval = min( mult*k + 1, n )
  NNarray[1:maxval, :] = findorderednnbrute(S[1:maxval, :],k)

  query_inds = min( maxval+1, n):n
  data_inds = 1:n
  ksearch = k
  while length(query_inds) > 0
    ksearch = min(maximum(query_inds), 2ksearch)
    data_inds = 1:min(maximum(query_inds), n)
	NN = getknn(S[data_inds, :]', S[query_inds, :]', ksearch)[1]

    less_than_l = hcat([NN[l, :] .<= query_inds[l] for l ∈ 1:size(NN, 1)]...) |> permutedims
	sum_less_than_l = vec(mapslices(sum, less_than_l, dims = 2))
    ind_less_than_l = findall(sum_less_than_l .>= k+1)
	NN_k = hcat([NN[l,:][less_than_l[l,:]][1:(k+1)] for l ∈ ind_less_than_l]...) |> permutedims
    NNarray[query_inds[ind_less_than_l], :] = NN_k

    query_inds = query_inds[Not(ind_less_than_l)]
  end

  return NNarray
end

function builddag(NNarray, T = Float32)
  n, k = size(NNarray)
  I = [1]
  J = [1]
  V = T[1.0] # NB would be better if we could inherit the eltype somehow
  for j in 2:n
    i = NNarray[j, :]
    i = collect(skipmissing(i))
    push!(J, repeat([j], length(i))...)
    push!(I, i...)
	push!(V, repeat([1], length(i))...)
  end
  R = sparse(I,J,V,n,n)
  return R
end


# n=100
# S = rand(n, 2)
# k=5
# ord = ordermaxmin(S)              # calculate maxmin ordering
# Sord = S[ord, :];                 # reorder locations
# NNarray = findorderednn(Sord, k)  # find k nearest neighbours/"parents"
# R = builddag(NNarray)             # build the DAG
# Q = R' * R                        # moralise




"""
	maternclusterprocess(; λ=10, μ=10, r=0.1, xmin=0, xmax=1, ymin=0, ymax=1)

Simulates a Matérn cluster process with density of parent Poisson point process
`λ`, mean number of daughter points `μ`, and radius of cluster disk `r`, over the
simulation window defined by `x/ymin` and `xymax`.

See also the R package
[`spatstat`](https://cran.r-project.org/web/packages/spatstat/index.html),
which provides functions for simulating from a range of point processes and
which can be interfaced from Julia using
[`RCall`](https://juliainterop.github.io/RCall.jl/stable/).

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
