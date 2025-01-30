@doc raw"""
	spatialgraph(S)
	spatialgraph(S, Z)
	spatialgraph(g::GNNGraph, Z)
Given spatial data `Z` measured at spatial locations `S`, constructs a
[`GNNGraph`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/stable/api/gnngraph/#GNNGraph-type)
ready for use in a graph neural network that employs [`SpatialGraphConv`](@ref) layers. 

When $m$ independent replicates are collected over the same set of
$n$ spatial locations,
```math
\{\boldsymbol{s}_1, \dots, \boldsymbol{s}_n\} \subset \mathcal{D},
```
where $\mathcal{D} \subset \mathbb{R}^d$ denotes the spatial domain of interest, 
`Z` should be given as an $n \times m$ matrix and `S` should be given as an $n \times d$ matrix. 
Otherwise, when $m$ independent replicates
are collected over differing sets of spatial locations,
```math
\{\boldsymbol{s}_{ij}, \dots, \boldsymbol{s}_{in_i}\} \subset \mathcal{D}, \quad i = 1, \dots, m,
```
`Z` should be given as an $m$-vector of $n_i$-vectors, and `S` should be given as an $m$-vector of $n_i \times d$ matrices.

The spatial information between neighbours is stored as an edge feature, with the specific 
information controlled by the keyword arguments `stationary` and `isotropic`. 
Specifically, the edge feature between node $j$ and node $j'$ stores the spatial 
distance $\|\boldsymbol{s}_{j'} - \boldsymbol{s}_j\|$ (if `isotropic`), the spatial 
displacement $\boldsymbol{s}_{j'} - \boldsymbol{s}_j$ (if `stationary`), or the matrix of  
locations $(\boldsymbol{s}_{j'}, \boldsymbol{s}_j)$ (if `!stationary`).  

Additional keyword arguments inherit from [`adjacencymatrix()`](@ref) to determine the neighbourhood of each node, with the default being a randomly selected set of 
`k=30` neighbours within a disc of radius `r=0.15` units.

# Examples
```
using NeuralEstimators

# Number of replicates and spatial dimension
m = 5  
d = 2  

# Spatial locations fixed for all replicates
n = 100
S = rand(n, d)
Z = rand(n, m)
g = spatialgraph(S, Z)

# Spatial locations varying between replicates
n = rand(50:100, m)
S = rand.(n, d)
Z = rand.(n)
g = spatialgraph(S, Z)
```
"""
function spatialgraph(S::AbstractMatrix; stationary = true, isotropic = true, store_S::Bool = false, kwargs...)

	# Determine neighbourhood based on keyword arguments 
	kwargs = (;kwargs...)
	k = haskey(kwargs, :k) ? kwargs.k : 30
	r = haskey(kwargs, :r) ? kwargs.r : 0.15
	random = haskey(kwargs, :random) ? kwargs.random : false

	#TODO
	if !isotropic 
		error("Anistropy is not currently implemented (although it is documented in anticipation of future functionality); please contact the package maintainer")
	end
	if !stationary 
		error("Nonstationarity is not currently implemented (although it is documented anticipation of future functionality); please contact the package maintainer")
	end
	ndata = DataStore()
	S = f32(S)
	A = adjacencymatrix(S; k = k, r = r, random = random) 
	S = permutedims(S) # need final dimension to be n-dimensional
	if store_S
		ndata = (ndata..., S = S)
	end
	GNNGraph(A, ndata = ndata, edata = permutedims(A.nzval))
end
spatialgraph(S::AbstractVector; kwargs...) = batch(spatialgraph.(S; kwargs...)) # spatial locations varying between replicates


# Wrappers that allow data to be passed into an already-constructed graph
# (useful for partial simulation on the fly with the parameters held fixed)
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
	permutedims(f32(Z), (1, 3, 2))
end
function reshapeZ(Z::V) where V <: AbstractVector{M} where M <: AbstractMatrix{T} where T 
	# method for multidimensional processes with spatial locations varying between replicates
	z = reduce(hcat, Z)
	reshape(z, size(z, 1), 1, size(z, 2))
end 

# Wrapper that allows Z to be included at construction time
function spatialgraph(S, Z; kwargs...) 
	g = spatialgraph(S; kwargs...)
	spatialgraph(g, Z)
end

# NB Not documenting for now, but spatialgraph is set up for multivariate data. Eventually, we will write:
# "Let $q$ denote the dimension of the spatial process (e.g., $q = 1$ for 
# univariate spatial processes, $q = 2$ for bivariate processes, etc.)". For fixed locations, we will then write: 
# "`Z` should be given as a $q \times n \times m$ array (alternatively as an $n \times m$ matrix when $q = 1$) and `S` should be given as a $n \times d$ matrix."
# And for varying locations, we will write: 
# "`Z` should be given as an $m$-vector of $q \times n_i$ matrices (alternatively as an $m$-vector of $n_i$-vectors when $q = 1$), and `S` should be given as an $m$-vector of $n_i \times d$ matrices."
# Then update examples to show q > 1:
# # Examples
# ```
# using NeuralEstimators
#
# # Number of replicates, and spatial dimension
# m = 5  
# d = 2  
#
# # Spatial locations fixed for all replicates
# n = 100
# S = rand(n, d)
# Z = rand(n, m)
# g = spatialgraph(S)
# g = spatialgraph(g, Z)
# g = spatialgraph(S, Z)
#
# # Spatial locations varying between replicates
# n = rand(50:100, m)
# S = rand.(n, d)
# Z = rand.(n)
# g = spatialgraph(S)
# g = spatialgraph(g, Z)
# g = spatialgraph(S, Z)
#
# # Mutlivariate processes: spatial locations fixed for all replicates
# q = 2 # bivariate spatial process
# n = 100
# S = rand(n, d)
# Z = rand(q, n, m)  
# g = spatialgraph(S)
# g = spatialgraph(g, Z)
# g = spatialgraph(S, Z)
#
# # Mutlivariate processes: spatial locations varying between replicates
# n = rand(50:100, m)
# S = rand.(n, d)
# Z = rand.(q, n)
# g = spatialgraph(S)
# g = spatialgraph(g, Z) 
# g = spatialgraph(S, Z) 
# ```

@doc raw"""
	IndicatorWeights(h_max, n_bins::Integer)
	(w::IndicatorWeights)(h::Matrix) 
For spatial locations $\boldsymbol{s}$ and  $\boldsymbol{u}$, creates a spatial weight function defined as

```math 
\boldsymbol{w}(\boldsymbol{s}, \boldsymbol{u}) \equiv (\mathbb{I}(h \in B_k) : k = 1, \dots, K)',
```

where $\mathbb{I}(\cdot)$ denotes the indicator function, 
$h \equiv \|\boldsymbol{s} - \boldsymbol{u} \|$ is the spatial distance between $\boldsymbol{s}$ and 
$\boldsymbol{u}$, and $\{B_k : k = 1, \dots, K\}$ is a set of $K =$`n_bins` equally-sized distance bins covering the spatial distances between 0 and `h_max`. 

# Examples 
```
using NeuralEstimators 

h_max = 1
n_bins = 10
w = IndicatorWeights(h_max, n_bins)
h = rand(1, 30) # distances between 30 pairs of spatial locations 
w(h)
```
"""
struct IndicatorWeights{T} 
    h_cutoffs::T
end 
function IndicatorWeights(h_max, n_bins::Integer) 
	h_cutoffs = range(0, stop=h_max, length=n_bins+1)
	h_cutoffs = collect(h_cutoffs)
	IndicatorWeights(h_cutoffs)
end
function (l::IndicatorWeights)(h::M) where M <: AbstractMatrix{T} where T
	h_cutoffs = l.h_cutoffs
	bins_upper = h_cutoffs[2:end]   # upper bounds of the distance bins
	bins_lower = h_cutoffs[1:end-1] # lower bounds of the distance bins 
	N = [bins_lower[i:i] .< h .<= bins_upper[i:i] for i in eachindex(bins_upper)] # NB avoid scalar indexing by i:i
	N = reduce(vcat, N)
	f32(N)
end
Flux.trainable(l::IndicatorWeights) =  NamedTuple()


@doc raw"""
	KernelWeights(h_max, n_bins::Integer)
	(w::KernelWeights)(h::Matrix) 
For spatial locations $\boldsymbol{s}$ and  $\boldsymbol{u}$, creates a spatial weight function defined as

```math 
\boldsymbol{w}(\boldsymbol{s}, \boldsymbol{u}) \equiv (\exp(-(h - \mu_k)^2 / (2\sigma_k^2)) : k = 1, \dots, K)',
```

where $h \equiv \|\boldsymbol{s} - \boldsymbol{u}\|$ is the spatial distance between $\boldsymbol{s}$ and $\boldsymbol{u}$, and ${\mu_k : k = 1, \dots, K}$ and ${\sigma_k : k = 1, \dots, K}$ are the means and standard deviations of the Gaussian kernels for each bin, covering the spatial distances between 0 and h_max.

# Examples 
```
using NeuralEstimators 

h_max = 1
n_bins = 10
w = KernelWeights(h_max, n_bins)
h = rand(1, 30) # distances between 30 pairs of spatial locations 
w(h)
```
""" 
struct KernelWeights 
	mu 
	sigma 
end 
function KernelWeights(h_max, n_bins::Integer) 
	h_cutoffs = range(0, stop=h_max, length=n_bins+1) 
	h_cutoffs = collect(h_cutoffs)
	mu = [(h_cutoffs[i] + h_cutoffs[i+1]) / 2 for i in 1:n_bins] # midpoints of the intervals 
	sigma = [(h_cutoffs[i+1] - h_cutoffs[i]) / 4 for i in 1:n_bins] # std dev so that 95% of mass is within the bin 
	mu = f32(mu)
	sigma = f32(sigma)
	KernelWeights(mu, sigma) 
end 
function (l::KernelWeights)(h::M) where M <: AbstractMatrix{T} where T 
	mu = l.mu 
	sigma = l.sigma 
	N = [exp.(-(h .- mu[i:i]).^2 ./ (2 * sigma[i:i].^2)) for i in eachindex(mu)] # Gaussian kernel for each bin (NB avoid scalar indexing by i:i)
	N = reduce(vcat, N) 
	f32(N) 
end 
Flux.trainable(l::KernelWeights) = NamedTuple()


# ---- GraphConv ----

# 3D array version of GraphConv to allow the option to forego spatial information
"""
	(l::GraphConv)(g::GNNGraph, x::A) where A <: AbstractArray{T, 3} where {T}

Given a graph with node features consisting of a three dimensional array of size `in` × m × n, 
where n is the number of nodes in the graph, this method yields an array with 
dimensions `out` × m × n. 

# Examples
```
using NeuralEstimators, Flux, GraphNeuralNetworks

q = 2                       # dimension of response variable
n = 100                     # number of nodes in the graph
e = 200                     # number of edges in the graph
m = 30                      # number of replicates of the graph
g = rand_graph(n, e)        # fixed structure for all graphs
Z = rand(d, m, n)           # node data varies between graphs
g = GNNGraph(g; ndata = Z)

# Construct and apply graph convolution layer
l = GraphConv(d => 16)
l(g)
```
"""
function (l::GraphConv)(g::GNNGraph, x::A) where A <: AbstractArray{T, 3} where {T}
    check_num_nodes(g, x)
    m = GraphNeuralNetworks.propagate(copy_xj, g, l.aggr, xj = x)
    l.σ.(l.weight1 ⊠ x .+ l.weight2 ⊠ m .+ l.bias) # ⊠ is shorthand for batched_mul
end

@doc raw"""
    SpatialGraphConv(in => out, g=relu; args...)

Implements a spatial graph convolution for isotropic spatial processes [(Sainsbury-Dale et al., 2025)](https://arxiv.org/abs/2310.02600), 

```math
 \boldsymbol{h}^{(l)}_{j} =
 g\Big(
 \boldsymbol{\Gamma}_{\!1}^{(l)} \boldsymbol{h}^{(l-1)}_{j}
 +
 \boldsymbol{\Gamma}_{\!2}^{(l)} \bar{\boldsymbol{h}}^{(l)}_{j}
 +
 \boldsymbol{\gamma}^{(l)}
 \Big),
 \quad
 \bar{\boldsymbol{h}}^{(l)}_{j} = \sum_{j' \in \mathcal{N}(j)}\boldsymbol{w}^{(l)}(\|\boldsymbol{s}_{j'} - \boldsymbol{s}_j\|) \odot f^{(l)}(\boldsymbol{h}^{(l-1)}_{j}, \boldsymbol{h}^{(l-1)}_{j'}),
```

where $\boldsymbol{h}^{(l)}_{j}$ is the hidden feature vector at location
$\boldsymbol{s}_j$ at layer $l$, $g(\cdot)$ is a non-linear activation function
applied elementwise, $\boldsymbol{\Gamma}_{\!1}^{(l)}$ and
$\boldsymbol{\Gamma}_{\!2}^{(l)}$ are trainable parameter matrices,
$\boldsymbol{\gamma}^{(l)}$ is a trainable bias vector, $\mathcal{N}(j)$ denotes the
indices of neighbours of $\boldsymbol{s}_j$, $\boldsymbol{w}^{(l)}(\cdot)$ is a
(learnable) spatial weighting function, $\odot$ denotes elementwise multiplication, 
and $f^{(l)}(\cdot, \cdot)$ is a (learnable) function. 

By default, the function $f^{(l)}(\cdot, \cdot)$ is modelled using a [`PowerDifference`](@ref) function. 
One may alternatively employ a nonlearnable function, for example, `f = (hᵢ, hⱼ) -> (hᵢ - hⱼ).^2`, 
specified through the keyword argument `f`.  

The spatial distances between locations must be stored as an edge feature, as facilitated by [`spatialgraph()`](@ref). 
The input to $\boldsymbol{w}(\cdot)$ is a $1 \times n$ matrix (i.e., a row vector) of spatial distances. 
The output of $\boldsymbol{w}(\cdot)$ must be either a scalar; a vector of the same dimension as the feature vectors of the previous layer; 
or, if the features vectors of the previous layer are scalars, a vector of arbitrary dimension. 
To promote identifiability, the weights are normalised to sum to one (row-wise) within each neighbourhood set. 
By default, $\boldsymbol{w}(\cdot)$ is taken to be a multilayer perceptron with a single hidden layer, 
although a custom choice for this function can be provided using the keyword argument `w`. 

# Arguments
- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `g = relu`: Activation function.
- `bias = true`: Add learnable bias?
- `init = glorot_uniform`: Initialiser for $\boldsymbol{\Gamma}_{\!1}^{(l)}$, $\boldsymbol{\Gamma}_{\!2}^{(l)}$, and $\boldsymbol{\gamma}^{(l)}$. 
- `f = nothing`
- `w = nothing` 
- `w_width = 128`: (Only applicable if `w = nothing`) The width of the hidden layer in the MLP used to model $\boldsymbol{w}(\cdot, \cdot)$. 
- `w_out = in`: (Only applicable if `w = nothing`) The output dimension of $\boldsymbol{w}(\cdot, \cdot)$.  

# Examples
```
using NeuralEstimators, Flux, GraphNeuralNetworks

# Toy spatial data
n = 250                # number of spatial locations
m = 5                  # number of independent replicates
S = rand(n, 2)         # spatial locations
Z = rand(n, m)         # data
g = spatialgraph(S, Z) # construct the graph

# Construct and apply spatial graph convolution layer
l = SpatialGraphConv(1 => 10)
l(g)
```
"""
struct SpatialGraphConv{W<:AbstractMatrix, A, B,C, F} <: GNNLayer
	Γ1::W
    Γ2::W
	b::B
    w::A
	f::C
	g::F
end
function SpatialGraphConv(
	ch::Pair{Int,Int},
	g = relu;
	init = glorot_uniform,
	bias::Bool = true,
	w = nothing,
	f = nothing,
	w_out::Union{Integer, Nothing} = nothing, 
	w_width::Integer = 128
	)

	in, out = ch

	# Spatial weighting function
	if isnothing(w) 
		# Options for w:
		# 1. Scalar output 
		# 2. Vector output with scalar input features, in which case the scalar features will be repeated to be of appropriate dimension 
		# 3. Vector output with vector input features, in which case the output dimension of w and the input dimension of the feature vectors must match 
		if isnothing(w_out)
			w_out = in
		else 
			@assert in == 1 || w_out == in "With vector-valued input features, the output of w must either be scalar or a vector of the same dimension as the input features"
		end 
		w = Chain(
				Dense(1 => w_width, g, init = init),
				Dense(w_width => w_out, g, init = init)
			)	
	else 
		@assert !isnothing(w_out) "Since you have specified the weight function w(), please also specify its output dimension `w_out`"
	end

	# Function of Z
	if isnothing(f)
		f = PowerDifference([0.5f0], [2.0f0])
	end

	# Weight matrices 
	Γ1 = init(out, in)
	Γ2 = init(out, w_out)

	# Bias vector
	b = bias ? Flux.create_bias(Γ1, true, out) : false

    SpatialGraphConv(Γ1, Γ2, b, w, f, g)
end
function (l::SpatialGraphConv)(g::GNNGraph, x::M) where M <: AbstractMatrix{T} where {T}
	l(g, reshape(x, size(x, 1), 1, size(x, 2)))
end
function (l::SpatialGraphConv)(g::GNNGraph, x::A) where A <: AbstractArray{T, 3} where {T}

    check_num_nodes(g, x)

	# Number of independent replicates
	m = size(x, 2)

	# Extract spatial information (typically the spatial distance between neighbours)
	s = :e ∈ keys(g.edata) ? g.edata.e : permutedims(g.graph[3]) 

	# Coerce to matrix
	if isa(s, AbstractVector)
		s = permutedims(s)
	end

	# Compute spatial weights and normalise over the neigbhourhoods 
	# Three options for w:
	# 1. Scalar output 
	# 2. Vector output with scalar input features, in which case the scalar features will be repeated to be of appropriate dimension 
	# 3. Vector output with vector input features, in which case the dimensionalities must match 
	w = l.w(s) 

	w̃ = normalise_edge_neighbors(g, w) # Sanity check: aggregate_neighbors(g, +, w̃) # zeros and ones 
	
	# Coerce to three-dimensional array, repeated to match the number of independent replicates
	w̃ = coerce3Darray(w̃, m)

	# Compute spatially-weighted sum of input features over each neighbourhood 
	#msg = apply_edges((l, xi, xj, w̃) -> w̃ .* l.f(xi, xj), g, l, x, x, w̃)
	msg = apply_edges((xi, xj, w̃) -> w̃ .* l.f(xi, xj), g, x, x, w̃)
	h̄ = aggregate_neighbors(g, +, msg) # sum over each neighbourhood individually 

	l.g.(l.Γ1 ⊠ x .+ l.Γ2 ⊠ h̄ .+ l.b) # ⊠ is shorthand for batched_mul  #NB any missingness will cause the feature vector to be entirely missing
end
function Base.show(io::IO, l::SpatialGraphConv)
    in_channel  = size(l.Γ1, ndims(l.Γ1))
    out_channel = size(l.Γ1, ndims(l.Γ1)-1)
    print(io, "SpatialGraphConv(", in_channel, " => ", out_channel)
    l.g == identity || print(io, ", ", l.g)
    print(io, ", w=", l.w)
    print(io, ")")
end

function coerce3Darray(x, m)
	if isa(x, AbstractVector)
		x = permutedims(x)
	end
	if isa(x, AbstractMatrix)
		x = reshape(x, size(x, 1), 1, size(x, 2))
	end
	x = repeat(x, 1, m, 1)  
end

"""
    normalise_edges(g, e)

Graph-wise normalisation of the edge features `e` to sum to one.
"""
function normalise_edges(g::GNNGraph, e)
    @assert size(e)[end] == g.num_edges
    gi = graph_indicator(g, edges = true)
    den = reduce_edges(+, g, e)
    den = gather(den, gi)
    return e ./ (den .+ eps(eltype(e)))
end

@doc raw"""
    normalise_edge_neighbors(g, e)

Normalise the edge features `e` to sum to one over each node's neighborhood, 

```math
\tilde{\mathbf{e}}_{j\to i} = \frac{\mathbf{e}_{j\to i}} {\sum_{j'\in N(i)} \mathbf{e}_{j'\to i}}.
```
"""
function normalise_edge_neighbors(g::AbstractGNNGraph, e)
	@assert size(e)[end] == g.num_edges
    s, t = edge_index(g)
    den = gather(scatter(+, e, t), t)
    return e ./ (den .+ eps(eltype(e)))
end

#TODO GNNSummary given a different name (maybe just GNN)? 
#TODO Does it make sense to show GraphConv here? Maybe better to show SpatialGraphConv? 
@doc raw"""
	GNNSummary(propagation, readout)
A graph neural network (GNN) module designed to serve as the inner network `ψ`
in the [`DeepSet`](@ref) representation when the data are graphical (e.g.,
irregularly observed spatial data).

The `propagation` module transforms graph data into a set of
hidden-feature graphs. The `readout` module aggregates these feature graphs into
a single hidden feature vector of fixed length. The network `ψ` is then defined as the composition of the
propagation and readout modules.

The data should be stored as a `GNNGraph` or `Vector{GNNGraph}`, where
each graph is associated with a single parameter vector. The graphs may contain
subgraphs corresponding to independent replicates.

# Examples
```
using NeuralEstimators, Flux, GraphNeuralNetworks
using Flux: batch
using Statistics: mean

# Propagation module
r  = 1     # dimension of response variable
nₕ = 32    # dimension of node feature vectors
propagation = GNNChain(GraphConv(r => nₕ), GraphConv(nₕ => nₕ))

# Readout module
readout = GlobalPool(mean)

# Inner network
ψ = GNNSummary(propagation, readout)

# Outer network
d = 3     # output dimension 
w = 64    # width of hidden layer
ϕ = Chain(Dense(nₕ, w, relu), Dense(w, d))

# DeepSet object 
ds = DeepSet(ψ, ϕ)

# Apply to data 
g₁ = rand_graph(11, 30, ndata = rand32(r, 11)) 
g₂ = rand_graph(13, 40, ndata = rand32(r, 13))
g₃ = batch([g₁, g₂])  
ds(g₁)                # single graph 
ds(g₃)                # graph with subgraphs corresponding to independent replicates
ds([g₁, g₂, g₃])      # vector of graphs, corresponding to multiple data sets 
```
"""
struct GNNSummary{F, G}
	propagation::F   # propagation module
	readout::G       # readout module
end
Base.show(io::IO, D::GNNSummary) = print(io, "\nThe propagation and readout modules of a graph neural network (GNN), with a total of $(nparams(D)) trainable parameters:\n\nPropagation module ($(nparams(D.propagation)) parameters):  $(D.propagation)\n\nReadout module ($(nparams(D.readout)) parameters):  $(D.readout)")

function (ψ::GNNSummary)(g::GNNGraph)

	# Propagation module
	h = ψ.propagation(g)
	Z = :Z ∈ keys(h.ndata) ? h.ndata.Z : first(values(h.ndata))

	# Readout module, computes a fixed-length vector (a summary statistic) for each replicate
	# R is a matrix with:
	# nrows = number of summary statistics
	# ncols = number of independent replicates
	R = ψ.readout(h, Z)

	# Reshape from three-dimensional array to matrix 
	R = reshape(R, size(R, 1), :) #NB not ideal to do this here, I think, makes the output of summarystatistics() quite confusing. (keep in mind the behaviour of summarystatistics on a vector of graphs and a single graph) 

	return R
end

# ---- Adjacency matrices ----

@doc raw"""
	adjacencymatrix(S::Matrix, k::Integer; maxmin = false, combined = false)
	adjacencymatrix(S::Matrix, r::AbstractFloat)
	adjacencymatrix(S::Matrix, r::AbstractFloat, k::Integer; random = true)
	adjacencymatrix(M::Matrix; k, r, kwargs...)

Computes a spatially weighted adjacency matrix from spatial locations `S` based 
on either the `k`-nearest neighbours of each location; all nodes within a disc of fixed radius `r`;
or, if both `r` and `k` are provided, a subset of `k` neighbours within a disc
of fixed radius `r`.

If `S` is a square matrix, it is treated as a distance matrix; otherwise, it
should be an $n$ x $d$ matrix, where $n$ is the number of spatial locations
and $d$ is the spatial dimension (typically $d$ = 2). In the latter case,
the distance metric is taken to be the Euclidean distance. Note that use of a 
maxmin ordering currently requires a matrix of spatial locations (not a distance matrix).

When using the `k` nearest neighbours, if `maxmin=false` (default) the neighbours are chosen based on all points in
the graph. If `maxmin=true`, a so-called maxmin ordering is applied,
whereby an initial point is selected, and each subsequent point is selected to
maximise the minimum distance to those points that have already been selected.
Then, the neighbours of each point are defined as the `k`-nearest neighbours
amongst the points that have already appeared in the ordering. If `combined=true`, the 
neighbours are defined to be the union of the `k`-nearest neighbours and the 
`k`-nearest neighbours subject to a maxmin ordering. 

Two subsampling strategies are implemented when choosing a subset of `k` neighbours within 
a disc of fixed radius `r`. If `random=true` (default), the neighbours are randomly selected from 
within the disc. If `random=false`, a deterministic algorithm is used 
that aims to preserve the distribution of distances within the neighbourhood set, by choosing 
those nodes with distances to the central node corresponding to the 
$\{0, \frac{1}{k}, \frac{2}{k}, \dots, \frac{k-1}{k}, 1\}$ quantiles of the empirical 
distribution function of distances within the disc (this in fact yields up to $k+1$ neighbours, 
since both the closest and furthest nodes are always included). 

By convention with the functionality in `GraphNeuralNetworks.jl` which is based on directed graphs, 
the neighbours of location `i` are stored in the column `A[:, i]` where `A` is the 
returned adjacency matrix. Therefore, the number of neighbours for each location is
given by `collect(mapslices(nnz, A; dims = 1))`, and the number of times each node is 
a neighbour of another node is given by `collect(mapslices(nnz, A; dims = 2))`.

By convention, we do not consider a location to neighbour itself (i.e., the diagonal elements of the adjacency matrix are zero). 

# Examples
```
using NeuralEstimators, Distances, SparseArrays

n = 250
d = 2
S = rand(Float32, n, d)
k = 10
r = 0.10

# Memory efficient constructors
adjacencymatrix(S, k)
adjacencymatrix(S, k; maxmin = true)
adjacencymatrix(S, k; maxmin = true, combined = true)
adjacencymatrix(S, r)
adjacencymatrix(S, r, k)
adjacencymatrix(S, r, k; random = false)

# Construct from full distance matrix D
D = pairwise(Euclidean(), S, dims = 1)
adjacencymatrix(D, k)
adjacencymatrix(D, r)
adjacencymatrix(D, r, k)
adjacencymatrix(D, r, k; random = false)
```
"""
function adjacencymatrix(M::Matrix; k::Union{Integer, Nothing} = nothing, r::Union{F, Nothing} = nothing, kwargs...) where F <: AbstractFloat
	# convenience keyword-argument function, used internally by spatialgraph()
	if isnothing(r) & isnothing(k)
		error("One of k or r must be set")
	elseif isnothing(r) 
		adjacencymatrix(M, k; kwargs...)
	elseif isnothing(k)
		adjacencymatrix(M, r)
	else
		adjacencymatrix(M, r, k; kwargs...)
	end
end

function adjacencymatrix(M::Mat, r::F, k::Integer; random::Bool = true) where Mat <: AbstractMatrix{T} where {T, F <: AbstractFloat}

	@assert k > 0
	@assert r > 0

	if !random
		A = adjacencymatrix(M, r) 
		A = subsetneighbours(A, k)
		A = dropzeros!(A) # remove self loops
		return A 
	end 

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
			if i != j # add self loops after construction, to ensure consistent number of neighbours
				if m == n # square matrix, so assume M is a distance matrix
					dᵢⱼ = M[i, j]
				else  # rectangular matrix, so assume S is a matrix of spatial locations
					sⱼ  = M[j, :]
					dᵢⱼ = norm(sᵢ - sⱼ)
				end
				if dᵢⱼ <= r
					push!(I, i)
					push!(J, j)
					push!(V, dᵢⱼ)
					kᵢ += 1
				end
			end
			if kᵢ == k 
				break 
			end
		end
	end
	A = sparse(J,I,V,n,n)
	A = dropzeros!(A) # remove self loops 
	return A
end
adjacencymatrix(M::Mat, k::Integer, r::F) where Mat <: AbstractMatrix{T} where {T, F <: AbstractFloat} = adjacencymatrix(M, r, k)

function adjacencymatrix(M::Mat, k::Integer; maxmin::Bool = false, moralise::Bool = false, combined::Bool = false) where Mat <: AbstractMatrix{T} where T

	@assert k > 0

	if combined 
		a1 = adjacencymatrix(M, k; maxmin = false, combined = false)
		a2 = adjacencymatrix(M, k; maxmin = true, combined = false) 
		A = a1 + (a1 .!= a2) .* a2 
		return A 
	end

	I = Int64[]
	J = Int64[]
	V = T[]
	n = size(M, 1)
	m = size(M, 2)

	if m == n # square matrix, so assume M is a distance matrix
		D = M
	else      # otherwise, M is a matrix of spatial locations
		S = M
		# S = S + 50 * eps(T) * rand(T, size(S, 1), size(S, 2)) # add some random noise to break ties
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
		A = sparse(J,I,V,n,n) # NB the neighbours of location i are stored in the column A[:, i]
	else
		@assert m != n "`adjacencymatrix` with maxmin-ordering requires a matrix of spatial locations, not a distance matrix"
		ord     = ordermaxmin(S)          # calculate ordering
		Sord    = S[ord, :]               # re-order locations
		NNarray = findorderednn(Sord, k)  # find k nearest neighbours/"parents"
		R = builddag(NNarray, T)          # build DAG
		A = moralise ?  R' * R : R        # moralise

		# Add distances to A
		# NB This is memory inefficient, especially for large n; only optimise if we find that this approach works well and this is a bottleneck
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
	A = dropzeros!(A) # remove self loops 
	return A
end

## helper functions
deletecol!(A,cind) = SparseArrays.fkeep!((i,j,v) -> j != cind, A)
findnearest(A::AbstractArray, x) = argmin(abs.(A .- x))
findnearest(V::SparseVector, q) = V.nzind[findnearest(V.nzval, q)] # efficient version for SparseVector that doesn't materialise a dense array
function subsetneighbours(A, k) 

	τ = [i/k for i ∈ 0:k] # probability levels (k+1 values)
	n = size(A, 1)

	# drop self loops 
	dropzeros!(A)
	for j ∈ 1:n 
		Aⱼ = A[:, j] # neighbours of node j 
		if nnz(Aⱼ) > k+1 # if there are fewer than k+1 neighbours already, we don't need to do anything 
			# compute the empirical τ-quantiles of the nonzero entries in Aⱼ
			quantiles = quantile(nonzeros(Aⱼ), τ) 
			# zero-out previous neighbours in Aⱼ
			deletecol!(A, j) 
			# find the entries in Aⱼ that are closest to the empirical quantiles 
			for q ∈ quantiles
				i = findnearest(Aⱼ, q)
				v = Aⱼ[i]
				A[i, j] = v
			end 
		end
	end
	A = dropzeros!(A) # remove self loops 
	return A
end


# Number of neighbours 

# # How it should be:
# s = [1,1,2,2,2,3,4,4,5,5]
# t = [2,3,1,4,5,3,2,5,2,4]
# v = [-5,-5,2,2,2,3,4,4,5,5]
# g = GNNGraph(s, t, v; ndata = (Z = ones(1, 5), )) #TODO shouldn't need to specify name Z
# A = adjacency_matrix(g)
# @test A == sparse(s, t, v)

# l = SpatialGraphConv(1 => 1, identity; aggr = +, bias = false) 
# l.w.β .= ones(Float32, 1)
# l.Γ1  .= zeros(Float32, 1)
# l.Γ2  .= ones(Float32, 1)
# node_features(l(g)) 

# # First node:
# i = 1
# ρ = exp.(l.w.β) # positive range parameter
# d = [A[2, i]]
# e = exp.(-d ./ ρ)
# sum(e)

# # Second node:
# i = 2
# ρ = exp.(l.w.β) # positive range parameter
# d = [A[1, i], A[4, i], A[5, i]]
# e = exp.(-d ./ ρ)
# sum(e)


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
		A = D .< r # bit-matrix specifying which locations are within a disc or r

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

	A = dropzeros!(A) # remove self loops

	return A
end

function findneighbours(d, k::Integer)
	V = partialsort(d, 1:k)
	J = [findall(v .== d) for v ∈ V]
	J = reduce(vcat, J)
	J = unique(J)
	J = J[1:k] # in the event of ties, there can be too many elements in J, so use only the first 1:k
    return J, V 
end

# TODO this function is much, much slower than the R version... need to optimise. Might be splat penalty; try reduce(hcat, .)
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
  V = T[1] 
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


# To remove dependence on Distributions, here we define a sampler from 
# the Poisson distribution, equivalent to rand(Poisson(λ))
function rpoisson(λ)
    k = 0                   # Start with k = 0
    p = exp(-λ)              # Initial probability value
    cumulative_prob = p      # Start the cumulative probability
    u = rand()               # Generate a uniform random number between 0 and 1
    
    # Keep adding terms to the cumulative probability until it exceeds u
    while u > cumulative_prob
        k += 1
        p *= λ / k           # Update the probability for the next value of k
        cumulative_prob += p  # Update the cumulative probability
    end
    
    return k
end

"""
	maternclusterprocess(; λ=10, μ=10, r=0.1, xmin=0, xmax=1, ymin=0, ymax=1, unit_bounding_box=false)

Simulates a Matérn cluster process with density of parent Poisson point process
`λ`, mean number of daughter points `μ`, and radius of cluster disk `r`, over the
simulation window defined by `xmin` and `xmax`, `ymin` and `ymax`.

If `unit_bounding_box` is `true`, then the simulated points will be scaled so that
the longest side of their bounding box is equal to one (this may change the simulation window). 

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
function maternclusterprocess(; λ = 10, μ = 10, r = 0.1, xmin = 0, xmax = 1, ymin = 0, ymax = 1, unit_bounding_box::Bool=false)

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
	# numbPointsParent=rand(Poisson(areaTotalExt*λ)) #Poisson number of points
	numbPointsParent=rpoisson(areaTotalExt*λ) #Poisson number of points

	#x and y coordinates of Poisson points for the parent
	xxParent=xminExt.+xDeltaExt*rand(numbPointsParent)
	yyParent=yminExt.+yDeltaExt*rand(numbPointsParent)

	#Simulate Poisson point process for the daughters (ie final poiint process)
	# numbPointsDaughter=rand(Poisson(μ),numbPointsParent)
	numbPointsDaughter=[rpoisson(μ) for _ in 1:numbPointsParent]
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

	S = hcat(xx, yy)

	unit_bounding_box ? unitboundingbox(S) : S
end

"""
#Examples 
```
n = 5
S = rand(n, 2)
unitboundingbox(S)
```
"""
function unitboundingbox(S::Matrix)
	Δs = maximum(S; dims = 1) -  minimum(S; dims = 1)
	r = maximum(Δs) 
	S/r # note that we would multiply range estimates by r
end
