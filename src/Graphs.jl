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
`Z` should be given as an $m$-vector of $n_i$-vectors,
and `S` should be given as an $m$-vector of $n_i \times d$ matrices.

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
function spatialgraph(S::AbstractMatrix; stationary = true, isotropic = true, store_S::Bool = false, pyramid_pool::Bool = false, kwargs...)

	# Determine neighbourhood based on keyword arguments 
	kwargs = (;kwargs...)
	k = haskey(kwargs, :k) ? kwargs.k : 30
	r = haskey(kwargs, :r) ? kwargs.r : 0.15
	random = haskey(kwargs, :random) ? kwargs.random : false

	if !isotropic #TODO (need to modify adjacencymatrix() to do this)
		error("Anistropy is not currently implemented (although it is documented in anticipation of future functionality); please contact the package maintainer")
	end
	if !stationary #TODO (need to modify adjacencymatrix() to do this)
		error("Nonstationarity is not currently implemented (although it is documented anticipation of future functionality); please contact the package maintainer")
	end
	ndata = DataStore()
	S = Float32.(S)
	A = adjacencymatrix(S; k = k, r = r, random = random) 
	S = permutedims(S) # need final dimension to be n-dimensional
	if store_S
		ndata = (ndata..., S = S)
	end
	# if pyramid_pool # NB not documenting pyramid_pool for now because it is experimental
	# 	clusterings = computeclusters(S)
	# 	ndata = (ndata..., clusterings = clusterings)
	# end
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
	permutedims(Float32.(Z), (1, 3, 2))
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

# ---- GraphConv ----

# 3D array version of GraphConv to allow the option to forego spatial information

"""
	(l::GraphConv)(g::GNNGraph, x::A) where A <: AbstractArray{T, 3} where {T}

Given a graph with node features a three dimensional array of size `in` × m × n, 
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


# ---- SpatialGraphConv ----

# import Flux: Bilinear
# # function (b::Bilinear)(Z::A) where A <: AbstractArray{T, 3} where T
# # 	@assert size(Z, 2) == 2
# # 	x = Z[:, 1, :]
# # 	y = Z[:, 2, :]
# # 	b(x, y)
# # end

# With a skip connection
# GNN = GNNChain(
# 	GraphSkipConnection(SpatialGraphConv(1 => 16)),
# 	SpatialGraphConv(16 + 1 => 32) # one extra input dimension corresponding to the input data
# )
# GNN(g)

#TODO
# stationary but anisotropic
# g = spatialgraph(S, Z; isotropic = false)
# layer = SpatialGraphConv(1 => 16; isotropic = false)
# layer(g)
#
# nonstationary
# g = spatialgraph(S, Z; stationary = false)
# layer = SpatialGraphConv(1 => 16; stationary = false)
# layer(g)

#TODO improve documentation for global features 
@doc raw"""
    SpatialGraphConv(in => out, g=relu; bias=true, init=glorot_uniform, ...)

Implements a spatial graph convolution, 

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
 \bar{\boldsymbol{h}}^{(l)}_{j} = \sum_{j' \in \mathcal{N}(j)}\boldsymbol{w}(\boldsymbol{s}_j, \boldsymbol{s}_{j'}; \boldsymbol{\beta}^{(l)}) \odot f(\boldsymbol{h}^{(l-1)}_{j}, \boldsymbol{h}^{(l-1)}_{j'}),
```

where $\boldsymbol{h}^{(l)}_{j}$ is the hidden feature vector at location
$\boldsymbol{s}_j$ at layer $l$, $g(\cdot)$ is a non-linear activation function
applied elementwise, $\boldsymbol{\Gamma}_{\!1}^{(l)}$ and
$\boldsymbol{\Gamma}_{\!2}^{(l)}$ are trainable parameter matrices,
$\boldsymbol{\gamma}^{(l)}$ is a trainable bias vector, $\mathcal{N}(j)$ denotes the
indices of neighbours of $\boldsymbol{s}_j$, $\boldsymbol{w}(\cdot, \cdot; \boldsymbol{\beta}^{(l)})$ is a
learnable weight function parameterised by $\boldsymbol{\beta}^{(l)}$, $\odot$
denotes elementwise multiplication, and $f(\cdot, \cdot)$ is a (possibly learnable) function. 

The function $f(\cdot, \cdot)$ is by default taken to be $f(\boldsymbol{h}^{(l-1)}_{j}, \boldsymbol{h}^{(l-1)}_{j'}) = |a \boldsymbol{h}^{(l-1)}_{j} - (1 - a) \boldsymbol{h}^{(l-1)}_{j'}|^b$ for learnable parameters $a ∈ [0, 1]$ and $b > 0$. One may alternatively employ a nonlearnable function, for example, `f = (hᵢ, hⱼ) -> (hᵢ - hⱼ).^2`. 

The spatial information must be stored as an edge feature, as facilitated by [`spatialgraph()`](@ref): 
- When modelling isotropic processes (`isotropic = true`), $\boldsymbol{w}(\cdot, \cdot)$ is made a function of spatial distance, so that $\boldsymbol{w}(\boldsymbol{s}_j, \boldsymbol{s}_{j'}) \equiv \boldsymbol{w}(\|\boldsymbol{s}_{j'} - \boldsymbol{s}_j\|)$. 
- When modelling stationary processes (`isotropic = false` and `stationary = true`), $\boldsymbol{w}(\cdot, \cdot)$ is made a function of spatial displacement, so that $\boldsymbol{w}(\boldsymbol{s}_j, \boldsymbol{s}_{j'}) \equiv \boldsymbol{w}(\boldsymbol{s}_{j'} - \boldsymbol{s}_j)$. 
- When modelling nonstationary processes (`stationary = false`), $\boldsymbol{w}(\cdot, \cdot)$ is made a function of the raw spatial locations. 
In all cases, the output of $\boldsymbol{w}(\cdot, \cdot)$ must be either a scalar; a vector of the same dimension as the feature vectors of the previous layer; or a vector of arbitrary dimension if the features vectors of the previous layer are scalars. 
To promote identifiability, the weights are normalised to sum to one (row-wise) within each neighbourhood set. 
By default, $\boldsymbol{w}(\cdot, \cdot)$ is taken to be a multilayer perceptron with a single hidden layer, 
although a custom choice for this function can be provided using the keyword argument `w`. 

# Arguments
- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `g = relu`: Activation function.
- `bias = true`: Add learnable bias?
- `init = glorot_uniform`: Initialiser for $\boldsymbol{\Gamma}_{\!1}^{(l)}$, $\boldsymbol{\Gamma}_{\!2}^{(l)}$, and $\boldsymbol{\gamma}^{(l)}$. 
- `isotropic = true`
- `stationary = true`
- `f = nothing`
- `w = nothing` 
- `w_width = 128`: (Only applicable if `w = nothing`) The width of the hidden layer in the MLP used to model $\boldsymbol{w}(\cdot, \cdot)$. 
- `w_out = in`: (Only applicable if `w = nothing`) The output dimension of $\boldsymbol{w}(\cdot, \cdot)$.  
- `glob = false`: If `true``, a global features will be computed directly from the entire spatial graph.


# Examples
```
using NeuralEstimators, Flux, GraphNeuralNetworks
using Statistics: mean

# Toy spatial data
m = 5                  # number of replicates
d = 2                  # spatial dimension
n = 250                # number of spatial locations
S = rand(n, d)         # spatial locations
Z = rand(n, m)         # data
g = spatialgraph(S, Z) # construct the graph

# Construct and apply spatial graph convolution layer
l = SpatialGraphConv(1 => 10, relu)
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
    glob::Bool 
end
@layer SpatialGraphConv
WeightedGraphConv = SpatialGraphConv; export WeightedGraphConv # alias for backwards compatability
function SpatialGraphConv(
	ch::Pair{Int,Int},
	g = sigmoid;
	init = glorot_uniform,
	bias::Bool = true,
	d::Integer = 2,
	isotropic::Bool = true,
	stationary::Bool = true,
	w = nothing,
	f = nothing,
	w_out::Union{Integer, Nothing} = nothing, 
	w_width::Integer = 128, 
	glob::Bool = false 
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
		if !stationary isotropic = false end
		if !isotropic #TODO (need to modify adjacencymatrix() to do this)
			error("Anistropy is not currently implemented (although it is documented in anticipation of future functionality); please contact the package maintainer")
		end
		if !stationary #TODO (need to modify adjacencymatrix() to do this)
			error("Nonstationarity is not currently implemented (although it is documented anticipation of future functionality); please contact the package maintainer")
		end
		w = if isotropic
			Chain(
				Dense(1 => w_width, g, init = init),
				Dense(w_width => w_out, g, init = init)
				)
		elseif stationary 
			Chain(
				Dense(d => w_width, g, init = init),
				Dense(w_width => w_out, g, init = init)
				)
		else
			Chain(
				Bilinear((d, d) => w_width, g, init = init),  
				Dense(w_width => w_out, g, init = init)
				)
		end 	
	else 
		@assert !isnothing(w_out) "Since you have specified the weight function w(), please also specify its output dimension `w_out`"
		# TODO need to add checks that w_out is consistent with the three allowed scenarios for w() described above 
	end

	# Function of Z
	if isnothing(f)
		# TODO f = appropriately constructed MLP (actually this is difficult since we have two 3D arrays as inputs...)
		f = PowerDifference([0.5f0], [2.0f0])
	end

	# Weight matrices 
	Γ1 = init(out, in)
	Γ2 = init(out, w_out)

	# Bias vector
	b = bias ? Flux.create_bias(Γ1, true, out) : false

    SpatialGraphConv(Γ1, Γ2, b, w, f, g, glob)
end
function (l::SpatialGraphConv)(g::GNNGraph)
	Z = :Z ∈ keys(g.ndata) ? g.ndata.Z : first(values(g.ndata)) 
	h = l(g, Z)
	if l.glob 
		@ignore_derivatives GNNGraph(g, gdata = (g.gdata..., R = h))
	else 
		@ignore_derivatives GNNGraph(g, ndata = (g.ndata..., Z = h))
	end
end
function (l::SpatialGraphConv)(g::GNNGraph, x::M) where M <: AbstractMatrix{T} where {T}
	l(g, reshape(x, size(x, 1), 1, size(x, 2)))
end
function (l::SpatialGraphConv)(g::GNNGraph, x::A) where A <: AbstractArray{T, 3} where {T}

    check_num_nodes(g, x)

	# Number of independent replicates
	m = size(x, 2)

	#TODO document that w() should act on 1xn matrix of distances 

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

	if l.glob 
		w̃ = normalise_edges(g, w) # Sanity check: sum(w̃; dims = 2)   # close to one 
	else 
		w̃ = normalise_edge_neighbors(g, w) # Sanity check: aggregate_neighbors(g, +, w̃) # zeros and ones 
	end 
	
	# Coerce to three-dimensional array and repeat to match the number of independent replicates (NB memory inefficient)
	if isa(w̃, AbstractVector)
		w̃ = permutedims(w̃)
	end
	if isa(w̃, AbstractMatrix)
		w̃ = reshape(w̃, size(w̃, 1), 1, size(w̃, 2))
	end
	w̃ = repeat(w̃, 1, m, 1)   

	# Compute spatially-weighted sum of input features over each neighbourhood 
	msg = apply_edges((l, xi, xj, w̃) -> w̃ .* l.f(xi, xj), g, l, x, x, w̃)
	if l.glob 
		h̄ = reduce_edges(+, g, msg) # sum over all neighbourhoods in the graph 
	else 
		h̄ = aggregate_neighbors(g, +, msg) # sum over each neighbourhood 
	end 
	
	if l.glob 
		return h̄
	else 
		return l.g.(l.Γ1 ⊠ x .+ l.Γ2 ⊠ h̄ .+ l.b) # ⊠ is shorthand for batched_mul
	end 
end
function Base.show(io::IO, l::SpatialGraphConv)
    in_channel  = size(l.Γ1, ndims(l.Γ1))
    out_channel = size(l.Γ1, ndims(l.Γ1)-1)
    print(io, "SpatialGraphConv(", in_channel, " => ", out_channel)
    l.g == identity || print(io, ", ", l.g)
    print(io, ", w=", l.w)
    print(io, ")")
end

#TODO export
#TODO parameters need to be stored as an array to be trainable... make a user-friendly constructor for this 
#TODO constrain a ∈ [0, 1] and b ∈ [0.5, 2]
"""

# Examples 
```
f = PowerDifference([0.5f0], [2.0f0])
x = rand32(3, 4)
y = rand32(3, 4)
f(x, y)

ps = Flux.params(f)

f = gpu(f)
x = gpu(x)
y = gpu(y)
f(x, y)
```
"""
struct PowerDifference{A,B}
	a::A
	b::B
end 
@layer PowerDifference
export PowerDifference
(f::PowerDifference)(x, y) = (abs.(f.a .* x - (1 .- f.a) .* y)).^f.b
(f::PowerDifference)(tup::Tuple) = f(tup[1], tup[2])


using GraphNeuralNetworks: scatter, gather
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
    softmax_edge_neighbors(g, e)

Normalise the edge features `e` to sum to one over each node's neighborhood, 

```math
\mathbf{e}'_{j\to i} = \frac{\mathbf{e}_{j\to i}}
                    {\sum_{j'\in N(i)} \mathbf{e}_{j'\to i}}.
```
"""
function normalise_edge_neighbors(g::AbstractGNNGraph, e)
    if g isa GNNHeteroGraph
        for (key, value) in g.num_edges
            @assert size(e)[end] == value
        end
    else
        @assert size(e)[end] == g.num_edges
    end
    s, t = edge_index(g)
    den = gather(scatter(+, e, t), t)
    return e ./ (den .+ eps(eltype(e)))
end

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
	#TODO might be faster to use apply_edges or another GNN function instead (and this might be neater in any case)
	# Bin the distances
	h_cutoffs = l.h_cutoffs
	bins_upper = h_cutoffs[2:end]   # upper bounds of the distance bins
	bins_lower = h_cutoffs[1:end-1] # lower bounds of the distance bins 
	N = [bins_lower[i:i] .< h .<= bins_upper[i:i] for i in eachindex(bins_upper)] # NB avoid scalar indexing by i:i
	N = reduce(vcat, N)
	Float32.(N)
end
@layer IndicatorWeights
Flux.trainable(l::IndicatorWeights) =  ()

#TODO document if I end up using this  
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

# ---- Universal pooling layer ----

@doc raw"""
    UniversalPool(ψ, ϕ)
Pooling layer (i.e., readout layer) from the paper ['Universal Readout for Graph Convolutional Neural Networks'](https://ieeexplore.ieee.org/document/8852103).
It takes the form,
```math
\boldsymbol{V} = ϕ(|G|⁻¹ \sum_{s\in G} ψ(\boldsymbol{h}_s)),
```
where ``\boldsymbol{V}`` denotes the summary vector for graph ``G``,
``\boldsymbol{h}_s`` denotes the vector of hidden features for node ``s \in G``,
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

# TODO explain the required dimenson of the feature matrix extracted by globalfeatures
@doc raw"""
	GNNSummary(propagation, readout; globalfeatures = nothing)

A graph neural network (GNN) module designed to serve as the summary network `ψ`
in the [`DeepSet`](@ref) representation when the data are graphical (e.g.,
irregularly observed spatial data).

The `propagation` module transforms graphical input data into a set of
hidden-feature graphs. The `readout` module aggregates these feature graphs into
a single hidden feature vector of fixed length (i.e., a vector of summary
statistics). The summary network is then defined as the composition of the
propagation and readout modules.

Optionally, one may also include a module that extracts features directly 
from the graph, through the keyword argument `globalfeatures`. This module, 
when applied to a `GNNGraph`, should return a matrix of features. 

The data should be stored as a `GNNGraph` or `Vector{GNNGraph}`, where
each graph is associated with a single parameter vector. The graphs may contain
subgraphs corresponding to independent replicates.

# Examples
```
using NeuralEstimators, Flux, GraphNeuralNetworks
using Flux: batch
using Statistics: mean

# Propagation module
d = 1      # dimension of response variable
nₕ = 32    # dimension of node feature vectors
propagation = GNNChain(GraphConv(d => nₕ), GraphConv(nₕ => nₕ))

# Readout module
readout = GlobalPool(mean)
nᵣ = nₕ   # dimension of readout vector

# Summary network
ψ = GNNSummary(propagation, readout)

# Inference network
p = 3     # number of parameters in the statistical model
w = 64    # width of hidden layer
ϕ = Chain(Dense(nᵣ, w, relu), Dense(w, p))

# Construct the estimator
θ̂ = DeepSet(ψ, ϕ)

# Apply the estimator to a single graph, a single graph with subgraphs
# (corresponding to independent replicates), and a vector of graphs
# (corresponding to multiple data sets each with independent replicates)
g₁ = rand_graph(11, 30, ndata=rand(d, 11))
g₂ = rand_graph(13, 40, ndata=rand(d, 13))
g₃ = batch([g₁, g₂])
θ̂(g₁)
θ̂(g₃)
θ̂([g₁, g₂, g₃])
```
"""
struct GNNSummary{F, G, H}
	propagation::F   # propagation module
	readout::G       # readout module
	globalfeatures::H
end
GNNSummary(propagation, readout; globalfeatures = nothing) = GNNSummary(propagation, readout, globalfeatures)
@layer GNNSummary
Base.show(io::IO, D::GNNSummary) = print(io, "\nThe propagation and readout modules of a graph neural network (GNN), with a total of $(nparams(D)) trainable parameters:\n\nPropagation module ($(nparams(D.propagation)) parameters):  $(D.propagation)\n\nReadout module ($(nparams(D.readout)) parameters):  $(D.readout)")

function (ψ::GNNSummary)(g::GNNGraph)

	# Propagation module
	h = ψ.propagation(g)

	# Readout module, computes a fixed-length vector (a summary statistic) for each replicate
	# R is a matrix with:
	# nrows = number of summary statistics
	# ncols = number of independent replicates
	# if isa(ψ.readout, SpatialPyramidPool)
	# 	R = ψ.readout(h)
	# else # Standard pooling layers
		Z = :Z ∈ keys(h.ndata) ? h.ndata.Z : first(values(h.ndata))
		R = ψ.readout(h, Z)
	# end

	if !isnothing(ψ.globalfeatures)
		R₂ = ψ.globalfeatures(g)
		if isa(R₂, GNNGraph)
			#TODO add assertion that there is data stored in gdata @assert  "The `globalfeatures` field of a `GNNSummary` object must return either an array or a graph with a non-empty field `gdata`"
			R₂ = first(values(R₂.gdata)) #TODO maybe want to enforce that it has an appropriate name (e.g., R)
		end
		R = vcat(R, R₂)
	end

	# reshape from three-dimensional array to matrix 
	R = reshape(R, size(R, 1), :) #TODO not ideal to do this here, I think, makes the output of summarystatistics() quite confusing. (keep in mind the behaviour of summarystatistics on a vector of graphs and a single graph) 

	return R
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

@doc raw"""
	adjacencymatrix(S::Matrix, k::Integer; maxmin = false, combined = false)
	adjacencymatrix(S::Matrix, r::AbstractFloat)
	adjacencymatrix(S::Matrix, r::AbstractFloat, k::Integer; random = true)
	adjacencymatrix(M::Matrix; k, r, kwargs...)

Computes a spatially weighted adjacency matrix from spatial locations `S` based 
on either the `k`-nearest neighbours of each location; all nodes within a disc of fixed radius `r`;
or, if both `r` and `k` are provided, a subset of `k` neighbours within a disc
of fixed radius `r`.

Several subsampling strategies are possible when choosing a subset of `k` neighbours within 
a disc of fixed radius `r`. If `random=true` (default), the neighbours are randomly selected from 
within the disc (note that this also approximately preserves the distribution of 
distances within the neighbourhood set). If `random=false`, a deterministic algorithm is used 
that aims to preserve the distribution of distances within the neighbourhood set, by choosing 
those nodes with distances to the central node corresponding to the 
$\{0, \frac{1}{k}, \frac{2}{k}, \dots, \frac{k-1}{k}, 1\}$ quantiles of the empirical 
distribution function of distances within the disc. 
(This algorithm in fact yields $k+1$ neighbours, since both the closest and furthest nodes are always included.) 
Otherwise, 

If `maxmin=false` (default) the `k`-nearest neighbours are chosen based on all points in
the graph. If `maxmin=true`, a so-called maxmin ordering is applied,
whereby an initial point is selected, and each subsequent point is selected to
maximise the minimum distance to those points that have already been selected.
Then, the neighbours of each point are defined as the `k`-nearest neighbours
amongst the points that have already appeared in the ordering. If `combined=true`, the 
neighbours are defined to be the union of the `k`-nearest neighbours and the 
`k`-nearest neighbours subject to a maxmin ordering. 

If `S` is a square matrix, it is treated as a distance matrix; otherwise, it
should be an $n$ x $d$ matrix, where $n$ is the number of spatial locations
and $d$ is the spatial dimension (typically $d$ = 2). In the latter case,
the distance metric is taken to be the Euclidean distance. Note that use of a 
maxmin ordering currently requires a matrix of spatial locations (not a distance matrix).

By convention with the functionality in `GraphNeuralNetworks.jl` which is based on directed graphs, 
the neighbours of location `i` are stored in the column `A[:, i]` where `A` is the 
returned adjacency matrix. Therefore, the number of neighbours for each location is
given by `collect(mapslices(nnz, A; dims = 1))`, and the number of times each node is 
a neighbour of another node is given by `collect(mapslices(nnz, A; dims = 2))`.

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

	if random == false
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
		# TODO This is memory inefficient, especially for large n; only optimise if we find that this approach works well and this is a bottleneck
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
deletecol!(A,cind) = SparseArrays.fkeep!(A,(i,j,v) -> j != cind)
findnearest(A::AbstractArray, x) = argmin(abs.(A .- x))
findnearest(V::SparseVector, q) = V.nzind[findnearest(V.nzval, q)] # efficient version for SparseVector that doesn't materialise a dense array
function selfloops!(A)
	#TODO export function, and replace previous documentation:
	# By convention, we consider a location to neighbour itself and, hence,
	# `k`-neighbour methods will yield `k`+1 neighbours for each location. Note that
	# one may use `dropzeros!()` to remove these self-loops from the constructed
	# adjacency matrix (see below).

	
	# add diagonal elements so that each node is considered its own neighbour
	T = eltype(A)
	for i ∈ 1:size(A, 1)
		A[i, i] = one(T)  # make element structurally nonzero
		A[i, i] = zero(T) # set to zero
	end
	return A
end
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
	A = dropzeros!(A) # remove self loops TODO Don't think is needed, since we already dropped them 
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
