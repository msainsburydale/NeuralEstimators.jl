# See also the extension ext/NeuralEstimatorsGNNExt.jl

@doc raw"""
	spatialgraph(S)
	spatialgraph(S, Z)
	spatialgraph(g::GNNGraph, Z)
Given spatial data `Z` measured at spatial locations `S`, constructs a
[`GNNGraph`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/stable/api/gnngraph/#GNNGraph-type)
ready for use in a graph neural network that employs [`SpatialGraphConv`](@ref) layers. 

When $m$ independent replicates are collected over the same set of $n$ spatial locations,
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
```julia
using NeuralEstimators, GraphNeuralNetworks

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
function spatialgraph end

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
The input to $\boldsymbol{w}^{(l)}(\cdot)$ is a $1 \times n$ matrix (i.e., a row vector) of spatial distances. 
The output of $\boldsymbol{w}^{(l)}(\cdot)$ must be either a scalar; a vector of the same dimension as the feature vectors of the previous layer; 
or, if the features vectors of the previous layer are scalars, a vector of arbitrary dimension. 
To promote identifiability, the weights are normalised to sum to one (row-wise) within each neighbourhood set. 
By default, $\boldsymbol{w}^{(l)}(\cdot)$ is taken to be a multilayer perceptron with a single hidden layer,
although a custom choice for this function can be provided using the keyword argument `w`.
The hidden layer of the default $\boldsymbol{w}^{(l)}(\cdot)$ uses the activation `g`, while its output layer
uses `softplus`, so that the weights are strictly positive. A custom `w` should likewise return non-negative
weights; note that an output activation which can return exactly zero for every edge (e.g., `relu`) risks
$\bar{\boldsymbol{h}}^{(l)}_{j}$ being identically zero with no gradient available to recover from it.

!!! note "GPU memory and the choice of batch size"
    The messages $f^{(l)}(\cdot, \cdot)$ are formed on the *edges* of the batched graph, so
    every intermediate array in the layer is of size `(w_out, m, E)`, where `m` is the number
    of independent replicates and `E` is the total number of edges in the batch, that is, the
    batch size multiplied by the number of edges per data set. Peak memory during training
    therefore grows in proportion to `m` × (batch size), and it is the peak that matters:
    the total volume of memory allocated over an epoch is independent of the batch size.

    The practical consequence is that, with many replicates, *increasing* the batch size can
    make training slower rather than faster, because memory pressure causes the CUDA.jl
    allocator to run the garbage collector inside the allocation path. The transition is
    abrupt rather than gradual, and once it is crossed the run time is dominated by garbage
    collection rather than by computation. If throughput degrades as the batch size is
    raised, reduce the batch size; if a large effective batch is needed for optimisation
    reasons, accumulate gradients over several smaller sub-batches instead.

# Arguments
- `in`: dimension of input features.
- `out`: dimension of output features.
- `g = relu`: activation function.
- `bias = true`: add learnable bias?
- `init = glorot_uniform`: initialiser for $\boldsymbol{\Gamma}_{\!1}^{(l)}$, $\boldsymbol{\Gamma}_{\!2}^{(l)}$, and $\boldsymbol{\gamma}^{(l)}$. 
- `f = nothing`
- `w = nothing` 
- `w_width = 128` (applicable only if `w = nothing`): the width of the hidden layer in the MLP used to model $\boldsymbol{w}^{(l)}(\cdot, \cdot)$. 
- `w_out = in` (applicable only if `w = nothing`): the output dimension of $\boldsymbol{w}^{(l)}(\cdot, \cdot)$.

# Examples
```julia
using NeuralEstimators, Flux, GraphNeuralNetworks

# Toy spatial data
n = 250                # number of spatial locations
m = 5                  # number of replicates
S = rand(n, 2)         # spatial locations
Z = rand(n, m)         # data
g = spatialgraph(S, Z) # construct the graph

# Construct and apply spatial graph convolution layer
l = SpatialGraphConv(1 => 10)
l(g)
```
"""
struct SpatialGraphConv{W <: AbstractMatrix, A, B, C, F} # <: GNNLayer
    Γ1::W
    Γ2::W
    b::B
    w::A
    f::C
    g::F
end
SpatialGraphConv(args...; kwargs...) = error("SpatialGraphConv requires GraphNeuralNetworks.jl to be loaded, i.e., `using GraphNeuralNetworks`")

#TODO Rename this "GNN"? 
@doc raw"""
	GNNSummary(propagation, readout)
A graph neural network (GNN) module designed to serve as the inner network `ψ`
in the [`DeepSet`](@ref) representation when the data are graphical (e.g.,
irregularly observed spatial data).

The `propagation` module transforms graph data into a set of
hidden-feature graphs. The `readout` module aggregates these feature graphs into
a single hidden feature vector of fixed length. The network `ψ` is then defined as the composition of the
propagation and readout modules.

The data should be stored as a `Vector{A}` where each element is associated with
one parameter vector. For spatial data collected over irregular locations, `A` is
typically a [`GNNGraph`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/api/gnngraph/#GraphNeuralNetworks.GNNGraphs.GNNGraph),
where independent replicates (possibly with differing spatial locations) are
stored as subgraphs. See [`spatialgraph`](@ref) for constructing these graphs
from matrices of spatial locations and data.

# Examples
```julia
using NeuralEstimators, Flux, GraphNeuralNetworks
using Statistics: mean

# Spatial data
n = 100                # number of spatial locations
m = 50                 # number of replicates
S = rand(n, 2)         # spatial locations
Z = rand(n, m)         # observed data
g = spatialgraph(S, Z) # construct the graph

# Propagation module
nₕ = 32    # dimension of node feature vectors
propagation = Chain(SpatialGraphConv(1 => nₕ), SpatialGraphConv(nₕ => nₕ))

# Readout module
readout = GlobalPool(mean)

# Inner network
ψ = GNNSummary(propagation, readout)

# Outer network
d = 3     # number of parameters
w = 64    # width of hidden layer
ϕ = Chain(Dense(nₕ, w, relu), Dense(w, d))

# DeepSet object 
ds = DeepSet(ψ, ϕ)

# Apply to data 
ds(g)        # single graph with subgraphs corresponding to independent replicates
ds([g, g])   # vector of graphs, corresponding to multiple data sets 
```
"""
struct GNNSummary{F, G}
    propagation::F   # propagation module
    readout::G       # readout module
end
Base.show(io::IO, D::GNNSummary) = print(io, "\nThe propagation and readout modules of a graph neural network (GNN), with a total of $(nparams(D)) trainable parameters:\n\nPropagation module ($(nparams(D.propagation)) parameters):  $(D.propagation)\n\nReadout module ($(nparams(D.readout)) parameters):  $(D.readout)")
GNNSummary(args...; kwargs...) = error("GNNSummary requires GraphNeuralNetworks.jl to be loaded, i.e., `using GraphNeuralNetworks`")
nparams(model) = length(Optimisers.trainables(model)) > 0 ? sum(length, Optimisers.trainables(model)) : 0

#TODO clean up this documentation (e.g., don't bother with the bin notation)
#TODO there is a more general structure that we could define, that has message(xi, xj, e) as a slot
@doc raw"""
	NeighbourhoodVariogram(h_max, n_bins) 
	(l::NeighbourhoodVariogram)(g::GNNGraph)

Computes the empirical variogram, 

```math
\hat{\gamma}(h \pm \delta) = \frac{1}{2|N(h \pm \delta)|} \sum_{(i,j) \in N(h \pm \delta)} (Z_i - Z_j)^2
```

where $N(h \pm \delta) \equiv \left\{(i,j) : \|\boldsymbol{s}_i - \boldsymbol{s}_j\| \in (h-\delta, h+\delta)\right\}$ 
is the set of pairs of locations separated by a distance within $(h-\delta, h+\delta)$, and $|\cdot|$ denotes set cardinality. 

The distance bins are constructed to have constant width $2\delta$, chosen based on the maximum distance 
`h_max` to be considered, and the specified number of bins `n_bins`. 

The input type is a `GNNGraph`, and the empirical variogram is computed based on the corresponding graph structure. 
Specifically, only locations that are considered neighbours will be used when computing the empirical variogram. 

# Examples 
```julia
using NeuralEstimators, GraphNeuralNetworks, Distances, LinearAlgebra
  
# Simulate Gaussian spatial data with exponential covariance function 
θ = 0.1                                 # true range parameter 
n = 250                                 # number of spatial locations 
S = rand(n, 2)                          # spatial locations 
D = pairwise(Euclidean(), S, dims = 1)  # distance matrix 
Σ = exp.(-D ./ θ)                       # covariance matrix 
L = cholesky(Symmetric(Σ)).L            # Cholesky factor 
m = 5                                   # number of replicates 
Z = L * randn(n, m)                     # simulated data 

# Construct the spatial graph 
r = 0.15                                # radius of neighbourhood set
g = spatialgraph(S, Z, r = r)

# Construct the variogram object with 10 bins
nv = NeighbourhoodVariogram(r, 10) 

# Compute the empirical variogram 
nv(g)
```
"""
struct NeighbourhoodVariogram{T} # <: GNNLayer
    h_cutoffs::T
    # TODO inner constructor, add 0 into h_cutoffs if it is not already in there 
end
NeighbourhoodVariogram(args...; kwargs...) = error("NeighbourhoodVariogram requires GraphNeuralNetworks.jl to be loaded, i.e., `using GraphNeuralNetworks`")

# ---- Adjacency matrices ----

@doc raw"""
	adjacencymatrix(S::Matrix, k::Integer; metric = Euclidean())
	adjacencymatrix(S::Matrix, r::AbstractFloat; metric = Euclidean())
	adjacencymatrix(S::Matrix, r::AbstractFloat, k::Integer; random = true, metric = Euclidean())
	adjacencymatrix(S::Matrix; k, r, kwargs...)

Computes a spatially weighted adjacency matrix from spatial locations `S` based
on either the `k`-nearest neighbours of each location; all nodes within a disc of fixed radius `r`;
or, if both `r` and `k` are provided, a subset of `k` neighbours within a disc
of fixed radius `r`.

`S` should be an $n$ x $d$ matrix, where $n$ is the number of spatial locations
and $d$ is the spatial dimension (typically $d$ = 2).

The distance metric defaults to the Euclidean distance, and may be changed using the keyword
argument `metric`, which accepts any metric from
[Distances.jl](https://github.com/JuliaStats/Distances.jl). For instance, for locations given
as longitude–latitude pairs, `metric = Haversine(6371.0)` gives great-circle distances in
kilometres, in which case `r` is also interpreted in kilometres. The neighbour search is
performed using a spatial index from
[NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl), selected
automatically to suit the given metric.

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
Distinct locations that happen to coincide are, however, treated as neighbours of one another,
and are stored with an edge weight of zero.

!!! note "Precomputed dissimilarity matrices"
    Earlier versions accepted a square matrix, which was interpreted as a precomputed distance
    matrix. This is no longer supported: it was slower and required $O(n^2)$ memory, and the
    `metric` keyword argument covers the same ground (for example, great-circle distances)
    while remaining $O(n \log n)$.

# Examples
```julia
using NeuralEstimators, Distances, SparseArrays

n = 250
d = 2
S = rand(Float32, n, d)
k = 10
r = 0.10

adjacencymatrix(S, k)
adjacencymatrix(S, r)
adjacencymatrix(S, r, k)
adjacencymatrix(S, r, k; random = false)

# Great-circle distance (in km) for longitude–latitude data
S = hcat(360 * rand(n) .- 180, 180 * rand(n) .- 90)
adjacencymatrix(S, 10; metric = Haversine(6371.0))
adjacencymatrix(S, 500.0; metric = Haversine(6371.0))
```
"""
function adjacencymatrix(S::Matrix; k::Union{Integer, Nothing} = nothing, r::Union{F, Nothing} = nothing, metric = Euclidean(), kwargs...) where {F <: AbstractFloat}
    # convenience keyword-argument function, used internally by spatialgraph()
    if isnothing(r) & isnothing(k)
        error("One of k or r must be set")
    elseif isnothing(r)
        adjacencymatrix(S, k; metric = metric)
    elseif isnothing(k)
        adjacencymatrix(S, r; metric = metric)
    else
        adjacencymatrix(S, r, k; metric = metric, kwargs...)
    end
end

# The type of spatial index is chosen by dispatch on the metric. This is a correctness
# requirement rather than a performance preference: a ball tree relies on the triangle
# inequality, so it returns wrong neighbours for a dissimilarity that violates it, and a
# KD-tree additionally requires a Minkowski metric. Anything weaker falls back to an
# exhaustive search, which is valid for any pre-metric
_spatialindex(Sᵀ, metric::MinkowskiMetric) = KDTree(Sᵀ, metric)
_spatialindex(Sᵀ, metric::Metric) = BallTree(Sᵀ, metric)
_spatialindex(Sᵀ, metric::PreMetric) = BruteTree(Sᵀ, metric)

# Guards against the most likely error when migrating from a version that accepted a
# precomputed distance matrix: silently treating one as n locations in n dimensions would give
# meaningless neighbourhoods rather than an error
function _checklocations(S::AbstractMatrix)
    n, d = size(S)
    if n == d && n > 2 && all(iszero, diag(S)) && issymmetric(S)
        throw(ArgumentError(
            "S appears to be a distance matrix (square, symmetric, with a zero diagonal), but " *
            "adjacencymatrix() expects an n × d matrix of spatial locations. Pass the locations " *
            "instead, together with a metric if the distances are not Euclidean (e.g. " *
            "metric = Haversine(6371.0)). If you only have a precomputed dissimilarity matrix, " *
            "construct the adjacency matrix yourself and pass it to spatialgraph()."
        ))
    end
    return nothing
end

function adjacencymatrix(S::Mat, r::F, k::Integer; random::Bool = true, metric = Euclidean()) where {Mat <: AbstractMatrix{T}} where {T, F <: AbstractFloat}
    @assert k > 0
    @assert r > 0
    _checklocations(S)

    if !random
        A = adjacencymatrix(S, r; metric = metric)
        return subsetneighbours(A, k)
    end

    n = size(S, 1)
    Sᵀ = permutedims(S)
    tree = _spatialindex(Sᵀ, metric)
    candidates = inrange(tree, Sᵀ, r)

    I = Int64[]
    J = Int64[]
    V = T[]
    nbrs = Int64[]
    dists = T[]
    for i ∈ 1:n
        empty!(nbrs)
        empty!(dists)
        sᵢ = view(Sᵀ, :, i)
        for j ∈ sort(candidates[i]) # NB inrange() returns the indices in no particular order
            j == i && continue # we do not consider a location to neighbour itself
            dᵢⱼ = T(metric(sᵢ, view(Sᵀ, :, j)))
            if dᵢⱼ <= r
                push!(nbrs, j)
                push!(dists, dᵢⱼ)
            end
        end
        # Uniform random subset of size min(k, mᵢ), by partial Fisher–Yates. NB this is
        # equivalent in distribution to scanning all locations in a random order and stopping
        # at k (the previous formulation), but costs O(mᵢ) rather than O(n) per location
        mᵢ = length(nbrs)
        for t ∈ 1:min(k, mᵢ)
            s = rand(t:mᵢ)
            nbrs[t], nbrs[s] = nbrs[s], nbrs[t]
            dists[t], dists[s] = dists[s], dists[t]
            push!(I, i)
            push!(J, nbrs[t])
            push!(V, dists[t])
        end
    end
    return sparse(J, I, V, n, n) # NB the neighbours of location i are stored in the column A[:, i]
end
adjacencymatrix(S::Mat, k::Integer, r::F; kwargs...) where {Mat <: AbstractMatrix{T}} where {T, F <: AbstractFloat} = adjacencymatrix(S, r, k; kwargs...)

function adjacencymatrix(S::Mat, k::Integer; metric = Euclidean()) where {Mat <: AbstractMatrix{T}} where {T}
    @assert k > 0
    _checklocations(S)

    n = size(S, 1)
    Sᵀ = permutedims(S)
    tree = _spatialindex(Sᵀ, metric)

    # Request one extra neighbour, since a location is always among its own nearest
    # neighbours, and no more than there are locations, since knn() errors if asked for more
    idx, dist = knn(tree, Sᵀ, min(k + 1, n), true)

    kₙ = min(k, n - 1) # the number of neighbours actually attainable
    I = Vector{Int64}(undef, 0)
    J = Vector{Int64}(undef, 0)
    V = Vector{T}(undef, 0)
    sizehint!(I, n * kₙ); sizehint!(J, n * kₙ); sizehint!(V, n * kₙ)
    for i ∈ 1:n
        idxᵢ = idx[i]
        distᵢ = dist[i]
        kᵢ = 0
        for t ∈ eachindex(idxᵢ)
            # NB the location itself is discarded by index rather than by discarding a zero
            # distance: locations that coincide are also at distance zero from one another, and
            # they are legitimate neighbours. For the same reason the location itself is not
            # necessarily listed first, so every candidate must be checked
            idxᵢ[t] == i && continue
            kᵢ == kₙ && break
            push!(I, i)
            push!(J, idxᵢ[t])
            push!(V, T(distᵢ[t]))
            kᵢ += 1
        end
    end
    return sparse(J, I, V, n, n) # NB the neighbours of location i are stored in the column A[:, i]
end

## helper functions
# Reduces each neighbourhood to at most k+1 members, chosen so that their distances to the
# central location fall closest to the {0, 1/k, …, 1} quantiles of the distances within that
# neighbourhood, thereby approximately preserving the distribution of distances.
# NB builds a new matrix rather than modifying A in place: deleting a column of a
# SparseMatrixCSC scans the entire matrix and inserting a single entry shifts its internal
# arrays, which together made the in-place formulation O(n × nnz)
function subsetneighbours(A::SparseMatrixCSC{T}, k::Integer) where {T}
    τ = [i/k for i ∈ 0:k] # probability levels (k+1 values)
    n = size(A, 1)
    rows = rowvals(A)
    vals = nonzeros(A)

    I = Int64[]
    J = Int64[]
    V = T[]
    selected = Int64[]
    for j ∈ 1:n
        nzⱼ = nzrange(A, j) # the stored entries of column j, i.e. the neighbours of node j
        if length(nzⱼ) <= k+1
            # if there are fewer than k+1 neighbours already, we don't need to do anything
            for p ∈ nzⱼ
                push!(I, rows[p]); push!(J, j); push!(V, vals[p])
            end
        else
            # compute the empirical τ-quantiles of the distances within the neighbourhood
            quantiles = quantile(view(vals, nzⱼ), τ)
            empty!(selected)
            for q ∈ quantiles
                # Find the entry closest to the empirical quantile, scanning in storage order
                # so that ties are broken towards the first such entry.
                # NB repeats are collapsed: two quantiles may select the same neighbour, and
                # sparse() sums duplicated indices rather than retaining one of them
                p★ = first(nzⱼ)
                δ★ = abs(vals[p★] - q)
                for p ∈ nzⱼ
                    δ = abs(vals[p] - q)
                    if δ < δ★
                        δ★ = δ
                        p★ = p
                    end
                end
                p★ ∈ selected || push!(selected, p★)
            end
            for p ∈ selected
                push!(I, rows[p]); push!(J, j); push!(V, vals[p])
            end
        end
    end
    return sparse(I, J, V, n, n) # NB preserves the orientation of the input
end

function adjacencymatrix(S::Mat, r::F; metric = Euclidean()) where {Mat <: AbstractMatrix{T}} where {T, F <: AbstractFloat}
    @assert r > 0
    _checklocations(S)

    n = size(S, 1)
    Sᵀ = permutedims(S)
    tree = _spatialindex(Sᵀ, metric)
    candidates = inrange(tree, Sᵀ, r)

    I = Int64[]
    J = Int64[]
    V = T[]
    for i ∈ 1:n
        sᵢ = view(Sᵀ, :, i)
        for j ∈ sort(candidates[i]) # NB inrange() returns the indices in no particular order
            j == i && continue # we do not consider a location to neighbour itself
            dᵢⱼ = T(metric(sᵢ, view(Sᵀ, :, j)))
            # NB inrange() is inclusive of r, whereas this method has always used a strict
            # inequality, so the candidates are filtered rather than taken as they are
            if dᵢⱼ < r
                push!(I, j)
                push!(J, i)
                push!(V, dᵢⱼ)
            end
        end
    end
    return sparse(I, J, V, n, n) # NB the neighbours of location i are stored in the column A[:, i]
end

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

# ---- Cluster processes ----

"""
	maternclusterprocess(; λ=10, μ=10, r=0.1, xmin=0, xmax=1, ymin=0, ymax=1, unit_bounding_box=false)
Generates a realisation from a Matérn cluster process (e.g., [Baddeley et al., 2015](https://www.taylorfrancis.com/books/mono/10.1201/b19708/spatial-point-patterns-adrian-baddeley-rolf-turner-ege-rubak), Ch. 12). 

The process is defined by a parent homogenous Poisson point process with intensity `λ` > 0, a mean number of daughter points `μ` > 0, and a cluster radius `r` > 0. The simulation is performed over a rectangular window defined by [`xmin, xmax`] × [`ymin`, `ymax`].

If `unit_bounding_box = true`, the simulated points will be scaled so that
the longest side of their bounding box is equal to one (this may change the simulation window). 

See also the R package
[`spatstat`](https://cran.r-project.org/web/packages/spatstat/index.html),
which provides functions for simulating from a range of point processes and
which can be interfaced from Julia using
[`RCall`](https://juliainterop.github.io/RCall.jl/stable/).

# Examples
```julia
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
function maternclusterprocess(; λ = 10, μ = 10, r = 0.1, xmin = 0, xmax = 1, ymin = 0, ymax = 1, unit_bounding_box::Bool = false)

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
    xxParent=xminExt .+ xDeltaExt*rand(numbPointsParent)
    yyParent=yminExt .+ yDeltaExt*rand(numbPointsParent)

    #Simulate Poisson point process for the daughters (ie final poiint process)
    # numbPointsDaughter=rand(Poisson(μ),numbPointsParent)
    numbPointsDaughter=[rpoisson(μ) for _ = 1:numbPointsParent]
    numbPoints=sum(numbPointsDaughter) #total number of points

    #Generate the (relative) locations in polar coordinates by
    #simulating independent variables.
    theta=2*pi*rand(numbPoints) #angular coordinates
    rho=r*sqrt.(rand(numbPoints)) #radial coordinates

    #Convert polar to Cartesian coordinates
    xx0=rho .* cos.(theta)
    yy0=rho .* sin.(theta)

    #replicate parent points (ie centres of disks/clusters)
    xx=vcat(fill.(xxParent, numbPointsDaughter)...)
    yy=vcat(fill.(yyParent, numbPointsDaughter)...)

    #Shift centre of disk to (xx0,yy0)
    xx=xx .+ xx0
    yy=yy .+ yy0

    #thin points if outside the simulation window
    booleInside=((xx .>= xmin) .& (xx .<= xmax) .& (yy .>= ymin) .& (yy .<= ymax))
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
    Δs = maximum(S; dims = 1) - minimum(S; dims = 1)
    r = maximum(Δs)
    S/r # note that we would multiply range estimates by r
end
