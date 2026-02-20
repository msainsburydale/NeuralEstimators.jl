# For the method definitions, see the extension ext/NeuralEstimatorsGNNExt.jl

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
```
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

The data should be stored as a `GNNGraph` or `Vector{GNNGraph}`, where
each graph is associated with a single parameter vector. The graphs may contain
subgraphs corresponding to independent replicates.

# Examples
```
using NeuralEstimators, Flux, GraphNeuralNetworks
using Statistics: mean

# Spatial data
n = 100                # number of spatial locations
m = 50                 # number of independent replicates
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
```
using NeuralEstimators, GraphNeuralNetworks, Distances, LinearAlgebra
  
# Simulate Gaussian spatial data with exponential covariance function 
θ = 0.1                                 # true range parameter 
n = 250                                 # number of spatial locations 
S = rand(n, 2)                          # spatial locations 
D = pairwise(Euclidean(), S, dims = 1)  # distance matrix 
Σ = exp.(-D ./ θ)                       # covariance matrix 
L = cholesky(Symmetric(Σ)).L            # Cholesky factor 
m = 5                                   # number of independent replicates 
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
using NeuralEstimators, GraphNeuralNetworks

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
    h_cutoffs = range(0, stop = h_max, length = n_bins+1)
    h_cutoffs = collect(h_cutoffs)
    IndicatorWeights(h_cutoffs)
end
function (l::IndicatorWeights)(h::M) where {M <: AbstractMatrix{T}} where {T}
    h_cutoffs = l.h_cutoffs
    bins_upper = h_cutoffs[2:end]   # upper bounds of the distance bins
    bins_lower = h_cutoffs[1:(end - 1)] # lower bounds of the distance bins 
    N = [bins_lower[i:i] .< h .<= bins_upper[i:i] for i in eachindex(bins_upper)] # NB avoid scalar indexing by i:i
    N = reduce(vcat, N)
    f32(N)
end
Flux.trainable(l::IndicatorWeights) = NamedTuple()

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
using NeuralEstimators, GraphNeuralNetworks

h_max = 1
n_bins = 10
w = KernelWeights(h_max, n_bins)
h = rand(1, 30) # distances between 30 pairs of spatial locations 
w(h)
```
"""
struct KernelWeights{T1, T2}
    mu::T1
    sigma::T2
end
function KernelWeights(h_max, n_bins::Integer)
    h_cutoffs = range(0, stop = h_max, length = n_bins+1)
    h_cutoffs = collect(h_cutoffs)
    mu = [(h_cutoffs[i] + h_cutoffs[i + 1]) / 2 for i = 1:n_bins] # midpoints of the intervals 
    sigma = [(h_cutoffs[i + 1] - h_cutoffs[i]) / 4 for i = 1:n_bins] # std dev so that 95% of mass is within the bin 
    mu = f32(mu)
    sigma = f32(sigma)
    KernelWeights(mu, sigma)
end
function (l::KernelWeights)(h::M) where {M <: AbstractMatrix{T}} where {T}
    mu = l.mu
    sigma = l.sigma
    N = [exp.(-(h .- mu[i:i]) .^ 2 ./ (2 * sigma[i:i] .^ 2)) for i in eachindex(mu)] # Gaussian kernel for each bin (NB avoid scalar indexing by i:i)
    N = reduce(vcat, N)
    f32(N)
end
Flux.trainable(l::KernelWeights) = NamedTuple()


# ---- Adjacency matrices ----

@doc raw"""
	adjacencymatrix(S::Matrix, k::Integer)
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
the distance metric is taken to be the Euclidean distance.

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
function adjacencymatrix(M::Matrix; k::Union{Integer, Nothing} = nothing, r::Union{F, Nothing} = nothing, kwargs...) where {F <: AbstractFloat}
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

function adjacencymatrix(M::Mat, r::F, k::Integer; random::Bool = true) where {Mat <: AbstractMatrix{T}} where {T, F <: AbstractFloat}
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
                    sⱼ = M[j, :]
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
    A = sparse(J, I, V, n, n)
    A = dropzeros!(A) # remove self loops 
    return A
end
adjacencymatrix(M::Mat, k::Integer, r::F) where {Mat <: AbstractMatrix{T}} where {T, F <: AbstractFloat} = adjacencymatrix(M, r, k)

function adjacencymatrix(M::Mat, k::Integer) where {Mat <: AbstractMatrix{T}} where {T}
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
    else
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
        A = sparse(J, I, V, n, n) # NB the neighbours of location i are stored in the column A[:, i]
    end
    A = dropzeros!(A) # remove self loops 
    return A
end

## helper functions
deletecol!(A, cind) = SparseArrays.fkeep!((i, j, v) -> j != cind, A)
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

function adjacencymatrix(M::Mat, r::F) where {Mat <: AbstractMatrix{T}} where {T, F <: AbstractFloat}
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
        A = sparse(I, J, V, n, n)
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
