module NeuralEstimatorsGNNExt

using NeuralEstimators
using Flux
using Flux: @ignore_derivatives, glorot_uniform
using GraphNeuralNetworks
using GraphNeuralNetworks: check_num_nodes
using NNlib: scatter, gather
using Statistics, Random, LinearAlgebra
import NeuralEstimators: subsetdata, numberreplicates, summarystatistics, spatialgraph
import NeuralEstimators: GNNSummary, SpatialGraphConv, IndicatorWeights, KernelWeights, NeighbourhoodVariogram
import NeuralEstimators: _first_N_minus_1_dims_identical

function subsetdata(Z::G, i) where {G <: GNNGraph}
    if typeof(i) <: Integer
        i = i:i
    end
    sym = collect(keys(Z.ndata))[1]
    if ndims(Z.ndata[sym]) == 3
        GNNGraph(Z; ndata = Z.ndata[sym][:, i, :])
    else
        # @warn "`subsetdata()` is slow for graphical data."
        # TODO Recall that I set the code up to have ndata as a 3D array; with this format, non-parametric bootstrap would be exceedingly fast (since we can subset the array data, I think).
        # TODO getgraph() doesn't currently work with the GPU: see https://github.com/CarloLucibello/GraphNeuralNetworks.jl/issues/161
        # TODO getgraph() doesn’t return duplicates. So subsetdata(Z, [1, 1]) returns just a single graph
        # TODO can't check for CuArray (and return to GPU) because CuArray won't always be defined (no longer depend on CUDA) and we can't overload exact signatures in package extensions... it's low priority, but will be good to fix when time permits. Hopefully, the above issue with GraphNeuralNetworks.jl will get fixed, and we can then just remove the call to cpu() below
        #flag = Z.ndata[sym] isa CuArray
        Z = cpu(Z)
        Z = getgraph(Z, i)
        #if flag Z = gpu(Z) end
        Z
    end
end

function numberreplicates(Z::G) where {G <: GNNGraph}
    x = :Z ∈ keys(Z.ndata) ? Z.ndata.Z : first(values(Z.ndata))
    if ndims(x) == 3
        size(x, 2)
    else
        Z.num_graphs
    end
end

# Multiple data sets: optimised version for graph data
function summarystatistics(d::DeepSet, Z::V) where {V <: AbstractVector{G}} where {G <: GNNGraph}
    @assert isnothing(d.ψ) || typeof(d.ψ) <: GNNSummary "For graph input data, the summary network ψ should be a `GNNSummary` object"

    if !isnothing(d.ψ)
        if @ignore_derivatives _first_N_minus_1_dims_identical(Z)

            # For efficiency, convert Z from a vector of (super)graphs into a single
            # supergraph before applying the neural network. Since each element of Z
            # may itself be a supergraph (where each subgraph corresponds to an
            # independent replicate), record the grouping of independent replicates
            # so that they can be combined again later in the function
            m = numberreplicates.(Z)

            # Propagation and readout
            g = @ignore_derivatives Flux.batch(Z) # NB batch() causes array mutation, so do not attempt to compute derivatives through this call
            R = d.ψ(g)

            # Split R based on the original vector of data sets Z
            if ndims(R) == 2

                # R is a matrix, with column dimension M = sum(m), and we split R
                # based on the original grouping specified by m
                # NB since this only works for identical m, there is some code redundancy here I believe
                ng = length(m)
                cs = cumsum(m)
                indices = [(cs[i] - m[i] + 1):cs[i] for i ∈ 1:ng]
                R̃ = [R[:, idx] for idx ∈ indices]
            elseif ndims(R) == 3
                R̃ = [R[:, :, i] for i ∈ 1:size(R, 3)]
            end
        else
            # Array sizes differ, so therefore cannot stack together; use simple (and slower) broadcasting method 
            R̃ = d.ψ.(Z)
        end

        # Now we have a vector of matrices, where each matrix corresponds to the
        # readout vectors R₁, …, Rₘ for a given data set. Now, aggregate these
        # readout vectors into a single summary statistic for each data set:
        t = d.a.(R̃)
    end

    if !isnothing(d.S)
        s = @ignore_derivatives d.S.(Z) # NB any expert summary statistics S are applied to the original data sets directly (so, if Z[i] is a supergraph, all subgraphs are independent replicates from the same data set)
        if !isnothing(d.ψ)
            t = vcat.(t, s)
        else
            t = s
        end
    end

    return t
end

function _first_N_minus_1_dims_identical(v::AbstractVector{<:GNNGraph})
    # For each graph, extract the node features as a vector
    vecs = [[x.ndata[k] for k in keys(x.ndata)] for x in v]

    # Assume all graphs have the same keys (same number of features)
    k = length(vecs[1])
    @assert all(length(vec) == k for vec in vecs)

    # Split vecs into k vectors, each collecting one feature across graphs
    arrays = [[vec[i] for vec in vecs] for i = 1:k]

    # Check that the dimensions match for each group of feature arrays
    return all(_first_N_minus_1_dims_identical.(arrays))
end

function spatialgraph(S::AbstractMatrix; stationary = true, isotropic = true, kwargs...)

    # Determine neighbourhood based on keyword arguments 
    kwargs = (; kwargs...)
    k = haskey(kwargs, :k) ? kwargs.k : 30
    r = haskey(kwargs, :r) ? kwargs.r : 0.15
    random = haskey(kwargs, :random) ? kwargs.random : false

    if !isotropic
        error("Anistropy is not currently implemented (although it is documented in anticipation of future functionality); please contact the package maintainer")
    end
    if !stationary
        error("Nonstationarity is not currently implemented (although it is documented anticipation of future functionality); please contact the package maintainer")
    end

    S = f32(S)
    A = adjacencymatrix(S; k = k, r = r, random = random)
    S = permutedims(S) # need final dimension to be n-dimensional
    GNNGraph(A, ndata = (S = S,), edata = permutedims(A.nzval))
end
spatialgraph(S::AbstractVector; kwargs...) = batch(spatialgraph.(S; kwargs...)) # spatial locations varying between replicates

# Wrappers that allow data to be passed into an already-constructed graph
# (useful for partial simulation on the fly with the parameters held fixed)
spatialgraph(g::GNNGraph, Z) = GNNGraph(g, ndata = (g.ndata..., Z = reshapeZ(Z)))
reshapeZ(Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray} = stackarrays(reshapeZ.(Z))
reshapeZ(Z::AbstractVector) = reshapeZ(reshape(Z, length(Z), 1))
reshapeZ(Z::AbstractMatrix) = reshapeZ(reshape(Z, 1, size(Z)...))
function reshapeZ(Z::A) where {A <: AbstractArray{T, 3}} where {T}
    # Z is given as a three-dimensional array, with
    # Dimension 1: q, dimension of the response variable (e.g., singleton with univariate data)
    # Dimension 2: n, number of spatial locations
    # Dimension 3: m, number of replicates
    # Permute dimensions 2 and 3 since GNNGraph requires final dimension to be n-dimensional
    permutedims(f32(Z), (1, 3, 2))
end
function reshapeZ(Z::V) where {V <: AbstractVector{M}} where {M <: AbstractMatrix{T}} where {T}
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
# univariate spatial processes, $q = 2$ for bivariate processes)". For fixed locations, we will then write: 
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

# ---- SpatialGraphConv ----

function SpatialGraphConv(
    ch::Pair{Int, Int},
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

function (l::SpatialGraphConv)(g::GNNGraph)
    # GNNGraph(g, ndata = l(g, node_features(g))) # this is the code for generic GNNLayer
    h = l(g, g.ndata.Z) # access the data Z directly, since the spatial locations S are also stored as node features
    GNNGraph(g, ndata = (Z = h, g.ndata.S))
end
function (l::SpatialGraphConv)(g::GNNGraph, x::M) where {M <: AbstractMatrix{T}} where {T}
    l(g, reshape(x, size(x, 1), 1, size(x, 2)))
end
function (l::SpatialGraphConv)(g::GNNGraph, x::A) where {A <: AbstractArray{T, 3}} where {T}
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
    in_channel = size(l.Γ1, ndims(l.Γ1))
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



function NeighbourhoodVariogram(h_max, n_bins::Integer)
    h_cutoffs = range(0, stop = h_max, length = n_bins+1)
    h_cutoffs = collect(h_cutoffs)
    NeighbourhoodVariogram(h_cutoffs)
end
function (l::NeighbourhoodVariogram)(g::GNNGraph)

    # NB in the case of a batched graph, see the comments in the method summarystatistics(d::DeepSet, Z::V) where {V <: AbstractVector{G}} where {G <: GNNGraph}
    Z = g.ndata.Z
    h = g.graph[3]

    message(xi, xj, e) = (xi - xj) .^ 2
    z = apply_edges(message, g, Z, Z, h) # (Zⱼ - Zᵢ)², possibly replicated 
    z = mean(z, dims = 2) # average over the replicates 
    z = vec(z)

    # Bin the distances
    h_cutoffs = l.h_cutoffs
    bins_upper = h_cutoffs[2:end]   # upper bounds of the distance bins
    bins_lower = h_cutoffs[1:(end - 1)] # lower bounds of the distance bins 
    N = [bins_lower[i:i] .< h .<= bins_upper[i:i] for i in eachindex(bins_upper)] # NB avoid scalar indexing by i:i
    N = reduce(hcat, N)

    # Compute the average over each bin
    N_card = sum(N, dims = 1)        # number of occurences in each distance bin 
    N_card = N_card + (N_card .== 0) # prevent division by zero 
    Σ = sum(z .* N, dims = 1)        # ∑(Zⱼ - Zᵢ)² in each bin
    vec(Σ ./ 2N_card)
end
Flux.trainable(l::NeighbourhoodVariogram) = NamedTuple()


end