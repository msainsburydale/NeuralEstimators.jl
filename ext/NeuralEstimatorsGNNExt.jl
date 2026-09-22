module NeuralEstimatorsGNNExt

using NeuralEstimators
using Flux
using Flux: @ignore_derivatives, glorot_uniform
using GraphNeuralNetworks
using GraphNeuralNetworks: check_num_nodes
using NNlib: scatter, gather, ⊠
using Statistics, Random, LinearAlgebra
import NeuralEstimators: subsetreplicates, numberreplicates, _deepsetsummaries, spatialgraph
import NeuralEstimators: GNNSummary, SpatialGraphConv, IndicatorWeights, KernelWeights, NeighbourhoodVariogram
import NeuralEstimators: PackedGraphs, _packbatch
using NeuralEstimators: ReplicatesInFeatures, ReplicatesInSubgraphs, GroupedPackedGraphs, DataAndSummaries
using NeuralEstimators: _aggregatemiddle, _aggregatereplicates, _rowofsummaries, numobs

function subsetreplicates(Z::G, i) where {G <: GNNGraph}
    if typeof(i) <: Integer
        i = i:i
    end
    x = :Z ∈ keys(Z.ndata) ? Z.ndata.Z : first(values(Z.ndata))
    if :Z ∈ keys(Z.ndata) && ndims(x) == 3 && size(x, 2) > 1
        # Replicates are stored in the second dimension of the node features: subsetting is a
        # cheap array slice. Splat the other node features (e.g., the spatial locations S) so
        # that they are preserved
        GNNGraph(Z; ndata = (Z.ndata..., Z = x[:, i, :]))
    else
        # Replicates are stored as subgraphs
        # @warn "`subsetreplicates()` is slow for graphical data."
        # TODO getgraph() doesn't currently work with the GPU: see https://github.com/CarloLucibello/GraphNeuralNetworks.jl/issues/161
        # TODO getgraph() doesn’t return duplicates. So subsetreplicates(Z, [1, 1]) returns just a single graph
        # TODO can't check for CuArray (and return to GPU) because CuArray won't always be defined (no longer depend on CUDA) and we can't overload exact signatures in package extensions... it's low priority, but will be good to fix when time permits. Hopefully, the above issue with GraphNeuralNetworks.jl will get fixed, and we can then just remove the call to cpu() below
        Z = cpu(Z)
        Z = getgraph(Z, i)
        Z
    end
end

function numberreplicates(Z::G) where {G <: GNNGraph}
    x = :Z ∈ keys(Z.ndata) ? Z.ndata.Z : first(values(Z.ndata))
    # NB a singleton second dimension is not the replicate axis: graphs constructed with
    # spatial locations varying between replicates store the replicates as subgraphs, and
    # their node features are of size (q, 1, n)
    if ndims(x) == 3 && size(x, 2) > 1
        size(x, 2)
    else
        Z.num_graphs
    end
end

# ---- Packing a batch of graphs ----

"""
    PackedGraphs(Z::AbstractVector{<:GNNGraph})

Packs a vector of graphical data sets into a single supergraph, recording the number of
replicates in each data set so that they can be aggregated after the readout.

Two storage layouts are supported, and may not be mixed within a batch:

- `ReplicatesInFeatures`: each data set is a single graph whose node features store the
  replicates in their second dimension, as produced by `spatialgraph(S::AbstractMatrix, Z)`.
  When the number of replicates varies, the replicate dimension is padded to a common length
  and a mask is recorded.
- `ReplicatesInSubgraphs`: the replicates of each data set are stored as subgraphs, as
  produced by `spatialgraph(S::AbstractVector, Z)`. No padding is needed, since the subgraphs
  are concatenated along the node dimension.

Peak memory in the graph layers scales as the number of replicate slots times the number of
edges in the batch (padded slots included), so with many replicates, large graphs or a large
batch, reduce the batchsize. During training and inference, batches with a varying number of
replicates are split into a few groups of similar size (see `GroupedPackedGraphs`),
which keeps the padding small.
"""
function PackedGraphs(Z::AbstractVector{<:GNNGraph})
    layout, m = _replicatelayout(Z)

    # NB Flux.batch() mutates arrays, so it must not appear on the automatic-differentiation
    # tape. Packing is data marshalling only: gradients reach the parameters of ψ through
    # ψ(⋅), and the gradient with respect to the data is never needed
    if layout isa ReplicatesInSubgraphs || allequal(m)
        return PackedGraphs(Flux.batch(Z), m, layout)
    else
        # Flux.batch() concatenates three-dimensional node features along their final
        # (node) dimension, which requires the leading dimensions to be identical, so pad
        # the replicate dimension to a common length and mask the padded entries
        M = maximum(m)
        padded = map(Z) do z
            GNNGraph(z; ndata = (z.ndata..., Z = _padreplicates(z.ndata.Z, M)))
        end
        return PackedGraphs(Flux.batch(padded), m, _replicatemask(m, M), layout)
    end
end

# Determines how the replicates of a batch of graphs are stored, and how many each data set
# has, checking that the batch can be packed
function _replicatelayout(Z::AbstractVector{<:GNNGraph})
    isempty(Z) && throw(ArgumentError("Z must contain at least one data set"))
    ks = keys(first(Z).ndata)
    all(z -> keys(z.ndata) == ks, Z) || throw(ArgumentError("all graphs in a batch must carry the same node-feature keys, found $(unique([keys(z.ndata) for z in Z]))"))

    num_subgraphs = [z.num_graphs for z in Z]
    feature_replicates = map(Z) do z
        x = :Z ∈ keys(z.ndata) ? z.ndata.Z : first(values(z.ndata))
        ndims(x) == 3 ? size(x, 2) : 1
    end
    in_features = any(>(1), feature_replicates)
    in_subgraphs = any(>(1), num_subgraphs)
    if in_features && in_subgraphs
        throw(ArgumentError("the replicates of a data set must be stored either in the second dimension of the node features or as subgraphs, but not both; found node features with $(maximum(feature_replicates)) replicates and a graph with $(maximum(num_subgraphs)) subgraphs"))
    end

    m = numberreplicates.(Z)
    layout = in_subgraphs ? ReplicatesInSubgraphs() : ReplicatesInFeatures()
    if layout isa ReplicatesInFeatures && !allequal(m)
        :Z ∈ ks || throw(ArgumentError("padding the replicate dimension requires the node features holding the data to be named Z, found $(ks)"))
    end
    return layout, m
end

# Splits a batch whose data sets have m[k] replicates and size s[k] (nodes plus edges) into at
# most maxgroups groups, so as to minimise the padded work Σ_g max(m in g) × Σ(s in g). The
# optimal groups are contiguous runs of the data sets sorted by m, so this is a small dynamic
# program over the sorted order. Of the optimal splits into 1, …, maxgroups groups, the one
# with the fewest groups whose cost is within rtol of the best is returned (fewer groups means
# fewer forward passes). Returns a vector of index vectors into the original batch
function _replicategroups(m::AbstractVector{<:Integer}, s::AbstractVector{<:Integer}; maxgroups::Integer = 4, rtol::Real = 0.05)
    K = length(m)
    length(s) == K || throw(ArgumentError("m and s must have the same length"))
    p = sortperm(m)
    mₛ = m[p]
    cs = [0; cumsum(s[p])]
    cost(i, j) = mₛ[j] * (cs[j + 1] - cs[i]) # group formed by sorted data sets i, …, j

    G = min(maxgroups, K)
    best = fill(typemax(Int), G, K)  # best[g, j]: cheapest split of the first j into g groups
    start = zeros(Int, G, K)         # first data set of the final group in that split
    for j ∈ 1:K
        best[1, j] = cost(1, j)
        start[1, j] = 1
    end
    for g ∈ 2:G, j ∈ g:K, i ∈ g:j
        c = best[g - 1, i - 1] + cost(i, j)
        if c < best[g, j]
            best[g, j] = c
            start[g, j] = i
        end
    end

    total = best[:, K]
    g = findfirst(≤((1 + rtol) * minimum(total)), total)
    groups = Vector{Vector{Int}}(undef, g)
    j = K
    for h ∈ g:-1:1
        i = start[h, j]
        groups[h] = p[i:j]
        j = i - 1
    end
    return groups
end

# Pads the replicate dimension by repeating the first replicate.
# The padded values are discarded by the mask, so any finite value would do. We repeat an
# existing replicate rather than padding with zeros so that the padded slices stay within the
# range of the real data: with zero padding, every edge of a padded replicate evaluates the
# function f at an exactly zero difference, which is a singular point for some plausible
# choices of f (for example, a fractional power, whose derivative there is infinite).
# Zero padding happens to be safe for the default PowerDifference under the current AD stack
# (ForwardDiff special-cases a zero base in its power rule, giving a zero partial rather than
# 0 * log(0) = NaN), but that is an implementation detail we would rather not depend on
function _padreplicates(x::AbstractArray{T, 3}, M) where {T}
    m = size(x, 2)
    m == M && return x
    return cat(x, repeat(view(x, :, 1:1, :), 1, M - m, 1); dims = 2)
end

# A matrix of ones and zeros indicating which of the M padded replicate slots are real.
# Constructed on the host; it reaches the device with the rest of the PackedGraphs object
function _replicatemask(m, M)
    mask = zeros(Float32, M, length(m))
    for (k, mₖ) in enumerate(m)
        mask[1:mₖ, k] .= 1.0f0
    end
    return mask
end

# Pack a batch of graphs before it is moved to the device (see _packbatch in src/utility.jl).
# When the replicates are stored in the node features and their number varies, the batch is
# grouped by the number of replicates so that little of it is padding
function _packbatch(Z::AbstractVector{<:GNNGraph})
    layout, m = _replicatelayout(Z)
    if layout isa ReplicatesInFeatures && !allequal(m)
        groups = _replicategroups(m, [z.num_nodes + z.num_edges for z in Z])
        if length(groups) > 1
            return GroupedPackedGraphs([PackedGraphs(Z[idx]) for idx in groups], invperm(reduce(vcat, groups)))
        end
    end
    return PackedGraphs(Z)
end
_packbatch(d::DataAndSummaries{<:AbstractVector{<:GNNGraph}}) = DataAndSummaries(_packbatch(d.Z), d.S)

# ---- Summary statistics for packed graphs ----

# Multiple data sets: optimised version for graph data
function _deepsetsummaries(d::DeepSet, Z::V) where {V <: AbstractVector{G}} where {G <: GNNGraph}
    P = @ignore_derivatives _packbatch(Z)
    return _deepsetsummaries(d, P)
end

# Groups of data sets with similar numbers of replicates: each group is a separate supergraph,
# and the columns of the concatenated summaries are put back into the original order
function _deepsetsummaries(d::DeepSet, G::GroupedPackedGraphs)
    t = reduce(hcat, map(P -> _deepsetsummaries(d, P), G.groups))
    return t[:, G.order]
end

# Replicates in the node features: the readout gives an array of size (nf, M, K), and the
# replicates of every data set are aggregated over the middle dimension in a single call
function _deepsetsummaries(d::DeepSet, P::PackedGraphs{<:Any, <:Any, <:Any, ReplicatesInFeatures})
    @assert typeof(d.ψ) <: GNNSummary "For graph input data, the summary network ψ should be a `GNNSummary` object"
    R = _readout(d.ψ, P.graph)
    K = numobs(P)
    nf = size(R, 1)
    rest = length(R) ÷ nf
    rest % K == 0 || throw(ArgumentError("the readout produced summaries of size $(size(R)), which does not divide evenly among the $K data sets in the batch; the readout module must reduce each graph to a fixed-length vector for each replicate"))
    R = reshape(R, nf, rest ÷ K, K)
    t = _aggregatemiddle(d.a, R, P.mask)
    if !isnothing(d.S)
        s = if isnothing(P.mask)
            @ignore_derivatives _rowofsummaries(d.S, P, t)
        else
            # Derive m from the traced mask, mirroring the treatment of padded PackedReplicates
            _rowofsummaries(d.S, P, t)
        end
        t = vcat(t, s)
    end
    return t
end

# Replicates as subgraphs: the readout gives one vector per replicate, with the replicates of
# all data sets stored contiguously in the final dimension, which is exactly the layout that
# _aggregatereplicates expects (a reshape for equal m, a segmented reduction otherwise)
function _deepsetsummaries(d::DeepSet, P::PackedGraphs{<:Any, <:Any, <:Any, ReplicatesInSubgraphs})
    @assert typeof(d.ψ) <: GNNSummary "For graph input data, the summary network ψ should be a `GNNSummary` object"
    R = _readout(d.ψ, P.graph)
    R = reshape(R, size(R, 1), :)
    size(R, 2) == sum(P.sample_sizes) || throw(ArgumentError("the readout produced $(size(R, 2)) summary vectors, but the batch contains $(sum(P.sample_sizes)) replicates in total; the readout module must reduce each subgraph to a fixed-length vector"))
    t = _aggregatereplicates(d.a, R, P.sample_sizes)
    if !isnothing(d.S)
        s = @ignore_derivatives _rowofsummaries(d.S, P, t)
        t = vcat(t, s)
    end
    return t
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
    A = haskey(kwargs, :metric) ?
        adjacencymatrix(S; k = k, r = r, random = random, metric = kwargs.metric) :
        adjacencymatrix(S; k = k, r = r, random = random)
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
        # NB the output layer uses softplus rather than the activation g. The weights are
        # normalised over each neighbourhood, so they need only be non-negative; but with a
        # relu output (the default g) a sizeable fraction of initialisations clamp every edge
        # weight to exactly zero, which zeroes h̄ and hence the whole Γ2 h̄ term, and relu's
        # zero gradient there means the weight function can never recover. softplus is
        # strictly positive, so the branch cannot die
        w = Chain(
            Dense(1 => w_width, g, init = init),
            Dense(w_width => w_out, softplus, init = init)
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
    # The weights depend only on the fixed spatial information s, so when w holds no
    # trainable parameters this whole path is a constant: keeping it off the tape avoids
    # tracing dozens of edge-sized intermediates per layer per gradient step
    w̃ = if _wtrainable(l.w)
        coerce3Darray(normalise_edge_neighbors(g, l.w(s)))
    else
        @ignore_derivatives coerce3Darray(normalise_edge_neighbors(g, l.w(s)))
    end
    # Sanity check: aggregate_neighbors(g, +, w̃) # zeros and ones

    # Compute spatially-weighted sum of input features over each neighbourhood 
    #msg = apply_edges((l, xi, xj, w̃) -> w̃ .* l.f(xi, xj), g, l, x, x, w̃)
    msg = apply_edges((xi, xj, w̃) -> w̃ .* l.f(xi, xj), g, x, x, w̃)
    h̄ = aggregate_neighbors(g, +, msg) # sum over each neighbourhood individually 

    l.g.(_densemul(l.Γ1, x) .+ _densemul(l.Γ2, h̄) .+ l.b) #NB any missingness will cause the feature vector to be entirely missing
end
function Base.show(io::IO, l::SpatialGraphConv)
    in_channel = size(l.Γ1, ndims(l.Γ1))
    out_channel = size(l.Γ1, ndims(l.Γ1)-1)
    print(io, "SpatialGraphConv(", in_channel, " => ", out_channel)
    l.g == identity || print(io, ", ", l.g)
    print(io, ", w=", l.w)
    print(io, ")")
end

# Whether the spatial weight function carries trainable parameters, and hence whether its
# evaluation must be differentiated through. Deliberately a dispatch-based trait rather than
# a runtime check of Optimisers.trainable: it is type-stable, and a weight function that is
# trainable can never be silently gated out (the fallback differentiates)
_wtrainable(::KernelWeights) = false
_wtrainable(::IndicatorWeights) = false
_wtrainable(_) = true

# Applies the weight matrix Γ to every (replicate, node) column of the three-dimensional
# feature array x. NB this is equivalent to batched_mul(Γ, x), but batched_mul broadcasts Γ
# over the final dimension of x, which is the node dimension, giving one tiny matrix
# multiplication per node (degenerate when there is a single replicate). Flattening the
# replicate and node dimensions instead performs the same arithmetic as a single matrix
# multiplication, forwards and in the pullback
function _densemul(Γ, x::AbstractArray{T, 3}) where {T}
    y = Γ * reshape(x, size(x, 1), :)
    return reshape(y, size(Γ, 1), size(x, 2), size(x, 3))
end

# Coerces the edge weights to a three-dimensional array with a singleton replicate
# dimension. NB the replicate dimension is deliberately left as a singleton to be broadcast
# against the messages, rather than materialised with repeat(): the weights do not vary over
# the replicates, so repeating them allocates a (w_out, m, num_edges) array (and its
# pullback) for no gain
function coerce3Darray(x)
    if isa(x, AbstractVector)
        x = permutedims(x)
    end
    if isa(x, AbstractMatrix)
        x = reshape(x, size(x, 1), 1, size(x, 2))
    end
    return x
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

# Applies the propagation and readout modules, returning the readout output unreshaped.
# The readout typically gives an array of size (nf, m, num_graphs), which the packed path
# relies on in order to aggregate the replicates of each data set in a single call
function _readout(ψ::GNNSummary, g::GNNGraph)

    # Propagation module
    h = ψ.propagation(g)
    Z = :Z ∈ keys(h.ndata) ? h.ndata.Z : first(values(h.ndata))

    # Readout module, computes a fixed-length vector (a summary statistic) for each replicate
    return ψ.readout(h, Z)
end

function (ψ::GNNSummary)(g::GNNGraph)
    # R is a matrix with:
    # nrows = number of summary statistics
    # ncols = number of replicates
    R = _readout(ψ, g)
    return reshape(R, size(R, 1), :)
end

function NeighbourhoodVariogram(h_max, n_bins::Integer)
    h_cutoffs = range(0, stop = h_max, length = n_bins+1)
    h_cutoffs = collect(h_cutoffs)
    NeighbourhoodVariogram(h_cutoffs)
end
function (l::NeighbourhoodVariogram)(g::GNNGraph)

    # NB in the case of a batched graph, see the comments in the method _deepsetsummaries(d::DeepSet, Z::V) where {V <: AbstractVector{G}} where {G <: GNNGraph}
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
