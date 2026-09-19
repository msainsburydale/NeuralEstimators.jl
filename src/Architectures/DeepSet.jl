# ---- DeepSet ----

#TODO Remove ElementwiseAggregator?

@concrete struct ElementwiseAggregator
    a
end
(e::ElementwiseAggregator)(x::A) where {A <: AbstractArray{T, N}} where {T, N} = e.a(x, dims = N)

@doc raw"""
    DeepSet(ψ, ϕ, a = mean; condition_on_sample_size = false)
    DeepSet(ψ; latent_dim, output_dim, a = mean, condition_on_sample_size = false, kwargs...)
	(object::DeepSet)(Z::V) where V <: AbstractVector{A} where A
    (object::DeepSet)(P::PackedReplicates)
The DeepSets representation ([Zaheer et al., 2017](https://arxiv.org/abs/1703.06114); [Sainsbury-Dale et al., 2024](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522)),
```math
\mathbf{S}(\mathbf{Z}) = \boldsymbol{\phi}(\mathbf{T}(\mathbf{Z})), \quad
\mathbf{T}(\mathbf{Z}) = \mathbf{a}(\{\boldsymbol{\psi}(\mathbf{Z}_i) : i = 1, \dots, m\}),
```
where 𝐙 ≡ (𝐙₁', …, 𝐙ₘ')' are exchangeable replicates of data, 
`ψ` and `ϕ` are neural networks, and `a` is a permutation-invariant aggregation
function. 

The function `a` must operate on arrays and have a keyword argument `dims` for 
specifying the dimension of aggregation (e.g., `mean`, `sum`, `maximum`, `minimum`, `logsumexp`).

The convenience constructor `DeepSet(ψ; latent_dim, output_dim, ...)` builds `ϕ` as an [`MLP`](@ref),
where `latent_dim` is the dimension of $\mathbf{T}(\mathbf{Z})$ (i.e., the output dimension of `ψ`) and 
`output_dim` is the dimension of $\mathbf{S}(\mathbf{Z})$. Additional keyword arguments are passed to [`MLP`](@ref).

`DeepSet` objects act on data of type `Vector{A}`, where each
element of the vector is associated with one data set (i.e., one set of exchangeable replicates), and where `A` depends on the chosen architecture for `ψ`. 
Exchangeable replicates within each data set are stored in the batch dimension. For example, with data collected over a two-dimensional grid and with `ψ` chosen to be a CNN, `A` should be a 4-dimensional array, 
with replicates stored in the 4ᵗʰ dimension. Alternatively, a vector of arrays may be wrapped in [`PackedReplicates`](@ref) so that the data are stored as a single multidimensional array and 
device-transfer overhead is reduced when training with a GPU.

For computational efficiency, array data are first concatenated along their final dimension 
(i.e., the replicates dimension) before being passed into the inner network `ψ`, 
thereby ensuring that `ψ` is applied to a single large array rather than multiple small ones.

When data sets of varying sample size $m$ are envisaged, set
`condition_on_sample_size = true` to concatenate $\log m$ with $\mathbf{T}(\mathbf{Z})$
before it is passed to `ϕ`:
```math
\mathbf{S}(\mathbf{Z}) = \boldsymbol{\phi}((\mathbf{T}(\mathbf{Z})', \log m)').
```
In this case, the input dimension of `ϕ` must be one greater than the dimension of $\mathbf{T}(\mathbf{Z})$.

!!! note "Variable sample sizes and Reactant"
    When training with `Reactant` on data sets of varying sample size, data `Z` must be wrapped in `PackedReplicates(Z; max_sample_size = M)` with a fixed cap `M` so that every batch has the same array shape.

!!! note "Graph data"
    Graph data via [`GNNSummary`](@ref) is currently supported only with the `Flux` backend.

# Examples
```julia
using NeuralEstimators, Flux

# Two data sets containing 3 and 4 replicates
d = 5  # number of parameters in the model
n = 10 # dimension of each replicate
Z = [rand32(n, m) for m ∈ (3, 4)]

# Construct DeepSet object
latent_dim = 16
output_dim = 8
ψ  = Chain(Dense(n, 32, relu), Dense(32, latent_dim, relu))
ϕ  = Chain(Dense(latent_dim, 32, relu), Dense(32, output_dim))
ds = DeepSet(ψ, ϕ)

# Convenience constructor
ds = DeepSet(ψ; latent_dim = latent_dim, output_dim = output_dim)

# Apply DeepSet object to data
ds(Z)
ds(PackedReplicates(Z))
```
"""
@concrete struct DeepSet
    ψ
    ϕ
    a
    S
end
function DeepSet(ψ, ϕ, a::Function = mean; condition_on_sample_size::Bool = false)
    S = condition_on_sample_size ? logsamplesize : nothing
    DeepSet(ψ, ϕ, ElementwiseAggregator(a), S)
end
function DeepSet(ψ; a::Function = mean, latent_dim::Integer, output_dim::Integer, condition_on_sample_size::Bool = false, backend = nothing, kwargs...)
    in_dim = latent_dim + Int(condition_on_sample_size)
    if isnothing(backend)
        try
            backend = _backendof(ψ)
        catch
        end
    end
    ϕ = MLP(in_dim, output_dim; backend = backend, kwargs...)
    DeepSet(ψ, ϕ, a; condition_on_sample_size)
end
Base.show(io::IO, D::DeepSet) = print(io, "\nDeepSet object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.a)\nConditioning on log sample size: $(!isnothing(D.S))\nOuter network:  $(D.ϕ)")

# Single data set
function (d::DeepSet)(Z::A) where {A}
    d.ϕ(_deepsetsummaries(d, Z))
end
# Multiple data sets
function (d::DeepSet)(Z::V) where {V <: AbstractVector{A}} where {A}
    # Stack into a single array before applying the outer network
    d.ϕ(_stacksummaries(_deepsetsummaries(d, Z)))
end
function (d::DeepSet)(P::PackedReplicates)
    d.ϕ(_deepsetsummaries(d, P))
end

# The summaries are returned as a single array (array data), or as one array per data set (graph data and the broadcasting fallback)
_stacksummaries(t::AbstractArray) = t
_stacksummaries(t::AbstractVector{<:AbstractArray}) = stackarrays(t)

# Single data set
function _deepsetsummaries(d::DeepSet, Z::A) where {A}
    t = d.a(d.ψ(Z))
    if !isnothing(d.S)
        s = @ignore_derivatives d.S(Z)
        t = vcat(t, s)
    end
    return t
end
# Multiple data sets: general fallback using broadcasting
function _deepsetsummaries(d::DeepSet, Z::V) where {V <: AbstractVector{A}} where {A}
    _deepsetsummaries.(Ref(d), Z)
end

# Multiple data sets: optimised version for array data
function _deepsetsummaries(d::DeepSet, Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
    if _first_N_minus_1_dims_identical(Z) #TODO is this check slow?
        # Packing is data marshalling only: gradients reach the parameters of ψ through ψ(⋅), and
        # the gradient with respect to the data is never needed, so keep the packing off the AD tape
        P = @ignore_derivatives PackedReplicates(Z)
        return _deepsetsummaries(d, P)
    else
        # Array sizes differ, so therefore cannot stack together; use simple (and slower) broadcasting method (identical to general fallback method defined above)
        return _deepsetsummaries.(Ref(d), Z)
    end
end

function _deepsetsummaries(d::DeepSet, P::PackedReplicates)
    t = _aggregatereplicates(d.a, d.ψ(P.data), P)
    if !isnothing(d.S)
        s = if isnothing(P.mask)
            @ignore_derivatives _rowofsummaries(d.S, P, t)
        else
            # Derive m from the traced mask so log m is not baked into the XLA graph
            _rowofsummaries(d.S, P, t)
        end
        t = vcat(t, s)
    end
    return t
end

# Applies S to each data set, returning a row vector with the same array type as t (so that vcat(t, s) stays on device)
function _rowofsummaries(S, Z, t::AbstractArray{T}) where {T}
    s = similar(t, T, 1, length(Z))
    copyto!(s, T[S(z) for z ∈ Z])
    return s
end
function _rowofsummaries(S, P::PackedReplicates, t::AbstractArray{T}) where {T}
    if isnothing(P.mask)
        s = similar(t, T, 1, numobs(P))
        copyto!(s, T.(S(P)))
        return s
    else
        return reshape(T.(S(P)), 1, :)
    end
end

"""
    _aggregatereplicates(a, ψa, mᵢ)

Aggregates the replicates of each data set, where the replicates of all data sets are stored contiguously
in the final dimension of `ψa` and `mᵢ[i]` gives the number of replicates in the `i`th data set.

Returns an array whose final dimension indexes the data sets.

The aggregation is done in a single vectorised call, rather than by aggregating each data set in turn.
The latter is much slower under automatic differentiation, since the pullback of each slice `ψa[.., idx]`
allocates an array the size of the whole of `ψa`, making the reverse pass quadratic in the number of data sets.
"""
function _aggregatereplicates(a::ElementwiseAggregator, ψa::AbstractArray, P::PackedReplicates)
    if isnothing(P.mask)
        return _aggregatereplicates(a, ψa, P.sample_sizes)
    else
        return _maskedaggregate(a.a, ψa, P.mask)
    end
end
function _aggregatereplicates(a::ElementwiseAggregator, ψa::AbstractArray{T, N}, mᵢ) where {T, N}
    K = length(mᵢ)
    if allequal(mᵢ)
        # Equal sample sizes: give the replicates their own dimension and aggregate over it,
        # which supports any aggregation function taking a dims keyword argument
        x = reshape(ψa, size(ψa)[1:(N - 1)]..., first(mᵢ), K)
        return dropdims(a.a(x, dims = N); dims = N)
    elseif _segmentable(a.a)
        # Varying sample sizes: aggregate with a single segmented reduction
        idx = @ignore_derivatives _bagindices(ψa, mᵢ)
        return _segmentedaggregate(a.a, ψa, idx, (size(ψa)[1:(N - 1)]..., K))
    else
        return _aggregateeachdataset(a, ψa, mᵢ)
    end
end
_aggregatereplicates(a, ψa, P::PackedReplicates) = _aggregatereplicates(a, ψa, P.sample_sizes)
_aggregatereplicates(a, ψa, mᵢ) = _aggregateeachdataset(a, ψa, mᵢ)

# Aggregates each data set in turn, for aggregation functions that cannot be expressed as a segmented reduction
function _aggregateeachdataset(a, ψa, mᵢ)
    cs = @ignore_derivatives cumsum(mᵢ)
    indices = @ignore_derivatives [(cs[i] - mᵢ[i] + 1):cs[i] for i ∈ eachindex(mᵢ)]
    return stackarrays(map(idx -> a(getobs(ψa, idx)), indices))
end

# Maps each replicate in the stacked array to the data set that it belongs to. The indices are placed on the
# same device as ψa, since the pullback of scatter() constructs its workspace with similar(idx, ⋅)
function _bagindices(ψa, mᵢ)
    idx = similar(ψa, Int32, sum(mᵢ))
    copyto!(idx, inverse_rle(1:length(mᵢ), mᵢ))
    return idx
end

# Aggregation functions that can be expressed as a segmented reduction over the final dimension.
# Each function listed here must have a corresponding method of _segmentedaggregate() below
_segmentable(a) = false
_segmentable(::typeof(mean)) = true
_segmentable(::typeof(sum)) = true
_segmentable(::typeof(maximum)) = true
_segmentable(::typeof(minimum)) = true
_segmentable(::typeof(logsumexp)) = true

_segmentedaggregate(::typeof(mean), ψa, idx, dstsize) = scatter(mean, ψa, idx; dstsize = dstsize)
_segmentedaggregate(::typeof(sum), ψa, idx, dstsize) = scatter(+, ψa, idx; dstsize = dstsize)
_segmentedaggregate(::typeof(maximum), ψa, idx, dstsize) = scatter(max, ψa, idx; dstsize = dstsize)
_segmentedaggregate(::typeof(minimum), ψa, idx, dstsize) = scatter(min, ψa, idx; dstsize = dstsize)
function _segmentedaggregate(::typeof(logsumexp), ψa, idx, dstsize)
    # Shift by the maximum of each data set for numerical stability. Since logsumexp is invariant to
    # this shift, the shift is treated as a constant (as it is in standard implementations of logsumexp)
    mx = @ignore_derivatives scatter(max, ψa, idx; dstsize = dstsize)
    e = exp.(ψa .- @ignore_derivatives(gather(mx, idx)))
    return mx .+ log.(scatter(+, e, idx; dstsize = dstsize))
end

# Padded PackedReplicates: rectangular layout (…, M, K) with a traced 0/1 mask of size (M, K)
function _maskedaggregate(a, ψa::AbstractArray{T, N}, mask) where {T, N}
    _segmentable(a) || throw(ArgumentError("Padded PackedReplicates only support mean, sum, maximum, minimum, and logsumexp aggregation"))
    M, K = size(mask)
    x = reshape(ψa, size(ψa)[1:(N - 1)]..., M, K)
    w = reshape(mask, ntuple(_ -> 1, N - 1)..., M, K)
    return _maskedaggregate_op(a, x, w, N)
end
_maskedaggregate_op(::typeof(mean), x, w, dims) =
    dropdims(sum(x .* w, dims = dims) ./ sum(w, dims = dims); dims = dims)
_maskedaggregate_op(::typeof(sum), x, w, dims) =
    dropdims(sum(x .* w, dims = dims); dims = dims)
function _maskedaggregate_op(::typeof(maximum), x, w, dims)
    T = eltype(x)
    dropdims(maximum(ifelse.(w .> 0, x, T(-Inf)); dims = dims); dims = dims)
end
function _maskedaggregate_op(::typeof(minimum), x, w, dims)
    T = eltype(x)
    dropdims(minimum(ifelse.(w .> 0, x, T(Inf)); dims = dims); dims = dims)
end
function _maskedaggregate_op(::typeof(logsumexp), x, w, dims)
    T = eltype(x)
    dropdims(logsumexp(ifelse.(w .> 0, x, T(-Inf)); dims = dims); dims = dims)
end

function _first_N_minus_1_dims_identical(arrays::Vector{<:AbstractArray})
    # Get the size of the first array up to N-1 dimensions
    first_size = size(arrays[1])[1:(end - 1)]

    # Loop over the remaining arrays and compare their first N-1 dimensions
    for i = 2:length(arrays)
        if size(arrays[i])[1:(end - 1)] != first_size
            return false  # Dimensions do not match
        end
    end

    return true  # All arrays have the same first N-1 dimensions
end
