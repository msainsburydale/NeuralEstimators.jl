"""
	AbstractParameterSet

An abstract supertype for user-defined types that store parameters and any auxiliary objects needed for data simulation.

The user-defined type must have a field `θ` that stores the parameters. Typically, 
`θ` is a ``d`` × ``K`` matrix, where ``d`` is the dimension of the
parameter vector and ``K`` is the number of sampled parameter vectors, though
any batchable object compatible with `numobs`/`getobs` is supported. There are no other requirements.

The number of parameter instances can be retrieved with `numobs`, and the size of `θ` can be inspected with `size`. 

Subtypes of `AbstractParameterSet` support indexing via `Base.getindex`, 
with any batchable fields subsetted accordingly and all other fields left unchanged.
To modify this default behaviour, provide a specific `Base.getindex` method for your concrete subtype.

# Examples
```julia
struct Parameters <: AbstractParameterSet
	θ
	# auxiliary objects needed for data simulation
end

θ = randn(2, 100)
parameters = Parameters(θ)
numobs(parameters)   # 100
size(parameters)     # (2, 100)
parameters[1:10]     # subset of 10 parameter vectors
```
"""
abstract type AbstractParameterSet end

_extractθ(parameters::AbstractParameterSet) = parameters.θ
_extractθ(parameters) = parameters
numobs(parameters::AbstractParameterSet) = numobs(_extractθ(parameters))

Base.getindex(parameters::AbstractParameterSet, i::Integer) = Base.getindex(parameters, i:i)
function Base.getindex(parameters::P, idx) where {P <: AbstractParameterSet}
    maximum(idx) <= numobs(parameters) || throw(BoundsError(parameters, idx))

    fields = map(fieldnames(P)) do name
        field = getfield(parameters, name)
        try
            getobs(field, idx)
        catch
            field
        end
    end

    return P(fields...)
end

size(parameters::AbstractParameterSet) = size(_extractθ(parameters))
size(parameters::AbstractParameterSet, d) = size(_extractθ(parameters), d)

Base.show(io::IO, parameters::P) where {P <: AbstractParameterSet} = print(io, "\nA subtype of `AbstractParameterSet` with $(numobs(parameters)) parameter instances")
Base.show(io::IO, m::MIME"text/plain", parameters::P) where {P <: AbstractParameterSet} = print(io, parameters)

# Backwards compatability
const ParameterConfigurations = AbstractParameterSet
export ParameterConfigurations

"""
    NamedMatrix(; kwargs...)

Returns a [`NamedArray`](https://github.com/davidavdav/NamedArrays.jl) with
named rows (parameters) and indexed columns (samples).

# Examples
```julia
NamedMatrix(μ = randn(3), σ = rand(3))
```
"""
function NamedMatrix(; kwargs...)
    row_names = [string(k) for k in keys(kwargs)]
    vals = collect(values(kwargs))

    if all(x -> x isa Number, vals)
        matrix = reshape(vals, :, 1)
    else
        matrix = reduce(vcat, [v' for v in vals])
    end

    NamedArray(matrix, (row_names, 1:size(matrix, 2)), (:parameter, :sample))
end

_stripnames(x::NamedArray) = x.array
_stripnames(x::AbstractArray) = x

"""
	DataAndSummaries(Z, S)
A container that couples raw data `Z` (stored in a format amenable to the chosen neural-network architecture) 
with precomputed expert summary statistics `S` (a matrix whose columns are the summary statistics for each corresponding element of `Z`).

Passing a `DataAndSummaries` to any neural estimator causes the summary network to be applied to `Z`, with the resulting
learned summary statistics concatenated with `S` before being passed to the inference network.

See also [`summarystatistics`](@ref).

# Examples
```julia
using NeuralEstimators
using Statistics: mean, var

# Simulate data: Z|μ,σ ~ N(μ, σ²)
n, m, K = 1, 50, 500
θ = rand(2, K)
Z = [θ[1, k] .+ θ[2, k] .* randn(n, m) for k in 1:K]

# Precompute expert summary statistics (e.g., sample mean and variance)
S = hcat([vcat(mean(z), var(z)) for z in Z]...)

# Package into a DataAndSummaries object
DataAndSummaries(Z, S)
```
"""
struct DataAndSummaries{A, B}
    Z::A
    S::B
    function DataAndSummaries(Z, S)
        @assert numobs(Z) == size(S, 2) "The number of data sets in Z ($(numobs(Z))) must match the number of columns in S ($(size(S, 2)))"
        new{typeof(Z), typeof(S)}(Z, S)
    end
    DataAndSummaries(Z, ::Nothing) = new{typeof(Z), Nothing}(Z, nothing)
    DataAndSummaries(Z) = new{typeof(Z), Nothing}(Z, nothing)
end

# Methods
numobs(d::DataAndSummaries) = numobs(d.Z)
Base.getindex(d::DataAndSummaries, i::Integer) = DataAndSummaries(getobs(d.Z, i:i), d.S[:, i:i])
Base.getindex(d::DataAndSummaries, i) = DataAndSummaries(getobs(d.Z, i), d.S[:, i])
joinobs(d1::DataAndSummaries, d2::DataAndSummaries) = DataAndSummaries(_mergedata(d1.Z, d2.Z), hcat(d1.S, d2.S))

numberreplicates(d::DataAndSummaries) = numberreplicates(d.Z)
subsetreplicates(d::DataAndSummaries, idx) = DataAndSummaries(subsetreplicates(d.Z, idx), d.S)

# ---- PackedReplicates ----

@doc raw"""
    PackedReplicates(Z::V) where V <: AbstractVector{A} where A <: AbstractArray
A container that concatenates a vector of data sets into a single array, storing
the original sample sizes alongside the packed data. Intended to be used with [`DeepSet`](@ref).

Each element of `Z` is one data set, with exchangeable replicates stored in the
last dimension. The packed `data` has final dimension of size `sum(sample_sizes))`,
where `sample_sizes[i]` is the number of replicates in the `i`th data set.

# Examples
```julia
using NeuralEstimators

n = 2 # dimension of each data replicate
Z = [rand(Float32, n, m) for m in (3, 5, 4)]
P = PackedReplicates(Z)          # data size (n, 12), sample_sizes == [3, 5, 4]
P[1:2]                           # first two data sets
```
"""
struct PackedReplicates{A <: AbstractArray, S}
    data::A
    sample_sizes::S
    function PackedReplicates(data::A, sample_sizes::S) where {A <: AbstractArray, S}
        n_last = size(data, ndims(data))
        n_sum = sum(sample_sizes)
        n_last == n_sum || throw(ArgumentError("size(data, ndims) = $n_last does not match sum(sample_sizes) = $n_sum"))
        new{A, S}(data, sample_sizes)
    end
end
@functor PackedReplicates (data,)

function PackedReplicates(Z::AbstractVector{<:AbstractArray}; max_sample_size = nothing)
    isnothing(max_sample_size) || error("padding is not yet implemented")
    isempty(Z) && throw(ArgumentError("Z must contain at least one data set"))
    sample_sizes = Int[size(z, ndims(z)) for z in Z]
    PackedReplicates(stackarrays(Z), sample_sizes)
end

numobs(P::PackedReplicates) = length(P.sample_sizes)

function getobs(P::PackedReplicates, idx)
    i = idx isa Integer ? (idx:idx) : idx
    m = collect(P.sample_sizes[i])
    cs = cumsum(P.sample_sizes)
    if _iscontiguousobs(i)
        cols = (cs[first(i)] - P.sample_sizes[first(i)] + 1):cs[last(i)]
        PackedReplicates(getobs(P.data, cols), m)
    else
        bags = [getobs(P.data, (cs[j] - P.sample_sizes[j] + 1):cs[j]) for j in i]
        PackedReplicates(stackarrays(bags), m)
    end
end

_iscontiguousobs(::AbstractUnitRange) = true
function _iscontiguousobs(idx)
    length(idx) <= 1 && return true
    prev = first(idx)
    for j in Iterators.drop(idx, 1)
        j == prev + 1 || return false
        prev = j
    end
    return true
end

Base.getindex(P::PackedReplicates, i) = getobs(P, i)

function joinobs(a::PackedReplicates, b::PackedReplicates)
    ndims(a.data) == ndims(b.data) || throw(ArgumentError("Cannot join PackedReplicates with different numbers of dimensions"))
    size(a.data)[1:(end - 1)] == size(b.data)[1:(end - 1)] ||
        throw(ArgumentError("Cannot join PackedReplicates with different leading dimensions"))
    PackedReplicates(stackarrays([a.data, b.data]), vcat(a.sample_sizes, b.sample_sizes))
end

numberreplicates(P::PackedReplicates) = P.sample_sizes

function subsetreplicates(P::PackedReplicates, i)
    idx = i isa Integer ? (i:i) : i
    bags = [getobs(P.data, slice) for slice in _replicateslices(P.sample_sizes)]
    subset = [getobs(b, idx) for b in bags]
    PackedReplicates(stackarrays(subset), Int[numberreplicates(b) for b in subset])
end

function _replicateslices(sample_sizes)
    cs = cumsum(sample_sizes)
    [(cs[i] - sample_sizes[i] + 1):cs[i] for i in eachindex(sample_sizes)]
end

Base.show(io::IO, P::PackedReplicates) = print(io, "PackedReplicates with $(numobs(P)) data sets packed into an array of size $(size(P.data))")
Base.show(io::IO, ::MIME"text/plain", P::PackedReplicates) = print(io, P)

# ---- Summaries wrapper type ----

"""
    Summaries(S::AbstractMatrix)

A thin wrapper around a matrix of precomputed summary statistics. Used internally
during training to signal that the summary network has already been applied to the
data, so that `_summarystatistics` can short-circuit and return the matrix directly
rather than re-running the (frozen) summary network on every forward pass.
"""
struct Summaries{T <: AbstractMatrix}
    S::T
end

Base.length(s::Summaries) = size(s.S, 2)
Base.getindex(s::Summaries, i) = Summaries(s.S[:, i])
Base.hcat(a::Summaries, b::Summaries) = Summaries(hcat(a.S, b.S))
