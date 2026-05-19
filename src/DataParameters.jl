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
    matrix = reduce(vcat, [v' for v in values(kwargs)])
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