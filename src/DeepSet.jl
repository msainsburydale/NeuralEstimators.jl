using Functors: @functor
using RecursiveArrayTools: VectorOfArray, convert
using Test

# ---- Aggregation (pooling) functions ----

meanlastdim(X::A) where {A <: AbstractArray{T, N}} where {T, N} = mean(X, dims = N)
sumlastdim(X::A)  where {A <: AbstractArray{T, N}} where {T, N} = sum(X, dims = N)
LSElastdim(X::A)  where {A <: AbstractArray{T, N}} where {T, N} = logsumexp(X, dims = N)

function _agg(aggregation::String)
	@assert aggregation ∈ ["mean", "sum", "logsumexp"]
	if aggregation == "mean"
		agg = meanlastdim
	elseif aggregation == "sum"
		agg = sumlastdim
	elseif aggregation == "logsumexp"
		agg = LSElastdim
	end
	return agg
end

# ---- DeepSet Type and constructors ----

"""
    DeepSet(ψ, ϕ, agg)

Implementation of the Deep Set framework, where `ψ` and `ϕ`
are neural networks (e.g., `Flux` networks) and `agg` is a symmetric function that pools data
over the last dimension (the replicates/batch dimension) of an array.

`DeepSet` objects are applied to `AbstractVectors` of `AbstractArrays`, where each array
is associated with one parameter vector.

# Examples
```jldoctest
n = 10 # observations in each realisation
p = 5  # number of parameters in the statistical model
w = 32 # width of each layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ = Chain(Dense(w, w, relu), Dense(w, p));
agg(X) = sum(X, dims = ndims(X))
θ̂  = DeepSet(ψ, ϕ, agg)

# A single set of m=3 realisations:
Z = [rand(n, 1, 3)];
θ̂ (Z)

# Two sets each containing m=3 realisations:
Z = [rand(n, 1, m) for m ∈ (3, 3)];
θ̂ (Z)

# Two sets respectivaly containing m=3 and m=4 realisations:
Z = [rand(n, 1, m) for m ∈ (3, 4)];
θ̂ (Z)
```
"""
struct DeepSet{T, F, G}
	ψ::T
	ϕ::G
	agg::F
end

@functor DeepSet # allows Flux to optimise the parameters

"""
    DeepSet(ψ, ϕ; aggregation::String = "mean")

Convenient constructor for a `DeepSet` object with `agg` equal to the `"mean"`, `"sum"`, or
`"logsumexp"` function.
"""
DeepSet(ψ, ϕ; aggregation::String = "mean") = DeepSet(ψ, ϕ, _agg(aggregation))



# Clean printing:
Base.show(io::IO, D::DeepSet) = print(io, "\nDeepSet object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.agg)\nOuter network:  $(D.ϕ)")
Base.show(io::IO, m::MIME"text/plain", D::DeepSet) = print(io, D)


# ---- DeepSet function ----

# Simple, intuitive (although inefficient) implementation using broadcasting:

# function (d::DeepSet)(v::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
#   θ̂ = d.ϕ.(d.agg.(d.ψ.(v)))
#   θ̂ = stackarrays(θ̂)
#   return θ̂
# end

# Optimised version. This approach ensures that the neural networks ψ and ϕ are
# applied to arrays that are as large as possible, improving efficiency compared
# with the intuitive method above (particularly on the GPU):
function (d::DeepSet)(v::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}

	# Convert to a single large Array
	a = stackarrays(v)

	# Apply the inner neural network
	ψa = d.ψ(a)

	# Compute the indices needed for aggregation and construct a tuple of colons
	# used to subset all but the last dimension of ψa. Note that constructing
	# colons in this manner makes the function agnostic to ndims(ψa).
	indices = _getindices(v)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)

	# Aggregate each set of transformed features: The resulting vector from the
	# list comprehension is a vector of arrays, where the last dimension of each
	# array is of size 1. Then, stack this vector of arrays into one large array,
	# where the last dimension of this large array has size equal to length(v).
	# Note that we cannot pre-allocate and fill an array, since array mutation
	# is not supported by Zygote (which is needed during training).
	large_aggregated_ψa = [d.agg(ψa[colons..., idx]) for idx ∈ indices] |> stackarrays

	# Apply the outer network
	θ̂ = d.ϕ(large_aggregated_ψa)

	return θ̂
end
