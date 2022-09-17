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

#TODO Change "agg" and "aggregation" to simply "a", which aligns with the notation I use in the paper.

"""
    DeepSet(ψ, ϕ, a)
	DeepSet(ψ, ϕ; a::String = "mean")

A Deep Set neural estimator,

```math
θ̂(𝐙) ≡ ϕ(a(\\{𝐙ᵢ : i = 1, …, m\\})),
```

where 𝐙 ≡ (𝐙₁', …, 𝐙ₘ')' are independent and identically distributed (iid)
realisations from the model under a single parameter vector 𝛉, `ψ` and `ϕ` are
neural networks, and `a` is a permutation-invariant aggregation function. Note
that `ψ` and `ϕ` depend on trainable parameters, but we omit this dependence for
notational convenience.


Although the above defintion of a neural estimator is with respect to a single
data set 𝐙, `DeepSet` estimators instead act on sets of data sets, stored as
`Vector`s of `Array`s, where each array corresponds to one set of iid
realisations from the model. The last dimension of each array stores the
realisations; for example, if 𝐙 is a 3-dimensional array, then 𝐙[:, :, 1]
contains the first realisation, 𝐙[:, :, 2] contains the second realisation, and
so on.

The neural networks `ψ` and `ϕ` and typically `Flux` neural networks. The
function `a` must act on an `Array` and, since it aggregates the iid realisations,
it must aggregate over the last dimension of the array.

There are two `DeepSet` constructors. The first constructor treats `a` as a keyword argument of type `String`, which can take values `"mean"` (default), `"sum"`, or `"logsumexp"` function. The second constructor treats `a` as a positional argument of type `Function`, and this constructor allows the user to provide a custom aggregation function.

# Examples
```
n = 10 # observations in each realisation
p = 5  # number of parameters in the statistical model
w = 32 # width of each layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂ = DeepSet(ψ, ϕ)

# Apply the estimator to a single set of m=3 realisations:
Z = [rand(n, 1, 3)];
θ̂(Z)

# Apply the estimator to two sets each containing m=3 realisations:
Z = [rand(n, 1, m) for m ∈ (3, 3)];
θ̂(Z)

# Apply the estimator to two sets containing m=3 and m=4 realisations, respectively:
Z = [rand(n, 1, m) for m ∈ (3, 4)];
θ̂(Z)
```
"""
struct DeepSet{T, F, G}
	ψ::T
	ϕ::G
	agg::F
end

DeepSet(ψ, ϕ; aggregation::String = "mean") = DeepSet(ψ, ϕ, _agg(aggregation))

@functor DeepSet # allows Flux to optimise the parameters

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
