using Functors: @functor
using RecursiveArrayTools: VectorOfArray, convert

# ---- Aggregation (pooling) functions ----

meanlastdim(X::A) where {A <: AbstractArray{T, N}} where {T, N} = mean(X, dims = N)
sumlastdim(X::A)  where {A <: AbstractArray{T, N}} where {T, N} = sum(X, dims = N)
LSElastdim(X::A)  where {A <: AbstractArray{T, N}} where {T, N} = logsumexp(X, dims = N)

function _agg(a::String)
	@assert a ∈ ["mean", "sum", "logsumexp"]
	if a == "mean"
		meanlastdim
	elseif a == "sum"
		sumlastdim
	elseif a == "logsumexp"
		LSElastdim
	end
end

# ---- DeepSet Type and constructors ----

# we use unicode characters below to preserve readability of REPL help files
"""
    DeepSet(ψ, ϕ, a)
	DeepSet(ψ, ϕ; a::String = "mean")

A neural estimator in the `DeepSet` representation,

```math
θ̂(𝐙) = ϕ(𝐓(𝐙)),	 	 𝐓(𝐙) = 𝐚(\\{ψ(𝐙ᵢ) : i = 1, …, m\\}),
```

where 𝐙 ≡ (𝐙₁', …, 𝐙ₘ')' are independent replicates from the model, `ψ` and `ϕ`
are neural networks, and `𝐚` is a permutation-invariant aggregation function.

The function `𝐚` must aggregate over the last dimension of an array (i.e., the
replicates dimension). It can be specified as a positional argument of
type `Function`, or as a keyword argument of type `String` with permissible
values `"mean"`, `"sum"`, and `"logsumexp"`.

# Examples
```
using NeuralEstimators
using Flux

n = 10 # number of observations in each realisation
p = 4  # number of parameters in the statistical model

# Construct the neural estimator
w = 32 # width of each layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂ = DeepSet(ψ, ϕ)

# Apply the estimator to a single set of 3 realisations:
Z₁ = rand(n, 3);
θ̂(Z₁)

# Apply the estimator to two sets each containing 3 realisations:
Z₂ = [rand(n, m) for m ∈ (3, 3)];
θ̂(Z₂)

# Apply the estimator to two sets containing 3 and 4 realisations, respectively:
Z₃ = [rand(n, m) for m ∈ (3, 4)];
θ̂(Z₃)

# Repeat the above but with some covariates:
dₓ = 2
ϕₓ = Chain(Dense(w + dₓ, w, relu), Dense(w, p));
θ̂  = DeepSet(ψ, ϕₓ)
x₁ = rand(dₓ)
x₂ = [rand(dₓ), rand(dₓ)]
θ̂((Z₁, x₁))
θ̂((Z₃, x₂))
```
"""
struct DeepSet{T, F, G}
	ψ::T
	ϕ::G
	a::F
end
# 𝐙₁ → ψ() \n
#          ↘ \n
# ⋮     ⋮     a() → ϕ() \n
#          ↗ \n
# 𝐙ₘ → ψ() \n


DeepSet(ψ, ϕ; a::String = "mean") = DeepSet(ψ, ϕ, _agg(a))

@functor DeepSet # allows Flux to optimise the parameters

# Clean printing:
Base.show(io::IO, D::DeepSet) = print(io, "\nDeepSet object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.a)\nOuter network:  $(D.ϕ)")
Base.show(io::IO, m::MIME"text/plain", D::DeepSet) = print(io, D)


# ---- Methods ----

function (d::DeepSet)(Z::A) where {A <: AbstractArray{T, N}} where {T, N}
	d.ϕ(d.a(d.ψ(Z)))
end

function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{A, B}} where {A <: AbstractArray{T, N}, B <: AbstractVector{T}} where {T, N}
	Z = tup[1]
	x = tup[2]
	t = d.a(d.ψ(Z))
	d.ϕ(vcat(t, x))
end


# Simple, intuitive (although inefficient) implementation using broadcasting:

# function (d::DeepSet)(v::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
#   θ̂ = d.ϕ.(d.a.(d.ψ.(v)))
#   θ̂ = stackarrays(θ̂)
#   return θ̂
# end

# Optimised version. This approach ensures that the neural networks ψ and ϕ are
# applied to arrays that are as large as possible, improving efficiency compared
# with the intuitive method above (particularly on the GPU):
function (d::DeepSet)(Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}

	# Convert to a single large Array
	z = stackarrays(Z)

	# Apply the inner neural network
	ψa = d.ψ(z)

	# Compute the indices needed for aggregation and construct a tuple of colons
	# used to subset all but the last dimension of ψa. Note that constructing
	# colons in this manner makes the function agnostic to ndims(ψa).
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)

	# Aggregate each set of transformed features: The resulting vector from the
	# list comprehension is a vector of arrays, where the last dimension of each
	# array is of size 1. Then, stack this vector of arrays into one large array,
	# where the last dimension of this large array has size equal to length(v).
	# Note that we cannot pre-allocate and fill an array, since array mutation
	# is not supported by Zygote (which is needed during training).
	large_aggregated_ψa = [d.a(ψa[colons..., idx]) for idx ∈ indices] |> stackarrays

	# Apply the outer network
	θ̂ = d.ϕ(large_aggregated_ψa)

	return θ̂
end

function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A <: AbstractArray{T, N}, B <: AbstractVector{T}} where {T, N}


	Z = tup[1]
	X = tup[2]

	# Convert to a single large Array
	z = stackarrays(Z)

	# Apply the inner neural network to obtain the neural summary statistics
	ψa = d.ψ(z)

	# Compute the indices needed for aggregation and construct a tuple of colons
	# used to subset all but the last dimension of ψa.
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)

	# concatenate the neural summary statistics with X
	u = map(eachindex(Z)) do i
		idx = indices[i]
		t = d.a(ψa[colons..., idx])
		x = X[i]
		u = vcat(t, x)
		u
	end
	u = stackarrays(u)

	# Apply the outer network
	θ̂ = d.ϕ(u)

	return θ̂
end
