# markdown code for documentation in docs/src/workflow/advancedusage.md:
# # Combining neural and expert summary statistics
#
# See [`DeepSetExpert`](@ref).

"""
	samplesize(Z)

Computes the sample size m for a set of independent realisations `Z`, often
useful as an expert summary statistic in `DeepSetExpert` objects.

Note that this function is a simple wrapper around `numberreplicates`, but this
function returns the number of replicates as the eltype of `Z`.
"""
samplesize(Z) = eltype(Z)(numberreplicates(Z))

samplesize(Z::V) where V <: AbstractVector = samplesize.(Z)


# ---- DeepSetExpert Type and constructors ----

# we use unicode characters below to preserve readability of REPL help files
"""
	DeepSetExpert(ψ, ϕ, S, a)
	DeepSetExpert(ψ, ϕ, S; a::String)
	DeepSetExpert(deepset::DeepSet, ϕ, S)


A neural estimator in the `DeepSet` representation with additional expert
summary statistics,

```math
θ̂(𝐙) = ϕ((𝐓(𝐙)', 𝐒(𝐙)')'),	 	 𝐓(𝐙) = 𝐚(\\{ψ(𝐙ᵢ) : i = 1, …, m\\}),
```

where 𝐙 ≡ (𝐙₁', …, 𝐙ₘ')' are independent replicates from the model,
`ψ` and `ϕ` are neural networks, `S` is a function that returns a vector
of expert summary statistics, and `𝐚` is a permutation-invariant
aggregation function.

The dimension of the domain of `ϕ` must be qₜ + qₛ, where qₜ and qₛ are the
dimensions of the ranges of `ψ` and `S`, respectively.

The constructor `DeepSetExpert(deepset::DeepSet, ϕ, S)` inherits `ψ` and `a`
from `deepset`.

See `?DeepSet` for discussion on the aggregation function `𝐚`.

# Examples
```
using NeuralEstimators
using Flux

n = 10 # number of observations in each realisation
p = 4  # number of parameters in the statistical model

# Construct the neural estimator
S = samplesize
qₛ = 1
qₜ = 32
w = 16
ψ = Chain(Dense(n, w, relu), Dense(w, qₜ, relu));
ϕ = Chain(Dense(qₜ + qₛ, w), Dense(w, p));
θ̂ = DeepSetExpert(ψ, ϕ, S)

# Apply the estimator to a single set of 3 realisations:
Z = rand(n, 3);
θ̂(Z)

# Apply the estimator to two sets each containing 3 realisations:
Z = [rand(n, m) for m ∈ (3, 3)];
θ̂(Z)

# Apply the estimator to two sets containing 3 and 4 realisations, respectively:
Z = [rand(n, m) for m ∈ (3, 4)];
θ̂(Z)
```
"""
struct DeepSetExpert{F, G, H, K}
	ψ::G
	ϕ::F
	S::H
	a::K
end
#TODO make this a superclass of DeepSet? Would be better to have a single class
# that dispatches to different methods depending on wether S is present or not.

Flux.@functor DeepSetExpert
Flux.trainable(d::DeepSetExpert) = (d.ψ, d.ϕ)

DeepSetExpert(ψ, ϕ, S; a::String = "mean") = DeepSetExpert(ψ, ϕ, S, _agg(a))
DeepSetExpert(deepset::DeepSet, ϕ, S) = DeepSetExpert(deepset.ψ, ϕ, S, deepset.a)

Base.show(io::IO, D::DeepSetExpert) = print(io, "\nDeepSetExpert object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.a)\nExpert statistics: $(D.S)\nOuter network:  $(D.ϕ)")
Base.show(io::IO, m::MIME"text/plain", D::DeepSetExpert) = print(io, D)


# ---- Methods ----

function (d::DeepSetExpert)(Z::A) where {A <: AbstractArray{T, N}} where {T, N}
	t = d.a(d.ψ(Z))
	s = d.S(Z)
	u = vcat(t, s)
	d.ϕ(u)
end

function (d::DeepSetExpert)(tup::Tup) where {Tup <: Tuple{A, B}} where {A <: AbstractArray{T, N}, B <: AbstractVector{T}} where {T, N}
	Z = tup[1]
	x = tup[2]
	t = d.a(d.ψ(Z))
	s = d.S(Z)
	u = vcat(Z, s, x)
	d.ϕ(u)
end

# # Simple, intuitive (although inefficient) implementation using broadcasting:
# function (d::DeepSetExpert)(v::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
#   stackarrays(d.(Z))
# end

# Optimised version. This approach ensures that the neural networks ϕ and ρ are
# applied to arrays that are as large as possible, improving efficiency compared
# with the intuitive method above (particularly on the GPU):
# Note I can't take the gradient of this function... Might have to open an issue with Zygote.
function (d::DeepSetExpert)(Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}

	# Convert to a single large Array
	z = stackarrays(Z)

	# Apply the inner neural network to obtain the neural summary statistics
	ψa = d.ψ(z)

	# Compute the indices needed for aggregation and construct a tuple of colons
	# used to subset all but the last dimension of ψa.
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)

	# Construct the combined neural and expert summary statistics
	u = map(eachindex(Z)) do i
		idx = indices[i]
		t = d.a(ψa[colons..., idx])
		s = d.S(Z[i])
		u = vcat(t, s)
		u
	end
	u = stackarrays(u)

	# Apply the outer network
	d.ϕ(u)
end

function (d::DeepSetExpert)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A <: AbstractArray{T, N}, B <: AbstractVector{T}} where {T, N}

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
		s = d.S(Z[i])
		x = X[i]
		u = vcat(t, s, x)
		u
	end
	u = stackarrays(u)

	# Apply the outer network
	d.ϕ(u)
end
