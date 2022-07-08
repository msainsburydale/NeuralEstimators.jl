
"""
	samplesize(Z::A) where {A <: AbstractArray{T, N}} where {T, N}

Computes the sample size m for a set of independent realisations `Z`, useful
as an expert summary statistic in `DeepSetExpert` objects.
"""
samplesize(Z::A) where {A <: AbstractArray{T, N}} where {T, N} = T(size(Z, ndims(Z)))

"""
	inversesamplesize(Z::A) where {A <: AbstractArray{T, N}} where {T, N}

Computes the inverse of the sample size for a set of independent realisations `Z`, useful
as an expert summary statistic in `DeepSetExpert` objects.
"""
inversesamplesize(Z::A) where {A <: AbstractArray{T, N}} where {T, N} = T(1/size(Z, ndims(Z)))


# ---- DeepSetExpert Type and constructors ----

"""
	DeepSetExpert(ψ, ϕ, S, agg)
Implementation of the Deep Set framework with `ψ` and `ϕ` neural networks, `agg`
a symmetric function that pools data over the last dimension of an array, and
`S` a vector of real-valued functions that serve as expert summary statistics.

The dimension of the domain of `ϕ` should be qₜ + qₛ, where qₜ is the range of `ϕ`
and qₛ is the dimension of `S`, that is, `length(S)`. `DeepSetExpert` objects are
applied to `AbstractVectors` of `AbstractArrays`, where each array is associated
with one parameter vector. The functions `ψ` and `S` both act on these arrays
individually (i.e., they are broadcasted over the `AbstractVector`).
"""
struct DeepSetExpert{F, G, H, K}
	# NB the efficient implementation requires ψ and agg to be kept separately.
	ψ::G
	ϕ::F
	S::H
	agg::K
end

"""
    DeepSetExpert(ψ, ϕ, S; aggregation::String = "mean")

`DeepSetExpert` constructor with `agg` equal to the `"mean"`, `"sum"`, or
`"logsumexp"` function.
"""
DeepSetExpert(ψ, ϕ, S; aggregation::String = "mean") = DeepSetExpert(ψ, ϕ, S, _agg(aggregation))


"""
    DeepSetExpert(deepset::DeepSet, ϕ, S)

`DeepSetExpert` constructor with the aggregation function `agg` and inner neural
network `ψ` inherited from `deepset`.

Note that we cannot inherit the outer network, `ϕ`, since `DeepSetExpert`
objects require the dimension of the domain of `ϕ` to be qₜ + qₛ.
"""
DeepSetExpert(deepset::DeepSet, ϕ, S) = DeepSetExpert(deepset.ψ, ϕ, S, deepset.agg)

# """
#     DeepSetExpert(deepset::DeepSet,  S)
#
# `DeepSetExpert` constructor with the aggregation function `agg` and neural
# networks `ψ` and `ϕ` inherited from `deepset`.
#
# This method assumed that the first layer of `ϕ` is `Dense`.
# """
# function DeepSetExpert(deepset::DeepSet, S)
# 	# add qₛ neurons to the first layer of ϕ
# 	ϕ = deepset.ϕ
# 	qₛ = length(S)
#
# 	DeepSetExpert(deepset, ϕ, S)
# end


# Allow Flux to optimise the parameters:
@functor DeepSetExpert (ψ, ϕ)

# Clean printing:
Base.show(io::IO, D::DeepSetExpert) = print(io, "\nDeepSetExpert object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.agg)\nExpert statistics: $(D.S)\nOuter network:  $(D.ϕ)")
Base.show(io::IO, m::MIME"text/plain", D::DeepSetExpert) = print(io, D)

# ---- DeepSetExpert function ----

function Sfun(x::A, S) where {A <: AbstractArray{T, N}} where {T, N}
	s = [s(x) for s ∈ S]
	s = convert(Base.typename(A).wrapper, s)
	return s
end

# Simple, intuitive (although inefficient) implementation using broadcasting:
function (d::DeepSetExpert)(v::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
  t = d.agg.(d.ψ.(v))
  # s = d.S.(v)
  s = Sfun.(v, Ref(d.S))
  u = vcat.(t, s)
  θ̂ = d.ϕ.(u)
  θ̂ = stackarrays(θ̂)
  return θ̂
end

# Optimised version. This approach ensures that the neural networks ϕ and ρ are
# applied to arrays that are as large as possible, improving efficiency compared
# with the intuitive method above (particularly on the GPU):
# Note I can't take the gradient of this function... Might have to open an issue with Zygote.
# function (d::DeepSetExpert)(v::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
#
# 	# Convert to a single large Array
# 	a = stackarrays(v)
#
# 	# Apply the inner neural network
# 	ψa = d.ψ(a)
#
# 	# Compute the indices needed for aggregation and construct a tuple of colons
# 	# used to subset all but the last dimension of ψa. Note that constructing
# 	# colons in this manner makes the function agnostic to ndims(ψa).
# 	indices = get_indices(v)
# 	colons  = ntuple(_ -> (:), ndims(ψa) - 1)
#
# 	# Compute the neural and expert summary statistics
# 	u = map(eachindex(v)) do i
# 		idx = indices[i]
# 		t = d.agg(ψa[colons..., idx])
# 		s = d.S(v[i])
# 		u = vcat(t, s)
# 		u
# 	end
# 	u = stackarrays(u)
#
# 	# Apply the outer network
# 	θ̂ = d.ϕ(u)
#
# 	return θ̂
# end


# ---- Unused code ----

# # S: iterable of scalar-valued functions, e.g., S = (samplesize, logsumexp).
# function expertstatistics(S, x::A) where {A <: AbstractArray{T, N}} where {T, N}
# 	s = [s(x) for s ∈ S]
# 	s = T.(s)
# 	# Note that the following doesn't work for reshaped arrays: Should make a function that does the following and checks
# 	# Need to do something like the following for the gpu, because the default is a cpu object even if x is on the gpu. The following is very error prone, though.
# 	# ArrayType = Base.typename(A).wrapper # get the non-parametrized type name: See https://stackoverflow.com/a/55977768/16776594
# 	# s = convert(ArrayType, s)
# 	return s
# end
#
# function expertstatistics(S, v::V) where {V <: AbstractArray{A}} where {A <: AbstractArray{T, N}} where {T, N}
# 	s = [expertstatistics(S, x) for x ∈ v] # This may not be on the GPU
# 	return s
# end

# function wrappertype(object)
# 	Base.typename(typeof(object)).wrapper
# end
