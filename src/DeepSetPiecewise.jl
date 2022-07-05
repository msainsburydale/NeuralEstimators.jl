
# Idea:
# 1. Have some arbitrary number of neural estimators, each of which is intended
# 	 to be optimal for a specific non-overlapping range of sample sizes.
# 2. Want to store these neural estmators and the corresponding sample-size
#	 ranges in struct. The challenge is that we do not know the number of neural
#	 estimators in advance.
# 3. We can use a macro to automatically generate a struct with the correct length.
#	 Alternatively, and more simply, we can store the estimators as a collection
#    in a single field.



"""
	DeepSetPiecewise(estimators, m_cutoffs)
Given an arbitrary number of `estimators`, creates a piecewise neural estimator
based on the sample size cut offs, `m_cutoffs`, which should contain one element
fewer than the number of estimators.

# Examples

Suppose that we have two neural estimators, `θ̂₁` and `θ̂₂`, taking the following
arbitrary forms:

```
n = 10
p = 5
w = 32

ψ₁ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₁ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂₁ = DeepSet(ψ₁, ϕ₁)

ψ₂ = Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu));
ϕ₂ = Chain(Dense(w, w, relu), Dense(w, w, relu), Dense(w, p));
θ̂₂ = DeepSet(ψ₂, ϕ₂)
```

Further suppose that we've trained `θ̂₁` for small sample sizes (e.g., m ≤ 30)
and `θ̂₂` for moderate-to-large sample sizes (e.g., m > 30). Then we construct a
piecewise Deep Set object with a cut-off sample size of 30 which dispatches
θ̂₁ if m ≤ 30 and θ̂₂ if m > 30:

```
θ̂ = DeepSetPiecewise((θ̂₁, θ̂₂), (30,))
Z = [rand(Float32, n, 1, m) for m ∈ (10, 50)]
θ̂(Z)
```
"""
struct DeepSetPiecewise
	estimators
	m_cutoffs
	# @assert length(m_cutoffs) == length(estimators) - 1
end

@functor DeepSetPiecewise (estimators,)


# Note that this is an inefficient implementation, analogous to the inefficient
# DeepSet implementation. A more efficient approach would be to subset Z based
# on m_cutoffs, apply the estimators to each block of Z, then recombine the estimates.
function (d::DeepSetPiecewise)(Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}


	m = size.(Z, N)

	estimators = d.estimators
	m_cutoffs  = d.m_cutoffs

	@assert length(m_cutoffs) == length(estimators) - 1

	m_cutoffs = [m_cutoffs..., Inf]

	θ̂ = map(eachindex(Z)) do i

		# find which estimator to use
		mᵢ = m[i]
		j = findfirst(mᵢ .<= m_cutoffs)

		# apply the estimator
		estimators[j](Z[[i]])
	end

	# Convert from vector of arrays to matrix
	θ̂ = stackarrays(θ̂)

	return θ̂
end
