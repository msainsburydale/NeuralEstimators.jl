
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
	PiecewiseEstimator(estimators, mchange)
Creates a piecewise estimator from a collection of `estimators`, based on the
collection of sample-size changepoints, `mchange`, which should contain one
element fewer than the number of `estimators`.

# Examples
```
# Suppose that we've trained two neural estimators. The first, θ̂₁, is trained
# for small sample sizes (e.g., m ≤ 30), and the second, `θ̂₂`, is trained for
# moderate-to-large sample sizes (e.g., m > 30). Then we construct a piecewise
# estimator with a sample-size changepoint of 30, which dispatches θ̂₁ if m ≤ 30
# and θ̂₂ if m > 30.

n = 2
p = 3
w = 8

ψ₁ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₁ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂₁ = DeepSet(ψ₁, ϕ₁)

ψ₂ = Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu));
ϕ₂ = Chain(Dense(w, w, relu), Dense(w, w, relu), Dense(w, p));
θ̂₂ = DeepSet(ψ₂, ϕ₂)

θ̂ = PiecewiseEstimator([θ̂₁, θ̂₂], [30])
Z = [rand(n, 1, m) for m ∈ (10, 50)]
θ̂(Z)
```
"""
struct PiecewiseEstimator
	estimators
	mchange
	# @assert length(mchange) == length(estimators) - 1
end

@functor PiecewiseEstimator (estimators,)


# Note that this is an inefficient implementation, analogous to the inefficient
# DeepSet implementation. A more efficient approach would be to subset Z based
# on mchange, apply the estimators to each block of Z, then recombine the estimates.
function (d::PiecewiseEstimator)(Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}


	m = size.(Z, N)

	estimators = d.estimators
	mchange  = d.mchange

	@assert length(mchange) == length(estimators) - 1

	mchange = [mchange..., Inf]

	θ̂ = map(eachindex(Z)) do i

		# find which estimator to use
		mᵢ = m[i]
		j = findfirst(mᵢ .<= mchange)

		# apply the estimator
		estimators[j](Z[[i]])
	end

	# Convert from vector of arrays to matrix
	θ̂ = stackarrays(θ̂)

	return θ̂
end
