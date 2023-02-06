"""
	PiecewiseEstimator(estimators, m_breaks)
Creates a piecewise estimator from a collection of `estimators`, based on the
collection of sample-size changepoints, `m_breaks`, which should contain one
element fewer than the number of `estimators`.

# Examples
```
# Suppose that we've trained two neural estimators. The first, θ̂₁, is trained
# for small sample sizes (e.g., m ≤ 30), and the second, `θ̂₂`, is trained for
# moderate-to-large sample sizes (e.g., m > 30). Then we construct a piecewise
# estimator with a sample-size changepoint of 30, which dispatches θ̂₁ if m ≤ 30
# and θ̂₂ if m > 30.

n = 2  # bivariate data
p = 3  # number of parameters in the model
w = 8  # width of each layer

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
	m_breaks
	# @assert length(m_breaks) == length(estimators) - 1
end

@functor PiecewiseEstimator (estimators,)


# Note that this is an inefficient implementation, analogous to the inefficient
# DeepSet implementation. A more efficient approach would be to subset Z based
# on m_breaks, apply the estimators to each block of Z, then recombine the estimates.
function (d::PiecewiseEstimator)(Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}


	m = size.(Z, N)

	estimators = d.estimators
	m_breaks  = d.m_breaks

	@assert length(m_breaks) == length(estimators) - 1

	m_breaks = [m_breaks..., Inf]

	θ̂ = map(eachindex(Z)) do i

		# find which estimator to use
		mᵢ = m[i]
		j = findfirst(mᵢ .<= m_breaks)

		# apply the estimator
		estimators[j](Z[[i]])
	end

	# Convert from vector of arrays to matrix
	θ̂ = stackarrays(θ̂)

	return θ̂
end
