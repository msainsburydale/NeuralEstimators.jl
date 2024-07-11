module NeuralEstimatorsFoldsExt 

using NeuralEstimators 
using Folds 
import NeuralEstimators: EM

## Identical to main definition, but use parallel version of map() if Folds is available
function (em::EM)(Z::V, θ₀::Union{Vector, Matrix, Nothing} = nothing; args...) where {V <: AbstractVector{A}} where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}

	if isnothing(θ₀)
		@assert !isnothing(em.θ₀) "Please provide initial estimates `θ₀` in the function call or in the `EM` object."
		θ₀ = em.θ₀
	end

	if isa(θ₀, Vector)
		θ₀ = repeat(θ₀, 1, length(Z))
	end

	estimates = Folds.map(eachindex(Z)) do i
		em(Z[i], θ₀[:, i]; args...)
	end
	estimates = reduce(hcat, estimates)

	return estimates
end

end