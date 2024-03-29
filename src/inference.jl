#TODO when we add them, this will be easily extended to NLE and NPE (whatever methods allows a density to be evaluated)
#TODO documentation and unit testing
#TODO also allow prior to be a vector with length(prior) == size(theta_grid, 2)
# theta_grid: a (fine) gridding of the parameter space, given as a matrix with p rows, where p is the number of parameters in the model
function sample(est::RatioEstimator,
				Z,
				N::Integer = 1000;
			    prior::Function = θ -> 1f0,
				method::String = "IS",
				theta_grid = nothing,
				kwargs...)

	if method == "IS"
		@assert !isnothing(theta_grid) "theta_grid is required when method = 'IS'"
		theta_grid = Float32.(theta_grid) # convert for efficiency and to avoid warnings
		r = vec(estimateinbatches(est, Z, theta_grid; kwargs...)) # r = vec(est(Z, theta_grid))
		density = prior.(eachcol(theta_grid)) .* r
		θ = StatsBase.wsample(eachcol(theta_grid), density, N; replace = true)
		reduce(hcat, θ)
	else
		@error "Only method = 'IS' (inverse-transform sampling) is currently implemented"
	end

end
function sample(est::RatioEstimator, Z::AbstractVector, args...; kwargs...)
	sample.(Ref(est), Z, args...; kwargs...)
end

function mle(est::RatioEstimator, Z; theta_grid = nothing, kwargs...)
	theta_grid = Float32.(theta_grid) # convert for efficiency and to avoid warnings
	r = vec(estimateinbatches(est, Z, theta_grid; kwargs...)) # r = vec(est(Z, theta_grid))
	theta_grid[:, argmax(r), :] # extra colon to preserve matrix output
end
function mle(est::RatioEstimator, Z::AbstractVector; kwargs...)
	reduce(hcat, mle.(Ref(est), Z; kwargs...))
end
