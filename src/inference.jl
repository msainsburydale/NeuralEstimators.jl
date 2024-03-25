#TODO when we add them, this will be easily extended to NLE and NPE (whatever methods allows a density to be evaluated)
#TODO documentation
#TODO maybe should also allow prior to be a vector with length(prior) == size(theta_grid, 2), so
function sample(est::RatioEstimator, Z, N::Integer = 10000; prior::Function = θ -> 1f0, method::String = "IS", theta_grid = nothing)
	if method == "IS"
		@assert !isnothing(theta_grid) "theta_grid must be given when method = 'IS'"
		theta_grid = Float32.(theta_grid) # convert for efficiency and to avoid warnings
		r = vec(est(Z, theta_grid))
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
