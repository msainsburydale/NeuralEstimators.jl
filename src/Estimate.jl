# NB Assumes that m is a vector or a Tuple of integers; should enforce this by
# requiring m to be iterable, or similar.
function estimate(
    estimators::AbstractVector, Ω, K, m;
	titles::Vector{String}, num_rep::Integer = 1, use_gpu = true,
	) where {P <: ParameterConfigurations}

	obj = map(m) do i
	 	estimate(estimators, params, i, type = type, titles = titles,
				 num_rep = num_rep, use_gpu = use_gpu)
	end

	# Extract the parameter estimates and run times.
	if length(m) > 1
		θ̂ = vcat(map(x -> x.θ̂, obj)...)
		runtime = vcat(map(x -> x.runtime, obj)...)
	else
		θ̂ = obj.θ̂
		runtime = obj.runtime
	end

	# Also provide the true parameters for comparison with the estimates
	θ = repeat(params.θ, outer = (1, num_rep))
	θ = DataFrame(θ', params.param_names)

	# TODO Here, merge θ and θ̂ and make a long form data frame that will be
	# useful for plotting.

	return (θ = θ, θ̂ = θ̂, runtime = runtime)
end

"""
The argument `use_gpu` should be a `Bool` or a `Vector{Bool}` with
`length(use_gpu) == length(estimators)`.
"""
function estimate(
	estimators::AbstractVector,
	params, m::Integer;
	titles::Vector{String},
	savepath::String, num_rep::Integer = 1,
	use_gpu = true,
	)

	println("Estimating with m = $m...")

	@assert length(use_gpu) ∈ (1, length(estimators))
	@assert eltype(use_gpu) == Bool
	if typeof(use_gpu) == Bool
		use_gpu = repeat([use_gpu], length(estimators))
	end

	# Simulate data
	println("	Simulating data...")
    y = simulate(params, m, num_rep)

	# Initialise a DataFrame to record the run times
	runtime = DataFrame(estimator = [], m = [], time = [])

	θ̂ = map(eachindex(estimators)) do i
		time = @elapsed θ̂ = _runondevice(estimators[i], y, use_gpu[i])
		push!(runtime, [titles[i], m, time])
		θ̂
	end

    # Convert to DataFrame and add estimator information
    θ̂ = hcat(θ̂...)
    θ̂ = DataFrame(θ̂', params.param_names)
    θ̂[!, "estimator"] = repeat(titles, inner = nrow(θ̂) ÷ length(titles))
    θ̂[!, "m"] = repeat([m], nrow(θ̂))

    return (θ = θ, θ̂ = θ̂, runtime = runtime)
end
