"""
	Estimates(θ, θ̂, runtime)

A set of true parameters `θ`, corresponding estimates `θ̂`, and the `runtime` to
obtain `θ̂`, as returned by a call to `estimate`.
"""
struct Estimates
	θ::DataFrame
	θ̂::DataFrame
	runtime::DataFrame
end

# Clean printing:
# Base.show(io::IO, E::Estimates) = print(io, "\nEstimates object with:\nTrue values θ: $(E.θ)\nEstimates θ̂:  $(E.θ̂)\nruntime:  $(print(E.runtime))")
# Base.show(io::IO, m::MIME"text/plain", E::Estimates) = print(io, E)

"""
	merge(estimates::Estimates)

Merge `estimates` into a single long-form `DataFrame` containing the true
parameters and the corresponding estimates.
"""
function merge(estimates::Estimates)
	θ = estimates.θ
	θ̂ = estimates.θ̂

	# Replicate θ to match the number of rows in θ̂. Note that the parameter
	# configuration, k, is the fastest running variable in θ̂, so we repeat θ
	# in an outer fashion.
	θ = repeat(θ, outer = nrow(θ̂) ÷ nrow(θ))

	# Transform θ and θ̂ to long form:
	θ = stack(θ, variable_name = :parameter, value_name = :truth)
	θ̂ = stack(θ̂, Not([:estimator, :m, :k]), variable_name = :parameter, value_name = :estimate)

	# Merge θ and θ̂: All we have to do is add :truth column to θ̂
	θ̂[!, :truth] = θ[:, :truth]

	return θ̂
end

"""
	estimate(estimators, parameters::P, m; <keyword args>) where {P <: ParameterConfigurations}

Using a collection of `estimators`, compute estimates from data simulated from a
set of `parameters` with invariant information `ξ`.

Note that `estimate()` requires the user to have defined a method `simulate(parameters, ξ, m::Integer)`.

# Keyword arguments
- `m::Vector{Integer} where I <: Integer`: sample sizes to estimate from.
- `estimator_names::Vector{String}`: names of the estimators (sensible default values provided).
- `parameter_names::Vector{String}`: names of the parameters (sensible default values provided).
- `num_rep::Integer = 1`: the number of times to replicate each parameter in `parameters`. 
- `use_gpu = true`: a `Bool` or a collection of `Bool` objects with length equal to the number of estimators.
"""
function estimate(
    estimators, ξ, parameters::P; m::Vector{I},
	estimator_names::Vector{String} = ["estimator$i" for i ∈ eachindex(estimators)],
	parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(parameters, 1)],
	num_rep::Integer = 1, use_gpu = true
	) where {P <: ParameterConfigurations, I <: Integer}

	obj = map(m) do i
	 	_estimate(
		estimators, ξ, parameters, m = i,
		estimator_names = estimator_names, parameter_names = parameter_names,
		num_rep = num_rep, use_gpu = use_gpu
		)
	end

	θ = obj[1].θ
	θ̂ = vcat(map(x -> x.θ̂, obj)...)
	runtime = vcat(map(x -> x.runtime, obj)...)

	estimates = Estimates(θ, θ̂, runtime)

	return estimates
end

function _estimate(
	estimators, ξ, parameters::P; m::Integer,
	estimator_names::Vector{String} = ["estimator$i" for i ∈ eachindex(estimators)],
	parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(parameters, 1)],
	num_rep::Integer = 1, use_gpu = true
	) where {P <: ParameterConfigurations}

	println("Estimating with m = $m...")

	E = length(estimators)
	p = size(parameters, 1)
	K = size(parameters, 2)
	@assert length(estimator_names) == E
	@assert length(parameter_names) == p

	@assert eltype(use_gpu) == Bool
	if typeof(use_gpu) == Bool use_gpu = repeat([use_gpu], E) end
	@assert length(use_gpu) == E

	# Simulate data
	println("	Simulating data...")
    y = simulate(parameters, ξ, m, num_rep)

	# Initialise a DataFrame to record the run times
	runtime = DataFrame(estimator = [], m = [], time = [])

	θ̂ = map(eachindex(estimators)) do i
		println("	Running estimator $(estimator_names[i])...")
		time = @elapsed θ̂ = _runondevice(estimators[i], y, use_gpu[i])
		push!(runtime, [estimator_names[i], m, time])
		θ̂
	end

    # Convert to DataFrame and add estimator information
    θ̂ = hcat(θ̂...)
    θ̂ = DataFrame(θ̂', parameter_names)
    θ̂[!, "estimator"] = repeat(estimator_names, inner = nrow(θ̂) ÷ E)
    θ̂[!, "m"] = repeat([m], nrow(θ̂))
	θ̂[!, "k"] = repeat(1:K, E * num_rep)

	# Also provide the true parameters for comparison with the estimates
	# θ = repeat(parameters.θ, outer = (1, num_rep))
	θ = DataFrame(parameters.θ', parameter_names)

    return (θ = θ, θ̂ = θ̂, runtime = runtime)
end

# TODO Helper function for combining θ and θ̂ into a single long form data frame that will be
# useful for plotting. Also want to merge into wide data frame for the scenario plots.
