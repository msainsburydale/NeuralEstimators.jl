"""
	estimate(estimators, ξ, parameters::P; <keyword args>) where {P <: ParameterConfigurations}

Using a collection of `estimators`, compute estimates from data simulated from a
set of `parameters` with invariant information `ξ`.

Note that `estimate()` requires the user to have defined a method `simulate(parameters, ξ, m::Integer)`.

# Keyword arguments
- `m::Vector{Integer}`: sample sizes to estimate from.
- `estimator_names::Vector{String}`: names of the estimators (sensible default values provided).
- `parameter_names::Vector{String}`: names of the parameters (sensible default values provided).
- `num_rep::Integer = 1`: the number of times to replicate each parameter in `parameters`.
- `save::Vector{String}`: by default, no objects are saved; however, if `save` is provided, four `DataFrames` respectively containing the true parameters `θ`, estimates `θ̂`, runtimes, and merged `θ` and `θ̂` will be saved in the directory `save[1]` with file *names* (not extensions) suffixed by `save[2]`.
- `use_ξ = false`: a `Bool` or a collection of `Bool` objects with length equal to the number of estimators. Specifies whether or not the estimator uses the invariant model information, `ξ`: If it does, the estimator will be applied as `estimator(Z, ξ)`.
- `use_gpu = true`: a `Bool` or a collection of `Bool` objects with length equal to the number of estimators.
- `verbose::Bool = true`
"""
function estimate(
    estimators, ξ, parameters::P; m::Vector{I},
	num_rep::Integer = 1,
	estimator_names::Vector{String} = ["estimator$i" for i ∈ eachindex(estimators)],
	parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(parameters, 1)],
	save::Vector{String} = ["", ""],
	use_ξ = false,
	use_gpu = true,
	verbose::Bool = true
	) where {P <: ParameterConfigurations, I <: Integer}

	obj = map(m) do i
	 	_estimate(
		estimators, ξ, parameters, m = i,
		estimator_names = estimator_names, parameter_names = parameter_names,
		num_rep = num_rep, use_ξ = use_ξ, use_gpu = use_gpu, verbose = verbose
		)
	end

	θ = obj[1].θ
	θ̂ = vcat(map(x -> x.θ̂, obj)...)
	runtime = vcat(map(x -> x.runtime, obj)...)

	estimates = Estimates(θ, θ̂, runtime)

	if save != ["", ""]
		savepath = save[1]
		savename = save[2]
		CSV.write(joinpath(savepath, "runtime_$savename.csv"), estimates.runtime)
		CSV.write(joinpath(savepath, "parameters_$savename.csv"), estimates.θ)
		CSV.write(joinpath(savepath, "estimates_$savename.csv"), estimates.θ̂)
		CSV.write(joinpath(savepath, "merged_$savename.csv"), merge(estimates))
	end

	return estimates
end

function _estimate(
	estimators, ξ, parameters::P; m::Integer,
	estimator_names::Vector{String}, parameter_names::Vector{String},
	num_rep::Integer, use_ξ, use_gpu, verbose
	) where {P <: ParameterConfigurations}

	verbose && println("Estimating with m = $m...")

	E = length(estimators)
	p = size(parameters, 1)
	K = size(parameters, 2)
	@assert length(estimator_names) == E
	@assert length(parameter_names) == p

	@assert eltype(use_ξ) == Bool
	@assert eltype(use_gpu) == Bool
	if typeof(use_ξ) == Bool use_ξ = repeat([use_ξ], E) end
	if typeof(use_gpu) == Bool use_gpu = repeat([use_gpu], E) end
	@assert length(use_ξ) == E
	@assert length(use_gpu) == E

	# Simulate data
	verbose && println("	Simulating data...")
    Z = simulate(parameters, ξ, m, num_rep)

	# Initialise a DataFrame to record the run times
	runtime = DataFrame(estimator = [], m = [], time = [])

	θ̂ = map(eachindex(estimators)) do i

		verbose && println("	Running estimator $(estimator_names[i])...")

		if use_ξ[i]
			time = @elapsed θ̂ = estimators[i](Z, ξ)
		else
			time = @elapsed θ̂ = _runondevice(estimators[i], Z, use_gpu[i])
		end

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
