"""
	Assessment(θ, θ̂, runtime)

A set of true parameters `θ`, corresponding estimates `θ̂`, and the `runtime` to
obtain `θ̂`, as returned by a call to `assess`.
"""
struct Assessment
	θ::DataFrame
	θ̂::DataFrame
	runtime::DataFrame
end

"""
	merge(assessment::Assessment)

Merge `assessment` into a single long-form `DataFrame` containing the true
parameters and the corresponding estimates.
"""
function merge(assessment::Assessment)

	θ = assessment.θ
	θ̂ = assessment.θ̂

	# Replicate θ to match the number of rows in θ̂. Note that the parameter
	# configuration, k, is the fastest running variable in θ̂, so we repeat θ
	# in an outer fashion.
	θ = repeat(θ, outer = nrow(θ̂) ÷ nrow(θ))

	# Transform θ and θ̂ to long form:
	θ = stack(θ, variable_name = :parameter, value_name = :truth)
	θ̂ = stack(θ̂, Not([:estimator, :m, :k, :replicate]), variable_name = :parameter, value_name = :estimate)

	# Merge θ and θ̂: All we have to do is add :truth column to θ̂
	θ̂[!, :truth] = θ[:, :truth]

	return θ̂
end

# Internal constructor for Assessment
function _Assessment(obj, save)

	θ = obj[1].θ
	θ̂ = vcat(map(x -> x.θ̂, obj)...)
	runtime = vcat(map(x -> x.runtime, obj)...)

	assessment = Assessment(θ, θ̂, runtime)

	if save != ["", ""]
		savepath = save[1]
		savename = save[2]
		CSV.write(joinpath(savepath, "runtime_$savename.csv"), assessment.runtime)
		CSV.write(joinpath(savepath, "parameters_$savename.csv"), assessment.θ)
		CSV.write(joinpath(savepath, "estimates_$savename.csv"), estimates.θ̂)
		CSV.write(joinpath(savepath, "merged_$savename.csv"), merge(assessment))
	end

	return assessment
end





"""
	assess(estimators, ξ, parameters::P; <keyword args>) where {P <: ParameterConfigurations}

Using a collection of `estimators`, compute estimates from data simulated from a
set of `parameters` with invariant information `ξ`.

Note that `assess()` requires the user to have defined a method `simulate(parameters, ξ, m::Integer)`.

# Keyword arguments
- `m::Vector{Integer}`: sample sizes to estimate from.
- `estimator_names::Vector{String}`: names of the estimators (sensible default values provided).
- `parameter_names::Vector{String}`: names of the parameters (sensible default values provided).
- `J::Integer = 1`: the number of times to replicate each parameter in `parameters`.
- `save::Vector{String}`: by default, no objects are saved; however, if `save` is provided, four `DataFrames` respectively containing the true parameters `θ`, estimates `θ̂`, runtimes, and merged `θ` and `θ̂` will be saved in the directory `save[1]` with file *names* (not extensions) suffixed by `save[2]`.
- `use_ξ = false`: a `Bool` or a collection of `Bool` objects with length equal to the number of estimators. Specifies whether or not the estimator uses the invariant model information, `ξ`: If it does, the estimator will be applied as `estimator(Z, ξ)`.
- `use_gpu = true`: a `Bool` or a collection of `Bool` objects with length equal to the number of estimators.
- `verbose::Bool = true`
"""
function assess(
    estimators, ξ, parameters::P;
	m::Vector{I}, J::Integer = 1,
	estimator_names::Vector{String} = ["estimator$i" for i ∈ eachindex(estimators)],
	parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(parameters, 1)],
	save::Vector{String} = ["", ""],
	use_ξ = false,
	use_gpu = true,
	verbose::Bool = true
	) where {P <: ParameterConfigurations, I <: Integer}

	obj = map(m) do i

		# Simulate data
		verbose && println("	Simulating data...")
		Z = simulate(parameters, ξ, i, J)

	 	_assess(
			estimators, ξ, parameters, Z, J = J,
			estimator_names = estimator_names, parameter_names = parameter_names,
			use_ξ = use_ξ, use_gpu = use_gpu, verbose = verbose
		)
	end

	return _Assessment(obj, save)
end


# A version for fixed Z.
# data should look like:
# n = 1
# K = 4
# Z = [[rand(n, 1, m) for k in 1:K] for m in (5, 10, 15)]
# TODO not sure what to do with ξ. Don't need it for simulation but some of the estimators may need it. For now just provide nothing.
function assess(
	estimators, ξ, parameters::P, Z; # TODO enforce Z to be a Vector{Vector{Array}}
	estimator_names::Vector{String} = ["estimator$i" for i ∈ eachindex(estimators)],
	parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(parameters, 1)],
	save::Vector{String} = ["", ""],
	use_ξ = false,
	use_gpu = true,
	verbose::Bool = true
	) where {P <: ParameterConfigurations}


	# Infer all_m from Z and check that Z is in the correct format
	all_m = broadcast.(x -> size(x)[end], Z)
	all_m = unique.(all_m)
	@assert length(all_m) == length(Z) "Z should be a Vector{Vector{Array}} where " #TODO finish error message

	# Check that the number of parameters are consistent with other quantities
	K = size(parameters, 2)
	KJ = unique(length.(Z))
	@assert length(KJ) == 1
	KJ = KJ[1]
	@assert KJ % K == 0 "The number of data sets in Z must be a multiple of the number of parameters"
	J = KJ ÷ K
	J > 1 && @info "There are more data sets than parameters; this is fine, but ensure that the data are repeated in an inner fashion, so that replicates from a given set of parameters are adjacent." #TODO tidy the wording up here.

	obj = map(Z) do z

		_assess(
			estimators, ξ, parameters, z, J = J,
			estimator_names = estimator_names, parameter_names = parameter_names,
			use_ξ = use_ξ, use_gpu = use_gpu, verbose = verbose
		)

	end

	return _Assessment(obj, save)
end


function _assess(
	estimators, ξ, parameters::P, Z;
	estimator_names::Vector{String}, parameter_names::Vector{String},
	J::Integer, use_ξ, use_gpu, verbose
	) where {P <: ParameterConfigurations}

	# Infer m from Z and check that Z is in the correct format
	m = unique(broadcast(x -> size(x)[end], Z))
	@assert length(m) == 1 "Z should be a Vector{Array} where " #TODO finish error message
	m = m[1]
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
	θ̂[!, "k"] = repeat(1:K, E * J)
	θ̂[!, "replicate"] = repeat(repeat(1:J, inner = K), E)

	# Also provide the true parameters for comparison with the estimates
	θ = DataFrame(parameters.θ', parameter_names)

    return (θ = θ, θ̂ = θ̂, runtime = runtime)
end
