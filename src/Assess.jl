
# ---- Assessment ----

"""
	Assessment(θandθ̂, runtime)

An object for storing the result of calling `assess()`, containing a `DataFrame`
of true parameters `θ` and corresponding estimates `θ̂`, and a `DataFrame`
containing the `runtime` for each estimator.
"""
struct Assessment
	θandθ̂::DataFrame
	runtime::DataFrame
end


# Given a set of true parameters θ and corresponding estimates θ̂ resulting
# from a call to assess(), merge θ and θ̂ into a single long-form DataFrame.
function _merge(θ, θ̂)

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
function _Assessment(obj)

	θ = obj[1].θ
	θ̂ = vcat(map(x -> x.θ̂, obj)...)
	runtime = vcat(map(x -> x.runtime, obj)...)

	assessment = Assessment(_merge(θ, θ̂), runtime)

	return assessment
end


# ---- assess() ----

"""
	assess(estimators, parameters, Z; <keyword args>)
	assess(estimators, parameters; <keyword args>)

Using a collection of `estimators`, compute estimates from data simulated from a
set of `parameters`.

Testing data can be automatically simulated by overloading `simulate` with a method
`simulate(parameters, m::Integer)`, and using the keyword argument `m` to specify
the desired sample sizes to assess. Alternatively, one may provide simulated data
`Z` as a `Vector{Vector{Array}}`, where each `Vector{Array}` is associated with
a single sample size (i.e., the size of the final dimension of the arrays should
be constant). If there are more simulated data sets than unique parameter
vectors, the data should be stored in an 'outer' fashion, so that the parameter
vectors run faster than the replicated data.


# Keyword arguments
- `m::Vector{Integer}`: sample sizes to estimate from.
- `estimator_names::Vector{String}`: names of the estimators (sensible default values provided).
- `parameter_names::Vector{String}`: names of the parameters (sensible default values provided).
- `J::Integer = 1`: the number of times to replicate each parameter in `parameters`.
- `ξ = nothing`: invariant model information.
- `use_ξ = false`: a `Bool` or a collection of `Bool` objects with length equal to the number of estimators. Specifies whether or not the estimator uses the invariant model information, `ξ`: If it does, the estimator will be applied as `estimator(Z, ξ)`.
- `use_gpu = true`: a `Bool` or a collection of `Bool` objects with length equal to the number of estimators.
- `verbose::Bool = true`
"""
function assess(
    estimators, parameters::P;
	m::Vector{I}, J::Integer = 1,
	estimator_names::Vector{String} = ["estimator$i" for i ∈ eachindex(estimators)],
	parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(parameters, 1)],
	ξ = nothing,
	use_ξ = false,
	use_gpu = true,
	verbose::Bool = true
	) where {P <: Union{AbstractMatrix, ParameterConfigurations}, I <: Integer}

	obj = map(m) do i

		# Simulate data
		verbose && println("	Simulating data...")
		Z = simulate(parameters, i, J)

	 	_assess(
			estimators, parameters, Z, J = J,
			estimator_names = estimator_names, parameter_names = parameter_names,
			ξ = ξ, use_ξ = use_ξ, use_gpu = use_gpu, verbose = verbose
		)
	end

	return _Assessment(obj)
end



function assess(
	estimators, parameters::P, Z;
	estimator_names::Vector{String} = ["estimator$i" for i ∈ eachindex(estimators)],
	parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(parameters, 1)],
	ξ = nothing,
	use_ξ = false,
	use_gpu = true,
	verbose::Bool = true
	) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

	# Infer all_m from Z and check that Z is in the correct format
	all_m = broadcast.(z -> size(z)[end], Z)
	all_m = unique.(all_m)
	@assert length(all_m) == length(Z) "The simulated data Z should be a `Vector{Vector{Array}}`, where each `Vector{Array}` is associated with a single sample size (i.e., the size of the final dimension of the arrays should be constant)."

	# Check that the number of parameters are consistent with other quantities
	K = size(parameters, 2)
	KJ = unique(length.(Z))
	@assert length(KJ) == 1
	KJ = KJ[1]
	@assert KJ % K == 0 "The number of data sets in Z must be a multiple of the number of parameters"
	J = KJ ÷ K
	J > 1 && @info "There are more simulated data sets than unique parameter vectors; ensure that the data are replicated in an 'outer' fashion, so that the parameter vectors run faster than the replicated data sets."

	obj = map(Z) do z

		_assess(
			estimators, parameters, z, J = J,
			estimator_names = estimator_names, parameter_names = parameter_names,
			ξ = ξ, use_ξ = use_ξ, use_gpu = use_gpu, verbose = verbose
		)

	end

	return _Assessment(obj)
end



"""
	numberofreplicates(X)

Generic function that returns the number of replicates in a given object.
"""
function numberofreplicates end

function numberofreplicates(X)
	size(X)[end]
end

function numberofreplicates(X::G) where {G <: GNNGraph}
	X.num_graphs
end


function _assess(
	estimators, parameters::P, Z;
	estimator_names::Vector{String}, parameter_names::Vector{String},
	J::Integer, ξ, use_ξ, use_gpu, verbose
	) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

	# Infer m from Z and check that Z is in the correct format
	m = unique(numberofreplicates.(Z))
	@assert length(m) == 1 "The simulated data Z should be a `Vector{Array}` where the size of the final dimension of the array is constant."
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
	runtime = DataFrame(estimator = String[], m = Int64[], time = Float64[])

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
	θ = DataFrame(_extractθ(parameters)', parameter_names)

    return (θ = θ, θ̂ = θ̂, runtime = runtime)
end


import Base: merge
function merge(assessment::Assessment, assessments::Assessment...)
	θandθ̂   = assessment.θandθ̂
	runtime = assessment.runtime
	for x in assessments
		θandθ̂   = vcat(θandθ̂,   x.θandθ̂)
		runtime = vcat(runtime, x.runtime)
	end
	Assessment(θandθ̂, runtime)
end
