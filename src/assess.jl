# ---- Assessment ----

"""
	Assessment(θandθ̂::DataFrame, runtime::DataFrame)

A type for storing the output of `assess()`. It contains two fields.
The field `runtime` contains the total `time` taken for each `estimator` for each sample size `m`.
The field `θandθ̂` is a long-form `DataFrame` containing the true parameters and corresponding estimates.
Specifically, its columns are:

- `estimator`: the name of the estimator
- `parameter`: the name of the parameter
- `truth`:     the true value of the parameter
- `estimate`:  the estimated value of the parameter
- `m`:         the sample size
- `k`:         the index of the parameter vector in the test set
- `j`: the index of the data set

Multiple `Assessment` objects can be combined with the function `merge`.
"""
struct Assessment
	θandθ̂::DataFrame
	runtime::DataFrame
end

function merge(assessment::Assessment, assessments::Assessment...)
	θandθ̂   = assessment.θandθ̂
	runtime = assessment.runtime
	for x in assessments
		θandθ̂   = vcat(θandθ̂,   x.θandθ̂)
		runtime = vcat(runtime, x.runtime)
	end
	Assessment(θandθ̂, runtime)
end


"""
	risk(assessment::Assessment; loss = (x, y) -> abs(x - y), average_over_parameters = true)

Estimates the Bayes risk with respect to the `loss` function for each
estimator, parameter, and sample size considered in `assessment`.

The argument `loss` should be a binary operator (default absolute-error loss).

If `average_over_parameters = true` (default), the risk is averaged over
all parameters; otherwise, the risk is evaluated over each parameter separately.
"""
function risk(assessment::Assessment; loss = (x, y) -> abs(x - y), average_over_parameters::Bool = true)

	df = assessment.θandθ̂
	grouping_variables = [:estimator, :m]
	if !average_over_parameters push!(grouping_variables, :parameter) end
	df = groupby(df, grouping_variables)
	df = combine(df, [:estimate, :truth] => ((x, y) -> loss.(x, y)) => :loss, ungroup = false)
	df = combine(df, :loss => mean => :risk)

	return df
end


# ---- assess() ----


"""
	assess(estimators, parameters, Z; <keyword args>)
	assess(estimators, parameters; <keyword args>)

Using a collection of `estimators`, compute estimates from data `Z` simulated
from a set of `parameters`.

The data `Z` should be an iterable collection, where each element contains
testing data for a single sample size. If there are more simulated data sets
than unique parameter vectors, the data should be stored in an 'outer' fashion,
so that the parameter vectors run faster than the replicated data.

# Keyword arguments
- `estimator_names::Vector{String}`: names of the estimators (sensible defaults provided).
- `parameter_names::Vector{String}`: names of the parameters (sensible defaults provided). If `ξ` is provided with a field `parameter_names`, those names will be used.
- `ξ = nothing`: an arbitrary collection of objects that are fixed (e.g., distance matrices).
- `use_ξ = false`: a `Bool` or a collection of `Bool` objects with length equal to the number of estimators. Specifies whether or not the estimator uses `ξ`: if it does, the estimator will be applied as `estimator(Z, ξ)`. This argument is useful when multiple `estimators` are provided, only some of which need `ξ`; hence, if only one estimator is provided and `ξ` is not `nothing`, `use_ξ` is automatically set to `true`.
- `use_gpu = true`: a `Bool` or a collection of `Bool` objects with length equal to the number of estimators.
- `verbose::Bool = true`

# Examples
```
using NeuralEstimators
using Flux

n = 10 # number of observations in each realisation
p = 4  # number of parameters in the statistical model

# Construct the neural estimator
w = 32 # width of each layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂ = DeepSet(ψ, ϕ)

# Generate fake parameters and corresponding data for a range of sample sizes:
K = 100
θ = rand(p, K)
Z = [[rand(n, m) for _ ∈ 1:K] for m ∈ (1, 10, 20)]

assessment = assess([θ̂], θ, Z)
risk(assessment)
risk(assessment, average_over_parameters = false)
```
"""
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
	J > 1 && verbose && @info "There are more simulated data sets than unique parameter vectors; ensure that the data are replicated in an 'outer' fashion, so that the parameter vectors run faster than the replicated data sets."

	assessments = map(Z) do z
		_assess(
			estimators, parameters, z, J = J,
			estimator_names = estimator_names, parameter_names = parameter_names,
			ξ = ξ, use_ξ = use_ξ, use_gpu = use_gpu, verbose = verbose
		)
	end

	return merge(assessments...)
end

# NB this chould be an exported method; only reason it's not is because
# I can't distinguish by Z. Also would need
function _assess(
	estimators, parameters::P, Z;
	estimator_names::Vector{String},
	parameter_names::Vector{String},
	J::Integer, ξ, use_ξ, use_gpu, verbose
	) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

	# Infer m from Z and check that it is constant
	m = unique(numberreplicates(Z))
	@assert length(m) == 1 "The simulated data `Z` should contain a fixed number of replicates for each parameter vector in `parameters`"
	m = m[1]
	verbose && println("Estimating with m = $m...")

	# Extract the parameter names from ξ if it was provided
	if !isnothing(ξ) && haskey(ξ, :parameter_names)
		parameter_names = ξ.parameter_names
	end

	E = length(estimators)
	p = size(parameters, 1)
	K = size(parameters, 2)
	@assert length(estimator_names) == E
	@assert length(parameter_names) == p

	# If only one estimator is provided and ξ is not nothing, use_ξ is
	# automatically set to true.
	if E == 1 && !isnothing(ξ)
		use_ξ = true
	end

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
			# pass ξ to the estimator by passing a closure to _runondevice().
			# This approach allows the estimator to use the gpu, and provides a
			# consistent format of the estimates regardless of whether or not
			# ξ is used.
			time = @elapsed θ̂ = _runondevice(z -> estimators[i](z, ξ), Z, use_gpu[i]) # old code: θ̂ = estimators[i](Z, ξ)
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
	θ̂[!, "j"] = repeat(repeat(1:J, inner = K), E) # NB "j" used to be "replicate"
	θ̂[!, "replicate"] = repeat(repeat(1:J, inner = K), E) # for backwards compatability

	# Also provide the true parameters for comparison with the estimates
	θ = DataFrame(_extractθ(parameters)', parameter_names)

	return Assessment(_merge(θ, θ̂), runtime)
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
	θ̂ = stack(θ̂, Not([:estimator, :m, :k, :j, :replicate]), variable_name = :parameter, value_name = :estimate)

	# Merge θ and θ̂: All we have to do is add :truth column to θ̂
	θ̂[!, :truth] = θ[:, :truth]

	return θ̂
end
