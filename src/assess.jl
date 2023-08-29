# ---- Assessment ----

"""
	Assessment(df::DataFrame, runtime::DataFrame)

A type for storing the output of `assess()`. The field `runtime` contains the
total time taken for each estimator. The field `df` is a long-form `DataFrame`
with columns:

- `estimator`: the name of the estimator
- `parameter`: the name of the parameter
- `truth`:     the true value of the parameter
- `estimate`:  the estimated value of the parameter
- `m`:         the sample size (number of iid replicates)
- `k`:         the index of the parameter vector in the test set
- `j`:         the index of the data set

Multiple `Assessment` objects can be combined with `merge()`.
"""
struct Assessment
	df::DataFrame
	runtime::DataFrame
end

function merge(assessment::Assessment, assessments::Assessment...)
	df   = assessment.df
	runtime = assessment.runtime
	for x in assessments
		df   = vcat(df,   x.df)
		runtime = vcat(runtime, x.runtime)
	end
	Assessment(df, runtime)
end

@doc raw"""
	risk(assessment::Assessment; loss = (x, y) -> abs(x - y), average_over_parameters = true)

Computes a Monte Carlo approximation of the Bayes risk,

```math
r_{\Omega}(\hat{\boldsymbol{\theta}}(\cdot))
\approx
\frac{1}{K} \sum_{\boldsymbol{\theta} \in \vartheta} \frac{1}{J} \sum_{\boldsymbol{Z} \in \mathcal{Z}_{\boldsymbol{\theta}}} L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\boldsymbol{Z})).
```

where ``\vartheta`` denotes a set of ``K`` parameter vectors sampled from the
prior ``\Omega(\cdot)`` and, for each ``\boldsymbol{\theta} \in \vartheta``, we
have ``J`` sets of ``m`` mutually independent realisations from the model
collected in ``\mathcal{Z}_{\boldsymbol{\theta}}``.

# Keyword arguments
- `loss = (x, y) -> abs(x - y)`: a binary operator (default absolute-error loss).
- `average_over_parameters::Bool = true`: if true (default), the loss is averaged over all parameters; otherwise, the loss is averaged over each parameter separately.
- `average_over_sample_sizes::Bool = true`: if true (default), the loss is averaged over all sample sizes ``m``; otherwise, the loss is averaged over each sample size separately.
"""
function risk(assessment::Assessment;
			  loss = (x, y) -> abs(x - y),
			  average_over_parameters::Bool = true,
			  average_over_sample_sizes::Bool = true)

	df = assessment.df
	grouping_variables = [:estimator]
	if !average_over_parameters push!(grouping_variables, :parameter) end
	if !average_over_sample_sizes push!(grouping_variables, :m) end
	df = groupby(df, grouping_variables)
	df = combine(df, [:estimate, :truth] => ((x, y) -> loss.(x, y)) => :loss, ungroup = false)
	df = combine(df, :loss => mean => :risk)

	return df
end


# ---- assess() ----

"""
	assess(estimators, θ, Z; <keyword args>)

Using a collection of `estimators`, compute estimates from data `Z` simulated
based on true parameter vectors stored in `θ`.

If `Z` contains more data sets than parameter vectors, the parameter matrix will
be recycled by horizontal concatenation.

The output is of type `Assessment`; see `?Assessment` for details.

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

# Generate testing parameters
K = 100
θ = rand(p, K)

# Data for a single sample size
m = 30
Z = [rand(n, m) for _ ∈ 1:K];
assessment = assess([θ̂], θ, Z);
risk(assessment)

# Multiple data sets for each parameter vector
J = 5
Z = repeat(Z, J);
assessment = assess([θ̂], θ, Z);
risk(assessment)

# With set-level information
qₓ = 2
ϕ  = Chain(Dense(w + qₓ, w, relu), Dense(w, p));
θ̂ = DeepSet(ψ, ϕ)
x = [rand(qₓ) for _ ∈ eachindex(Z)]
assessment = assess([θ̂], θ, (Z, x));
risk(assessment)
```
"""
function assess(
	estimators, θ::P, Z;
	estimator_names::Vector{String} = ["estimator$i" for i ∈ eachindex(estimators)],
	parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(θ, 1)],
	ξ = nothing,
	use_ξ = false,
	use_gpu = true,
	verbose::Bool = true
	) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

	θ = _extractθ(θ)

	p, K = size(θ)
	m = numberreplicates(Z)
	KJ = length(m)
	@assert KJ % K == 0 "The number of data sets in `Z` must be a multiple of the number of parameter vectors in `θ`"
	J = KJ ÷ K
	if J > 1
		# verbose && @info "There are more simulated data sets than unique parameter vectors: the parameter matrix will be recycled by horizontal concatenation."
		θ = repeat(θ, outer = (1, J))
	end

	# Extract the parameter names from ξ if it was provided
	if !isnothing(ξ) && haskey(ξ, :parameter_names)
		parameter_names = ξ.parameter_names
	end

	E = length(estimators)
	@assert length(estimator_names) == E
	@assert length(parameter_names) == p

	# Use ξ if it was provided alongside only a single estimator
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
	runtime = DataFrame(estimator = String[], time = Float64[])

	θ̂ = map(eachindex(estimators)) do i

		verbose && println("	Running estimator $(estimator_names[i])...")

		if use_ξ[i]
			# pass ξ to the estimator by passing a closure to estimateinbatches().
			# This approach allows the estimator to use the gpu, and provides a
			# consistent format of the estimates regardless of whether or not
			# ξ is used. NB this doesn't work because some elements of ξ may need to
			# be subsetted when batching... So, at the moment, we cannot use
			# ξ and the gpu, unless we are willing to move the entire data
			# set and ξ to the gpu.
			# time = @elapsed θ̂ = estimateinbatches(z -> estimators[i](z, ξ), Z, use_gpu = use_gpu[i])
			time = @elapsed θ̂ = estimators[i](Z, ξ)
		else
			time = @elapsed θ̂ = estimateinbatches(estimators[i], Z, use_gpu = use_gpu[i])
		end
		θ̂ = convert(Matrix, θ̂) # sometimes estimators return vectors rather than matrices, which can mess things up

		push!(runtime, [estimator_names[i], time])
		θ̂
	end

	# Convert to DataFrame and add estimator information
	θ̂ = hcat(θ̂...)
	θ̂ = DataFrame(θ̂', parameter_names)
	θ̂[!, "estimator"] = repeat(estimator_names, inner = nrow(θ̂) ÷ E)
	θ̂[!, "m"] = repeat(m, E)
	θ̂[!, "k"] = repeat(1:K, E * J)
	θ̂[!, "j"] = repeat(repeat(1:J, inner = K), E) # NB "j" used to be "replicate"
	θ̂[!, "replicate"] = repeat(repeat(1:J, inner = K), E) # NB "replicate" included for backwards compatability; I will remove it eventually

	# Also provide the true θ for comparison with the estimates
	assessment = Assessment(_merge(DataFrame(θ', parameter_names), θ̂), runtime)

	return assessment
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
