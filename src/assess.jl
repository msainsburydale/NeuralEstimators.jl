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

Note that if `estimator` is an `IntervalEstimator`, the column `estimate` will be replaced by the columns `lower` and `upper`, containing the lower and upper bounds of the interval, respectively.

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
	risk(assessment::Assessment; ...)

Computes a Monte Carlo approximation of an estimator's Bayes risk,

```math
r(\hat{\boldsymbol{\theta}}(\cdot))
\approx
\frac{1}{K} \sum_{k=1}^K L(\boldsymbol{\theta}^{(k)}, \hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(k)})),
```

where ``\{\boldsymbol{\theta}^{(k)} : k = 1, \dots, K\}`` denotes a set of ``K`` parameter vectors sampled from the
prior and, for each ``k``, data ``\boldsymbol{Z}^{(k)}`` are simulated from the statistical model conditional on ``\boldsymbol{\theta}^{(k)}``.

# Keyword arguments
- `loss = (x, y) -> abs(x - y)`: a binary operator defining the loss function (default absolute-error loss).
- `average_over_parameters::Bool = true`: if true (default), the loss is averaged over all parameters; otherwise, the loss is averaged over each parameter separately.
- `average_over_sample_sizes::Bool = true`: if true (default), the loss is averaged over all sample sizes ``m``; otherwise, the loss is averaged over each sample size separately.
"""
risk(assessment::Assessment; args...) = risk(assessment.df; args...)

function risk(df::DataFrame;
			  loss = (x, y) -> abs(x - y),
			  average_over_parameters::Bool = true,
			  average_over_sample_sizes::Bool = true)

	grouping_variables = [:estimator]
	if !average_over_parameters push!(grouping_variables, :parameter) end
	if !average_over_sample_sizes push!(grouping_variables, :m) end
	df = groupby(df, grouping_variables)
	df = combine(df, [:estimate, :truth] => ((x, y) -> loss.(x, y)) => :loss, ungroup = false)
	df = combine(df, :loss => mean => :risk)

	return df
end

@doc raw"""
	bias(assessment::Assessment; ...)

Computes a Monte Carlo approximation of an estimator's bias,

```math
{\rm{bias}}(\hat{\boldsymbol{\theta}}(\cdot))
\approx
\frac{1}{K} \sum_{k=1}^K \hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(k)}) - \boldsymbol{\theta}^{(k)},
```

where ``\{\boldsymbol{\theta}^{(k)} : k = 1, \dots, K\}`` denotes a set of ``K`` parameter vectors sampled from the
prior and, for each ``k``, data ``\boldsymbol{Z}^{(k)}`` are simulated from the statistical model conditional on ``\boldsymbol{\theta}^{(k)}``.

This function inherits the keyword arguments of [`risk`](@ref) (excluding the argument `loss`).
"""
bias(assessment::Assessment; args...) = bias(assessment.df; args...)

function bias(df::DataFrame; args...)

    df = risk(df; loss = (x, y) -> x - y, args...)

	rename!(df, :risk => :bias)

	return df
end

@doc raw"""
	rmse(assessment::Assessment; ...)

Computes a Monte Carlo approximation of an estimator's root-mean-squared error,

```math
{\rm{rmse}}(\hat{\boldsymbol{\theta}}(\cdot))
\approx
\sqrt{\frac{1}{K} \sum_{k=1}^K (\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(k)}) - \boldsymbol{\theta}^{(k)})^2},
```

where ``\{\boldsymbol{\theta}^{(k)} : k = 1, \dots, K\}`` denotes a set of ``K`` parameter vectors sampled from the
prior and, for each ``k``, data ``\boldsymbol{Z}^{(k)}`` are simulated from the statistical model conditional on ``\boldsymbol{\theta}^{(k)}``.

This function inherits the keyword arguments of [`risk`](@ref) (excluding the argument `loss`).
"""
rmse(assessment::Assessment; args...) = rmse(assessment.df; args...)

function rmse(df::DataFrame; args...)

	df = risk(df; loss = (x, y) -> (x - y)^2, args...)

    df[:, :risk] = sqrt.(df[:, :risk])
    rename!(df, :risk => :rmse)

	return df
end

"""
	coverage(assessment::Assessment; ...)

Computes a Monte Carlo approximation of an interval estimator's expected coverage.

# Keyword arguments
- `average_over_parameters::Bool = true`: if true, the coverage is averaged over all parameters; otherwise (default), the coverage is averaged over each parameter separately.
- `average_over_sample_sizes::Bool = true`: if true (default), the coverage is averaged over all sample sizes ``m``; otherwise, the coverage is averaged over each sample size separately.
"""
function coverage(assessment::Assessment;
				  average_over_parameters::Bool = false,
				  average_over_sample_sizes::Bool = true)

	df = assessment.df

	@assert all(["lower", "truth", "upper"] .∈ Ref(names(df))) "The assessment object should be derived from an IntervalEstimator, so that the dataframe contains the columns `lower`, `upper`, and `truth`"

	grouping_variables = [:estimator]
	if !average_over_parameters push!(grouping_variables, :parameter) end
	if !average_over_sample_sizes push!(grouping_variables, :m) end
	df = groupby(df, grouping_variables)
	df = combine(df, [:lower, :truth, :upper] => ((x, y, z) -> x .<= y .< z) => :within, ungroup = false)
	df = combine(df, :within => mean => :coverage)

	return df
end


# ---- assess() ----

"""
	assess(estimators, θ, Z)

Using a collection of `estimators`, compute estimates from data `Z` simulated
based on true parameter vectors stored in `θ`.

The data `Z` should be a `Vector`, with each element corresponding to a single
simulated data set. If `Z` contains more data sets than parameter vectors, the
parameter matrix `θ` will be recycled by horizontal concatenation via the call
`θ = repeat(θ, outer = (1, J))` where `J = length(Z) ÷ K` is the number of
simulated data sets and `K = size(θ, 2)` is the number of parameter vectors.

The output is of type `Assessment`; see `?Assessment` for details.

# Keyword arguments
- `estimator_names::Vector{String}`: names of the estimators (sensible defaults provided).
- `parameter_names::Vector{String}`: names of the parameters (sensible defaults provided). If `ξ` is provided with a field `parameter_names`, those names will be used.
- `ξ = nothing`: an arbitrary collection of objects that are fixed (e.g., distance matrices). Can also be provided as `xi`.
- `use_ξ = false`: a `Bool` or a collection of `Bool` objects with length equal to the number of estimators. Specifies whether or not the estimator uses `ξ`: if it does, the estimator will be applied as `estimator(Z, ξ)`. This argument is useful when multiple `estimators` are provided, only some of which need `ξ`; hence, if only one estimator is provided and `ξ` is not `nothing`, `use_ξ` is automatically set to `true`. Can also be provided as `use_xi`.
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
θ = rand(Float32, p, K)

# Data for a single sample size
m = 30
Z = [rand(Float32, n, m) for _ ∈ 1:K];
assessment = assess(θ̂, θ, Z);
risk(assessment)

# Multiple data sets for each parameter vector
J = 5
Z = repeat(Z, J);
assessment = assess(θ̂, θ, Z);
risk(assessment)

# With set-level information
qₓ = 2
ϕ  = Chain(Dense(w + qₓ, w, relu), Dense(w, p));
θ̂ = DeepSet(ψ, ϕ)
x = [rand(qₓ) for _ ∈ eachindex(Z)]
assessment = assess(θ̂, θ, (Z, x));
risk(assessment)
```
"""
function assess(
	estimators, θ::P, Z;
	estimator_names::Union{Nothing, Vector{String}} = nothing,
	parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(θ, 1)],
	ξ  = nothing, use_ξ  = false,
	xi = nothing, use_xi = false,
	use_gpu = true,
	verbose::Bool = true
	) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

	# Check duplicated arguments that are needed so that the R interface uses ASCII characters only
	@assert isnothing(ξ) || isnothing(xi) "Only one of `ξ` or `xi` should be provided"
	if !isnothing(xi) ξ = xi end
	if use_xi != false use_ξ = use_xi end  # note that here we check "use_xi != false" since use_xi might be a vector of bools, so it can't be used directly on the if statement

	θ = _extractθ(θ)

	p, K = size(θ)
	m = numberreplicates(Z)
	if !(typeof(m) <: Vector{Int}) # indicates that a vector of vectors has been given
		# verbose && @warn "The data `Z` should be a a vector, with each element of the vector corresponding to a single simulated data set... attempted to convert `Z` to the correct format."
		Z = vcat(Z...) # convert to a single vector
		m = numberreplicates(Z)
	end
	KJ = length(m) # this should be the same as length(Z)... will leave it as is to avoid breaking backwards compatability
	@assert KJ % K == 0 "The number of data sets in `Z` must be a multiple of the number of parameter vectors in `θ`"
	J = KJ ÷ K
	if J > 1
		# verbose && @info "There are more simulated data sets than unique parameter vectors: the parameter matrix will be recycled by horizontal concatenation."
		θ = repeat(θ, outer = (1, J))
	end

	# Extract the parameter names from ξ or θ, if provided
	if !isnothing(ξ) && haskey(ξ, :parameter_names)
		parameter_names = ξ.parameter_names
	elseif typeof(θ) <: NamedMatrix
		parameter_names = names(θ, 1)
	end

	if !(typeof(estimators) <: Vector) estimators = [estimators] end
	if isnothing(estimator_names) estimator_names = ["estimator$i" for i ∈ eachindex(estimators)] end
	E = length(estimators)
	@assert length(estimator_names) == E
	@assert length(parameter_names) == p

	if any(typeof.(estimators) .<: IntervalEstimator)
		@assert all(typeof.(estimators) .<: IntervalEstimator) "IntervalEstimators can only be assessed alongside other IntervalEstimators"
		estimate_names = repeat(parameter_names, outer = 2) .* repeat(["_lower", "_upper"], inner = 2)
	else
		estimate_names = parameter_names
	end

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
	θ̂ = DataFrame(θ̂', estimate_names)
	θ̂[!, "estimator"] = repeat(estimator_names, inner = nrow(θ̂) ÷ E)
	θ̂[!, "m"] = repeat(m, E)
	θ̂[!, "k"] = repeat(1:K, E * J)
	θ̂[!, "j"] = repeat(repeat(1:J, inner = K), E) # NB "j" used to be "replicate"
	θ̂[!, "replicate"] = repeat(repeat(1:J, inner = K), E) # NB same as "j"

	# Dataframe containing the true parameters
	θ = convert(Matrix, θ) # this shouldn't really be necessary, but for some reason plot(::IntervalEstimator) doesn't like it when θ here is stored as a NamedArray
	θ = DataFrame(θ', parameter_names)
	# Replicate θ to match the number of rows in θ̂. Note that the parameter
	# configuration, k, is the fastest running variable in θ̂, so we repeat θ
	# in an outer fashion.
	θ = repeat(θ, outer = nrow(θ̂) ÷ nrow(θ))
	# Transform θ to long form
	θ = stack(θ, variable_name = :parameter, value_name = :truth)

	# Merge true parameters and estimates
	if any(typeof.(estimators) .<: IntervalEstimator)
		df = _mergeIntervalEstimator(θ, θ̂)
	else
		df = _merge(θ, θ̂)
	end

	return Assessment(df, runtime)
end

function _mergeIntervalEstimator(θ, θ̂)


	# Convert θ̂ into appropriate form
	# Lower bounds:
	df = copy(θ̂)
	select!(df, Not(contains.(names(df), "upper")))
	df = stack(df, Not([:estimator, :m, :k, :j, :replicate]), variable_name = :parameter, value_name = :lower)
	df.parameter = replace.(df.parameter, r"_lower$"=>"")
	df1 = df
	# Upper bounds:
	df = copy(θ̂)
	select!(df, Not(contains.(names(df), "lower")))
	df = stack(df, Not([:estimator, :m, :k, :j, :replicate]), variable_name = :parameter, value_name = :upper)
	df.parameter = replace.(df.parameter, r"_upper$"=>"")
	df2 = df
	# Join lower and upper bounds:
	θ̂ = innerjoin(df1, df2, on = [:estimator, :m, :k, :j, :replicate, :parameter])

	# Merge θ and θ̂ by adding true parameters to θ̂
	θ̂[!, :truth] = θ[:, :truth]

	return θ̂
end

function _merge(θ, θ̂)

	# Transform θ̂ to long form
	θ̂ = stack(θ̂, Not([:estimator, :m, :k, :j, :replicate]), variable_name = :parameter, value_name = :estimate)

	# Merge θ and θ̂ by adding true parameters to θ̂
	θ̂[!, :truth] = θ[:, :truth]

	return θ̂
end


# NB might want to include both point estimates and interval estimates in the same plot (e.g., as a result of joining two assessment.df objects)... could simply add bounds if "upper" and "lower" are detected
function plot(assessment::Assessment)

  df = assessment.df
  num_estimators = length(unique(df.estimator))

  if all(["lower", "upper"] .∈ Ref(names(df)))
	  # Need line from (truth, lower) to (truth, upper). To do this, we need to
	  # merge lower and upper into a single column and then group by k.
	  df = stack(df, [:lower, :upper], variable_name = :bound, value_name = :interval)
	  figure =  data(df) * mapping(:truth, :interval, group = :k => nonnumeric, layout = :parameter) * visual(Lines, color = :black)
	  figure += data(df) * mapping(:truth, :interval, layout = :parameter) * visual(Scatter, color = :black, marker = '⎯')
  elseif num_estimators > 1
	  colors = [unique(df.estimator)[i] => ColorSchemes.Set1_4.colors[i] for i ∈ 1:num_estimators]
	  figure = data(df) * mapping(:truth, :estimate, color = :estimator, layout = :parameter) * visual(palettes=(color=colors,), alpha = 0.5)
  else
	  figure = data(df) * mapping(:truth, :estimate, layout = :parameter) * visual(color = :black, alpha = 0.5)
  end

  figure += mapping([0], [1]) * visual(ABLines, color=:red, linestyle=:dash)
  figure = draw(figure, facet=(; linkxaxes=:none, linkyaxes=:none)) #, axis=(; aspect=1)) # NB couldn't fix the aspect ratio without messing up the positioning of the titles
  return figure
end
# figure = plot(assessment)
# save("docs/src/assets/figures/univariate_point.png", figure, px_per_unit = 3, size = (600, 300))
# save("docs/src/assets/figures/univariate_uq.png", figure, px_per_unit = 3, size = (600, 300))
