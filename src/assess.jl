"""
	Assessment
A type for storing the output of `assess()`. The field `runtime` contains the
total time taken for each estimator. The field `estimates` is a long-form `DataFrame`
with columns:

- `parameter`: the name of the parameter
- `truth`:     the true value of the parameter
- `estimate`:  the estimated value of the parameter
- `k`:         the index of the parameter vector
- `j`:         the index of the data set (only relevant in the case that multiple data sets are associated with each parameter vector)

If the estimator is an [`IntervalEstimator`](@ref), the column `estimate` will be replaced by the columns `lower` and `upper`, containing the lower and upper bounds of the interval, respectively.

If the estimator is a [`QuantileEstimator`](@ref), there will also be a column `prob` indicating the probability level of the corresponding quantile estimate.

If the estimator is a [`PosteriorEstimator`](@ref), in addition to the fields listed above, the field `samples` stores the posterior samples as a long-form `DataFrame` with the columns `parameter`, `truth`, `k`, `j` (as given above), as well as:
- `draw`: the index of the draw within the posterior samples
- `value`: the value of the posterior sample for a given parameter and draw.

Use `merge()` to combine assessments from multiple estimators of the same type or `join()` to combine assessments from a [`PointEstimator`](@ref) and an [`IntervalEstimator`](@ref).
"""
struct Assessment
    estimates::DataFrame
    runtime::DataFrame
    samples::Union{DataFrame, Nothing}
    risk::Any
end
Assessment(estimates, runtime) = Assessment(estimates, runtime, nothing, nothing)

function merge(assessment::Assessment, assessments::Assessment...)
    estimates = assessment.estimates
    runtime = assessment.runtime
    samples = assessment.samples
    risk = assessment.risk

    # Add "estimator" column if it doesn't exist
    estimator_counter = 0
    if "estimator" ∉ names(estimates)
        estimator_counter += 1
        estimates[:, :estimator] .= "estimator$estimator_counter"
        runtime[:, :estimator] .= "estimator$estimator_counter"
        if !isnothing(samples)
            samples[:, :estimator] .= "estimator$estimator_counter"
        end
    end

    for x in assessments
        estimates2 = x.estimates
        runtime2 = x.runtime
        samples2 = x.samples
        risk2 = x.risk
        if "estimator" ∉ names(estimates2)
            estimator_counter += 1
            estimates2[:, :estimator] .= "estimator$estimator_counter"
            runtime2[:, :estimator] .= "estimator$estimator_counter"
            if !isnothing(samples2)
                samples2[:, :estimator] .= "estimator$estimator_counter"
            end
        end
        estimates = vcat(estimates, estimates2)
        runtime = vcat(runtime, runtime2)
        samples = if !isnothing(samples) && !isnothing(samples2)
            vcat(samples, samples2)
        else
            isnothing(samples) ⊻ isnothing(samples2) && @warn "Some assessments contain posterior samples and others do not; samples will be ignored in the merged result."
            nothing
        end
        risk = if !isnothing(risk) && !isnothing(risk2)
            vcat(risk, risk2)
        else
            isnothing(risk) ⊻ isnothing(risk2) && @warn "Some assessments contain a precomputed risk and others do not; risk will be ignored in the merged result."
            nothing
        end
    end
    Assessment(estimates, runtime, samples, risk)
end

function join(assessment::Assessment, assessments::Assessment...)
    estimates = assessment.estimates
    runtime = assessment.runtime
    estimator_flag = "estimator" ∈ names(estimates)
    if estimator_flag
        select!(estimates, Not(:estimator))
        select!(runtime, Not(:estimator))
    end
    for x in assessments
        estimates2 = x.estimates
        runtime2 = x.runtime
        if estimator_flag
            select!(estimates2, Not(:estimator))
            select!(runtime2, Not(:estimator))
        end
        estimates = innerjoin(estimates, estimates2, on = [:m, :k, :j, :parameter, :truth])
        runtime = runtime .+ runtime2
    end
    Assessment(estimates, runtime, nothing, nothing)
end

# ---- assess() methods and internal helper functions ----

"""
	assess(estimator, θ, Z; ...)
	assess(estimators::Vector, θ, Z; ...)
Assesses an `estimator` (or a collection of `estimators`) based on true parameters `θ` and corresponding simulated data `Z`.

The parameters `θ` should be given as a ``d`` × ``K`` matrix, where ``d`` is the parameter dimension and ``K`` is the number of sampled parameter vectors.

When `Z` contains more simulated data sets than the number ``K`` of sampled parameter vectors, `θ` will be recycled via horizontal concatenation: `θ = repeat(θ, outer = (1, J))`, where `J = numobs(Z) ÷ K` is the number of simulated data sets for each parameter vector. This allows assessment of the estimator's sampling distribution under fixed parameters.

The return value is of type [`Assessment`](@ref).

# Keyword arguments
- `estimator_name::String` (or `estimator_names::Vector{String}` for multiple estimators): name(s) of the estimator(s) (sensible defaults provided).
- `parameter_names::Vector{String}`: names of the parameters (sensible default provided).
- `use_gpu = true`: `Bool` or `Vector{Bool}` with length equal to the number of estimators.
- `probs = nothing` (applicable only to [`PointEstimator`](@ref)): probability levels taking values between 0 and 1. By default, no bootstrap uncertainty quantification is done; if `probs` is provided, it must be a two-element vector specifying the lower and upper probability levels for non-parametric bootstrap intervals (note that parametric bootstrap is not currently supported with `assess()`).
- `B::Integer = 400` (applicable only to [`PointEstimator`](@ref)): number of bootstrap samples.
- `pointsummary::Function = mean` (applicable only to estimators that yield posterior samples): a function that summarises a vector of posterior samples into a single point estimate for each marginal; any function mapping a vector to a scalar is valid (e.g., `median` for the posterior median).
- `N::Integer = 1000` (applicable only to estimators that yield posterior samples): number of posterior samples drawn for each data set.
- `kwargs...` (applicable only to estimators that yield posterior samples): additional keyword arguments passed to [`sampleposterior`](@ref).
"""
function assess end

# Point estimates
function assess(
    estimator, θ::P, Z;
    parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(θ, 1)],
    estimator_name::Union{Nothing, String} = nothing,
    estimator_names::Union{Nothing, String} = nothing,
    use_gpu::Bool = true,
    probs = nothing,
    B::Integer = 400,
    loss = Flux.Losses.mae,   # TODO this will be simplified if we add loss to the estimator object
    ξ = nothing, xi = nothing # deprecated since it isn't typically needed when assessing NeuralEstimators (a collection of objects passed to estimator)
) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

    # Check duplicated arguments that are needed so that the R interface uses ASCII characters only
    @assert isnothing(ξ) || isnothing(xi) "Only one of `ξ` or `xi` should be provided"
    if !isnothing(xi)
        ξ = xi
    end

    # Set up
    θ, parameter_names, d, K, J, m = _assess_setup(θ, Z, parameter_names)

    # Obtain point estimates
    if !isnothing(ξ)
        runtime = @elapsed estimates = estimator(Z, ξ) # note that the gpu is never used in this case
    else
        runtime = @elapsed estimates = estimate(estimator, Z, use_gpu = use_gpu)
    end
    estimates = convert(Matrix, estimates) # convert to Matrix in case estimator returns a different format (e.g., adjoint vector)
    runtime = DataFrame(runtime = runtime)

    # Compute the empirical risk
    loss = _loss(estimator, loss)
    empirical_risk = loss(estimates, θ)

    # Convert true and estimated parameter to DataFrame, then merge
    estimates = _estimates_to_df(estimates, parameter_names, K, J, m)
    θ_df = _truth_to_df(θ, parameter_names, nrow(estimates))
    estimates = _attach_truth(θ_df, estimates)

    if !isnothing(probs)
        @assert length(probs) == 2
        @assert !(Z isa Tuple) "bootstrap() is not currently set up for dealing with set-level information; please contact the package maintainer"
        bs = bootstrap.(Ref(estimator), Z, use_gpu = use_gpu, B = B)
        # compute bootstrap intervals and convert to same format returned by IntervalEstimator
        intervals = stackarrays(vec.(interval.(bs, probs = probs)), merge = false)
        # convert to dataframe and merge
        estimate_names = repeat(parameter_names, outer = 2) .* repeat(["_lower", "_upper"], inner = d)
        intervals = DataFrame(intervals', estimate_names)
        intervals[!, "m"] = m
        intervals[!, "k"] = repeat(1:K, J)
        intervals[!, "j"] = repeat(1:J, inner = K)
        intervals = _merge_interval(θ_df, intervals)
        estimates[:, "lower"] = intervals[:, "lower"]
        estimates[:, "upper"] = intervals[:, "upper"]
        estimates[:, "α"] .= 1 - (probs[2] - probs[1])
    end

    # Add estimator name if it was provided
    estimator_name = _resolve_estimator_name(estimator_name, estimator_names)
    _add_estimator_name!(estimates, runtime, estimator_name)

    return Assessment(estimates, runtime, nothing, empirical_risk)
end

# Posterior sampling 
function assess(
    estimator::Union{PosteriorEstimator, RatioEstimator}, θ::P, Z;
    parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(θ, 1)],
    estimator_name::Union{Nothing, String} = nothing,
    estimator_names::Union{Nothing, String} = nothing,
    N::Integer = 1000,
    pointsummary::Function = mean,
    kwargs... #Document these kwargs...
) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

    # Set up
    θ, parameter_names, d, K, J, m = _assess_setup(θ, Z, parameter_names)

    # Posterior samples 
    runtime = @elapsed samples = sampleposterior(estimator, Z, N; kwargs...)
    runtime = DataFrame(runtime = runtime)

    # Empirical risk
    empirical_risk = _computerisk(estimator, θ, Z)

    # Obtain point estimates 
    estimates = reduce(hcat, map.(pointsummary, eachrow.(samples)))

    # Convert true and estimated parameter to DataFrame, then merge
    estimates = _estimates_to_df(estimates, parameter_names, K, J, m)
    θ_df = _truth_to_df(θ, parameter_names, nrow(estimates))
    estimates = _attach_truth(θ_df, estimates)

    # Convert posterior samples to long form DataFrame 
    sample_dfs = Vector{DataFrame}(undef, length(samples))
    for (idx, S) in enumerate(samples)
        d, N = size(S)

        df_s = DataFrame(
            parameter = repeat(parameter_names, inner = N),
            truth = repeat(θ[:, idx], inner = N),
            draw = repeat(1:N, outer = d),
            value = vec(S')
        )
        df_s[!, "k"] .= ((idx - 1) % K) + 1
        df_s[!, "j"] .= ((idx - 1) ÷ K) + 1

        sample_dfs[idx] = df_s
    end
    samples_df = vcat(sample_dfs...)

    # Add estimator name if it was provided
    estimator_name = _resolve_estimator_name(estimator_name, estimator_names)
    _add_estimator_name!(estimates, runtime, estimator_name)
    if !isnothing(estimator_name)
        samples_df[!, "estimator"] .= estimator_name
    end

    return Assessment(estimates, runtime, samples_df, empirical_risk)
end

"""
    _assess_setup(θ, Z, parameter_names)

Shared pre-processing for all `assess()` methods. Validates inputs, resolves
`parameter_names` from a `NamedMatrix` where available, computes the replication
factor `J`, expands `θ` to match `Z` when `J > 1`, and builds the per-dataset
sample-size vector `m`.

Returns `(θ, parameter_names, d, K, J, m)` where:
- `θ`               — expanded parameter matrix (`d × KJ`)
- `parameter_names` — resolved names, length `d`  
- `d`               — parameter dimension
- `K`               — number of distinct parameter vectors
- `J`               — number of data sets per parameter vector
- `m`               — `Vector` of length `KJ` giving the number of replicates in
                      each element of `Z` (always `1` when `Z` is not a vector)
"""
function _assess_setup(θ::P, Z, parameter_names::Vector{String}) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

    # Unwrap ParameterConfigurations into plain matrix
    θ = _extractθ(θ)
    d, K = size(θ)

    # NamedMatrix rows carry authoritative parameter names; prefer them over the default
    if θ isa NamedMatrix
        parameter_names = names(θ, 1)
    end
    @assert length(parameter_names) == d "length(parameter_names) ($(length(parameter_names))) must equal the parameter dimension d ($d)"
    θ = convert(Matrix, θ)

    # Validate and compute the replication factor J
    KJ = numobs(Z)
    @assert KJ % K == 0 "The number of data sets in `Z` ($KJ) must be a multiple of the number of parameter vectors in `θ` ($K)"
    J = KJ ÷ K

    # Expand θ so each column aligns with the corresponding dataset in Z
    if J > 1
        θ = repeat(θ, outer = (1, J))
    end

    # Number of replicates per dataset element (only meaningful when Z is a Vector)
    m = Z isa AbstractVector ? numberreplicates(Z) : fill(1, KJ)

    return θ, parameter_names, d, K, J, m
end

_resolve_estimator_name(name, names) = isnothing(names) ? name : names

function _computerisk(estimator, θ, Z; use_gpu = true, batchsize = 32)
    loss = _loss(estimator)
    device = _checkgpu(use_gpu, verbose = false)
    estimator = device(estimator)
    dataset = _dataloader(estimator, Z, θ, batchsize)
    _risk(estimator, loss, dataset, device)
end

function _estimates_to_df(estimates, estimate_names, K, J, m)
    df = DataFrame(estimates', estimate_names)
    df[!, "m"] = m
    df[!, "k"] = repeat(1:K, J)
    df[!, "j"] = repeat(1:J, inner = K)
    return df
end

function _truth_to_df(θ, parameter_names, n_rows)
    df = DataFrame(θ', parameter_names)
    df = repeat(df, outer = n_rows ÷ nrow(df))
    df = stack(df, variable_name = :parameter, value_name = :truth)
    return df
end

function _attach_truth(θ, estimates)
    non_measure_vars = [:m, :k, :j]
    if "estimator" ∈ names(estimates)
        push!(non_measure_vars, :estimator)
    end

    df = stack(estimates, Not(non_measure_vars), variable_name = :parameter, value_name = :estimate)
    df[!, :truth] = θ[:, :truth]

    return df
end

# NB also used in assess(::QuantileEstimator)
function _merge_interval(θ, interval_estimates)
    non_measure_vars = [:m, :k, :j]
    if "estimator" ∈ names(interval_estimates)
        push!(non_measure_vars, :estimator)
    end

    # Stack lower bounds to long form
    df = select(interval_estimates, Not(endswith.("_upper").(names(interval_estimates))))
    df = stack(df, Not(non_measure_vars), variable_name = :parameter, value_name = :lower)
    df.parameter = replace.(df.parameter, r"_lower$" => "")

    # Extract upper bounds in matching order and attach directly
    upper = select(interval_estimates, [non_measure_vars..., Symbol.(filter(endswith("_upper"), names(interval_estimates)))...])
    upper = stack(upper, Not(non_measure_vars), variable_name = :parameter, value_name = :upper)
    df[!, :upper] = upper[:, :upper]
    df[!, :truth] = θ[:, :truth]

    return df
end

function _add_estimator_name!(df, runtime, name)
    if !isnothing(name)
        df[!, "estimator"] .= name
        runtime[!, "estimator"] .= name
    end
end

# Wrapper for assessing multiple estimators at once
function assess(
    estimators::Union{AbstractVector, Tuple}, θ::P, Z;
    estimator_names::Union{Nothing, Vector{String}} = nothing,
    use_gpu = true,
    verbose::Bool = true,
    ξ = nothing, xi = nothing,     # Deprecated
    use_xi = false, use_ξ = false, # Deprecated: a `Bool` or a collection of `Bool` objects with length equal to the number of estimators, specifying whether or not the estimator uses `ξ`
    kwargs...
) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    num_estimators = length(estimators)
    if isnothing(estimator_names)
        estimator_names = ["estimator$i" for i ∈ eachindex(estimators)]
    end
    @assert length(estimator_names) == num_estimators
    if use_xi != false
        use_ξ = use_xi
    end  # NB here we check "use_xi != false" since use_xi might be a vector of bools, so it can't be used directly in the if-statement
    @assert eltype(use_ξ) == Bool
    @assert eltype(use_gpu) == Bool
    if use_ξ isa Bool
        use_ξ = repeat([use_ξ], num_estimators)
    end
    if use_gpu isa Bool
        use_gpu = repeat([use_gpu], num_estimators)
    end
    @assert length(use_ξ) == num_estimators
    @assert length(use_gpu) == num_estimators

    # Assess the estimators
    assessments = map(1:num_estimators) do i
        verbose && println("	Running $(estimator_names[i])...")
        if use_ξ[i]
            assess(estimators[i], θ, Z, ξ = ξ; use_gpu = use_gpu[i], estimator_name = estimator_names[i], kwargs...)
        else
            assess(estimators[i], θ, Z; use_gpu = use_gpu[i], estimator_name = estimator_names[i], kwargs...)
        end
    end

    # Combine the assessment objects 
    if num_estimators == 2 && any(isa.(estimators, Union{IntervalEstimator, Ensemble{<:IntervalEstimator}})) && any(isa.(estimators, Union{PointEstimator, Ensemble{<:PointEstimator}}))
        assessment = join(assessments...)
    elseif all(assessment -> names(assessment.estimates) == names(assessments[1].estimates), assessments)
        assessment = merge(assessments...)
    else
        assessment = assessments
    end

    return assessment
end

# ---- Diagnostic functions acting on Assessment objects ----

function _groupedloss(df::DataFrame, loss;
    average_over_parameters::Bool = false,
    average_over_sample_sizes::Bool = true)
    grouping_variables = "estimator" ∈ names(df) ? [:estimator] : []
    if !average_over_parameters
        push!(grouping_variables, :parameter)
    end
    if !average_over_sample_sizes
        push!(grouping_variables, :m)
    end
    df = groupby(df, grouping_variables)
    df = combine(df, [:estimate, :truth] => ((x, y) -> loss.(x, y)) => :loss, ungroup = false)
    df = combine(df, :loss => mean => :loss)

    return df
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

If the `Assessment` object corresponds to an estimator with a self-defined loss (e.g., [`PosteriorEstimator`](@ref)), the precomputed risk is returned directly. Otherwise, the risk is computed from the estimates and true parameters using the provided `loss` function.

# Keyword arguments
- `loss = (x, y) -> abs(x - y)`: a binary operator defining the loss function (default: absolute-error loss)
- `average_over_parameters::Bool = false`: if `true`, the loss is averaged over all parameters; otherwise (default), it is computed separately for each parameter.
"""
function risk(assessment::Assessment; loss = nothing, args...)
    isnothing(loss) ? assessment.risk : risk(assessment.estimates; loss = loss, args...)
end

function risk(df::DataFrame;
    loss = (x, y) -> abs(x - y),
    average_over_parameters::Bool = false,
    average_over_sample_sizes::Bool = true)
    df = _groupedloss(df, loss; average_over_parameters, average_over_sample_sizes)
    rename!(df, :loss => :risk)

    return df
end

@doc raw"""
	bias(assessment::Assessment; average_over_parameters = false)

Computes a Monte Carlo approximation of an estimator's bias,

```math
{\textrm{bias}}(\hat{\boldsymbol{\theta}}(\cdot))
\approx
\frac{1}{K} \sum_{k=1}^K \{\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(k)}) - \boldsymbol{\theta}^{(k)}\},
```

where ``\{\boldsymbol{\theta}^{(k)} : k = 1, \dots, K\}`` denotes a set of ``K`` parameter vectors sampled from the
prior and, for each ``k``, data ``\boldsymbol{Z}^{(k)}`` are simulated from the statistical model conditional on ``\boldsymbol{\theta}^{(k)}``.
"""
bias(assessment::Assessment; args...) = bias(assessment.estimates; args...)

function bias(df::DataFrame;
    average_over_parameters::Bool = false,
    average_over_sample_sizes::Bool = true)
    df = _groupedloss(df, (x, y) -> x - y; average_over_parameters, average_over_sample_sizes)
    rename!(df, :loss => :bias)

    return df
end

@doc raw"""
	rmse(assessment::Assessment; average_over_parameters = false)

Computes a Monte Carlo approximation of an estimator's root-mean-squared error,

```math
{\textrm{rmse}}(\hat{\boldsymbol{\theta}}(\cdot))
\approx
\sqrt{\frac{1}{K} \sum_{k=1}^K \{\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(k)}) - \boldsymbol{\theta}^{(k)}\}^2},
```

where ``\{\boldsymbol{\theta}^{(k)} : k = 1, \dots, K\}`` denotes a set of ``K`` parameter vectors sampled from the
prior and, for each ``k``, data ``\boldsymbol{Z}^{(k)}`` are simulated from the statistical model conditional on ``\boldsymbol{\theta}^{(k)}``.
"""
rmse(assessment::Assessment; args...) = rmse(assessment.estimates; args...)

function rmse(df::DataFrame; average_over_parameters::Bool = false,
    average_over_sample_sizes::Bool = true)
    df = _groupedloss(df, (x, y) -> (x - y)^2; average_over_parameters, average_over_sample_sizes)
    df[:, :loss] = sqrt.(df[:, :loss])
    rename!(df, :loss => :rmse)

    return df
end

"""
	coverage(assessment::Assessment; ...)

Computes a Monte Carlo approximation of an interval estimator's expected coverage,
as defined in [Hermans et al. (2022, Definition 2.1)](https://arxiv.org/abs/2110.06581),
and the proportion of parameters below and above the lower and upper bounds, respectively.

# Keyword arguments
- `average_over_parameters::Bool = false`: if true, the coverage is averaged over all parameters; otherwise (default), it is computed over each parameter separately.
"""
function coverage(assessment::Assessment;
    average_over_parameters::Bool = false,
    average_over_sample_sizes::Bool = true)
    df = assessment.estimates

    @assert all(["lower", "truth", "upper"] .∈ Ref(names(df))) "The assessment object should contain the columns `lower`, `upper`, and `truth`"

    grouping_variables = "estimator" ∈ names(df) ? [:estimator] : []
    if !average_over_parameters
        push!(grouping_variables, :parameter)
    end
    if !average_over_sample_sizes
        push!(grouping_variables, :m)
    end
    df = groupby(df, grouping_variables)
    df = combine(df,
        [:lower, :truth, :upper] => ((x, y, z) -> x .<= y .< z) => :within,
        [:lower, :truth] => ((x, y) -> y .< x) => :below,
        [:truth, :upper] => ((y, z) -> y .> z) => :above,
        ungroup = false)
    df = combine(df,
        :within => mean => :coverage,
        :below => mean => :below_lower,
        :above => mean => :above_upper)

    return df
end

function empiricalprob(assessment::Assessment;
    average_over_parameters::Bool = false,
    average_over_sample_sizes::Bool = true)
    df = assessment.estimates

    @assert all(["prob", "estimate", "truth"] .∈ Ref(names(df)))

    grouping_variables = [:prob]
    if "estimator" ∈ names(df)
        push!(grouping_variables, :estimator)
    end
    if !average_over_parameters
        push!(grouping_variables, :parameter)
    end
    if !average_over_sample_sizes
        push!(grouping_variables, :m)
    end
    df = groupby(df, grouping_variables)
    df = combine(df,
        [:estimate, :truth] => ((x, y) -> x .> y) => :below,
        ungroup = false)
    df = combine(df, :below => mean => :empirical_prob)

    return df
end

function intervalscore(assessment::Assessment;
    average_over_parameters::Bool = false,
    average_over_sample_sizes::Bool = true)
    df = assessment.estimates

    @assert all(["lower", "truth", "upper"] .∈ Ref(names(df))) "The assessment object should contain the columns `lower`, `upper`, and `truth`"
    @assert "α" ∈ names(df) "The assessment object should contain the column `α` specifying the nominal coverage of the interval"
    α = df[1, :α]

    grouping_variables = "estimator" ∈ names(df) ? [:estimator] : []
    if !average_over_parameters
        push!(grouping_variables, :parameter)
    end
    if !average_over_sample_sizes
        push!(grouping_variables, :m)
    end
    truth = df[:, :truth]
    lower = df[:, :lower]
    upper = df[:, :upper]
    df[:, :interval_score] = (upper - lower) + (2/α) * (lower - truth) * (truth < lower) + (2/α) * (truth - upper) * (truth > upper)
    df = groupby(df, grouping_variables)
    df = combine(df, :interval_score => mean => :interval_score)

    return df
end
