"""
	Assessment(df::DataFrame, runtime::DataFrame)
A type for storing the output of `assess()`. The field `runtime` contains the
total time taken for each estimator. The field `df` is a long-form `DataFrame`
with columns:

- `estimator`: the name of the estimator
- `parameter`: the name of the parameter
- `truth`:     the true value of the parameter
- `estimate`:  the estimated value of the parameter
- `m`:         the sample size (number of exchangeable replicates) for the given data set
- `k`:         the index of the parameter vector
- `j`:         the index of the data set (in the case that multiple data sets are associated with each parameter vector)

If the estimator is an [`IntervalEstimator`](@ref), the column `estimate` will be replaced by the columns `lower` and `upper`, containing the lower and upper bounds of the interval, respectively.

If the estimator is a [`QuantileEstimator`](@ref), there will also be a column `prob` indicating the probability level of the corresponding quantile estimate.

Use `merge()` to combine assessments from multiple estimators of the same type or `join()` to combine assessments from a [`PointEstimator`](@ref) and an [`IntervalEstimator`](@ref).
"""
struct Assessment
    df::DataFrame
    runtime::DataFrame
end

function merge(assessment::Assessment, assessments::Assessment...)
    df = assessment.df
    runtime = assessment.runtime
    # Add "estimator" column if it doesn't exist
    estimator_counter = 0
    if "estimator" ∉ names(df)
        estimator_counter += 1
        df[:, :estimator] .= "estimator$estimator_counter"
        runtime[:, :estimator] .= "estimator$estimator_counter"
    end
    for x in assessments
        df2 = x.df
        runtime2 = x.runtime
        # Add "estimator" column if it doesn't exist
        if "estimator" ∉ names(df2)
            estimator_counter += 1
            df2[:, :estimator] .= "estimator$estimator_counter"
            runtime2[:, :estimator] .= "estimator$estimator_counter"
        end
        df = vcat(df, df2)
        runtime = vcat(runtime, runtime2)
    end
    Assessment(df, runtime)
end

function join(assessment::Assessment, assessments::Assessment...)
    df = assessment.df
    runtime = assessment.runtime
    estimator_flag = "estimator" ∈ names(df)
    if estimator_flag
        select!(df, Not(:estimator))
        select!(runtime, Not(:estimator))
    end
    for x in assessments
        df2 = x.df
        runtime2 = x.runtime
        if estimator_flag
            select!(df2, Not(:estimator))
            select!(runtime2, Not(:estimator))
        end
        df = innerjoin(df, df2, on = [:m, :k, :j, :parameter, :truth])
        runtime = runtime .+ runtime2
    end
    Assessment(df, runtime)
end

# - `probs` (applicable only to [`PointEstimator`](@ref) and [`QuantileEstimatorContinuous`](@ref)): probability levels taking values between 0 and 1. For a `PointEstimator`, the default is `nothing` (no bootstrap uncertainty quantification); if provided, it must be a two-element vector specifying the lower and upper probability levels for non-parametric bootstrap intervals. For a `QuantileEstimatorContinuous`, `probs` defines the probability levels at which the estimator is evaluated (default: `range(0.01, stop=0.99, length=100)`).
# NB ξ is undocumented now because it isn't needed when assessing NeuralEstimator objects, which is the use case 99% of the time for assess(). I leave it here for backwards compatibility only. 
# - `ξ = nothing`: an arbitrary collection of objects that are fixed (e.g., distance matrices). Can also be provided as `xi`. 
# - `use_ξ = false`: a `Bool` or a collection of `Bool` objects with length equal to the number of estimators. Specifies whether or not the estimator uses `ξ`: if it does, the estimator will be applied as `estimator(Z, ξ)`. This argument is useful when multiple `estimators` are provided, only some of which need `ξ`; hence, if only one estimator is provided and `ξ` is not `nothing`, `use_ξ` is automatically set to `true`. Can also be provided as `use_xi`.
"""
	assess(estimator, θ, Z; ...)
	assess(estimators::Vector, θ, Z; ...)
Assesses an `estimator` (or a collection of `estimators`) based on true parameters `θ` and corresponding simulated data `Z`.

The parameters `θ` should be given as a ``d`` × ``K`` matrix, where ``d`` is the parameter dimension and ``K`` is the number of sampled parameter vectors. 

When `Z` contain more simulated data sets than the number ``K`` of sampled parameter vectors, `θ` will be recycled via horizontal concatenation: `θ = repeat(θ, outer = (1, J))`, where `J = numobs(Z) ÷ K` is the number of simulated data sets for each parameter vector. This allows assessment of the estimator's sampling distribution under fixed parameters.

The return value is of type [`Assessment`](@ref). 

# Keyword arguments
- `parameter_names::Vector{String}`: names of the parameters (sensible defaults provided). 
- `estimator_names::Vector{String}`: names of the estimators (sensible defaults provided).
- `use_gpu = true`: `Bool` or collection of `Bool` objects with length equal to the number of estimators.
- `probs = nothing` (applicable only to [`PointEstimator`](@ref)): probability levels taking values between 0 and 1. By default, no bootstrap uncertainty quantification is done; if `probs` is provided, it must be a two-element vector specifying the lower and upper probability levels for the non-parametric bootstrap intervals (note that parametric bootstrap is not currently supported with `assess()`).  
- `B::Integer = 400` (applicable only to [`PointEstimator`](@ref)): number of bootstrap samples. 
"""
function assess(
    estimator,
    θ::P, Z;
    parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(θ, 1)],
    estimator_name::Union{Nothing, String} = nothing,
    estimator_names::Union{Nothing, String} = nothing,
    ξ = nothing, xi = nothing,
    use_gpu::Bool = true,
    probs = nothing,
    B::Integer = 400
) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

    # Check duplicated arguments that are needed so that the R interface uses ASCII characters only
    @assert isnothing(ξ) || isnothing(xi) "Only one of `ξ` or `xi` should be provided"
    if !isnothing(xi)
        ξ = xi
    end

    # Extract the matrix of parameters and check that the parameter names match the dimension of θ
    θ = _extractθ(θ)
    d, K = size(θ)
    if θ isa NamedMatrix
        parameter_names = names(θ, 1)
    end
    @assert length(parameter_names) == d

    # Get the number of data sets and check that it conforms with the number of parameter vectors stored in θ
    KJ = numobs(Z)
    @assert KJ % K == 0 "The number of data sets in `Z` must be a multiple of the number of parameter vectors stored in `θ`"
    J = KJ ÷ K
    if J > 1
        θ = repeat(θ, outer = (1, J))
    end

    # If the data are stored as a vector, get the number of replicates stored in each element 
    if Z isa AbstractVector
        m = numberreplicates(Z)
    else
        m = fill(1, KJ)
    end

    # Apply the estimator to the data 
    if !isnothing(ξ)
        runtime = @elapsed θ̂ = estimator(Z, ξ) # note that the gpu is never used in this case
    else
        runtime = @elapsed θ̂ = estimate(estimator, Z, use_gpu = use_gpu)
    end
    θ̂ = convert(Matrix, θ̂) # convert to Matrix in case estimator returns a different format (e.g., adjoint vector)

    # Convert to DataFrame and add information
    runtime = DataFrame(runtime = runtime)
    θ̂ = DataFrame(θ̂', parameter_names)
    θ̂[!, "m"] = m
    θ̂[!, "k"] = repeat(1:K, J)
    θ̂[!, "j"] = repeat(1:J, inner = K)

    # Add estimator name if it was provided
    if !isnothing(estimator_names)
        estimator_name = estimator_names
    end # deprecation coercion
    if !isnothing(estimator_name)
        θ̂[!, "estimator"] .= estimator_name
        runtime[!, "estimator"] .= estimator_name
    end

    # Dataframe containing the true parameters, repeated if necessary 
    θ = convert(Matrix, θ)
    θ = DataFrame(θ', parameter_names)
    θ = repeat(θ, outer = nrow(θ̂) ÷ nrow(θ))
    θ = stack(θ, variable_name = :parameter, value_name = :truth) # transform to long form

    # Merge true parameters and estimates
    df = _merge(θ, θ̂)

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
        intervals = _merge2(θ, intervals)
        df[:, "lower"] = intervals[:, "lower"]
        df[:, "upper"] = intervals[:, "upper"]
        df[:, "α"] .= 1 - (probs[2] - probs[1])
    end

    return Assessment(df, runtime)
end

function assess(
    estimator::PosteriorEstimator,
    θ::P, Z;
    parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(θ, 1)],
    estimator_name::Union{Nothing, String} = nothing,
    estimator_names::Union{Nothing, String} = nothing,
    N::Integer = 1000,
    kwargs...
) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

    # Extract the matrix of parameters and check that the parameter names match the dimension of θ
    θ = _extractθ(θ)
    d, K = size(θ)
    if θ isa NamedMatrix
        parameter_names = names(θ, 1)
    end
    @assert length(parameter_names) == d

    # Get the number of data sets and check that it conforms with the number of parameter vectors stored in θ
    KJ = numobs(Z)
    @assert KJ % K == 0 "The number of data sets in `Z` must be a multiple of the number of parameter vectors stored in `θ`"
    J = KJ ÷ K
    if J > 1
        θ = repeat(θ, outer = (1, J))
    end

    # If the data are stored as a vector, get the number of replicates stored in each element 
    if Z isa AbstractVector
        m = numberreplicates(Z)
    else
        m = fill(1, KJ)
    end

    # Obtain point estimates 
    runtime = @elapsed θ̂ = posteriormedian(estimator, Z, N; kwargs...)

    # Convert to DataFrame and add information
    runtime = DataFrame(runtime = runtime)
    θ̂ = DataFrame(θ̂', parameter_names)
    θ̂[!, "m"] = m
    θ̂[!, "k"] = repeat(1:K, J)
    θ̂[!, "j"] = repeat(1:J, inner = K)

    # Add estimator name if it was provided
    if !isnothing(estimator_names)
        estimator_name = estimator_names
    end # deprecation coercion
    if !isnothing(estimator_name)
        θ̂[!, "estimator"] .= estimator_name
        runtime[!, "estimator"] .= estimator_name
    end

    # Dataframe containing the true parameters, repeated if necessary 
    θ = convert(Matrix, θ)
    θ = DataFrame(θ', parameter_names)
    θ = repeat(θ, outer = nrow(θ̂) ÷ nrow(θ))
    θ = stack(θ, variable_name = :parameter, value_name = :truth) # transform to long form

    # Merge true parameters and estimates
    df = _merge(θ, θ̂)

    return Assessment(df, runtime)
end

function assess(
    estimator::Union{IntervalEstimator, Ensemble{<:IntervalEstimator}},
    θ::P, Z;
    parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(θ, 1)],
    estimator_name::Union{Nothing, String} = nothing,
    estimator_names::Union{Nothing, String} = nothing,
    use_gpu::Bool = true
) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

    # Extract the matrix of parameters and check that the parameter names match the dimension of θ
    θ = _extractθ(θ)
    d, K = size(θ)
    if θ isa NamedMatrix
        parameter_names = names(θ, 1)
    end
    @assert length(parameter_names) == d

    # Get the number of data sets and check that it conforms with the number of parameter vectors stored in θ
    KJ = numobs(Z)
    @assert KJ % K == 0 "The number of data sets in `Z` must be a multiple of the number of parameter vectors stored in `θ`"
    J = KJ ÷ K
    if J > 1
        θ = repeat(θ, outer = (1, J))
    end

    # If the data are stored as a vector, get the number of replicates stored in each element 
    if Z isa AbstractVector
        m = numberreplicates(Z)
    else
        m = fill(1, KJ)
    end

    # Apply the estimator to data 
    runtime = @elapsed θ̂ = estimate(estimator, Z, use_gpu = use_gpu)

    # Convert to DataFrame and add information
    runtime = DataFrame(runtime = runtime)
    estimate_names = repeat(parameter_names, outer = 2) .* repeat(["_lower", "_upper"], inner = d)
    θ̂ = DataFrame(θ̂', estimate_names)
    θ̂[!, "m"] = m
    θ̂[!, "k"] = repeat(1:K, J)
    θ̂[!, "j"] = repeat(1:J, inner = K)

    # Add estimator name if it was provided
    if !isnothing(estimator_names)
        estimator_name = estimator_names
    end # deprecation coercion
    if !isnothing(estimator_name)
        θ̂[!, "estimator"] .= estimator_name
        runtime[!, "estimator"] .= estimator_name
    end

    # Dataframe containing the true parameters, repeated if necessary 
    θ = convert(Matrix, θ)
    θ = DataFrame(θ', parameter_names)
    θ = repeat(θ, outer = nrow(θ̂) ÷ nrow(θ))
    θ = stack(θ, variable_name = :parameter, value_name = :truth) # transform to long form

    # Merge true parameters and estimates
    df = _merge2(θ, θ̂)
    probs = estimator isa Ensemble{<:IntervalEstimator} ? estimator[1].probs : estimator.probs
    df[:, "α"] .= 1 - (probs[2] - probs[1])

    return Assessment(df, runtime)
end

function assess(
    estimator::Union{QuantileEstimatorContinuous, QuantileEstimatorDiscrete, Ensemble{<:QuantileEstimatorContinuous}, Ensemble{<:QuantileEstimatorDiscrete}},
    θ::P, Z;
    parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(θ, 1)],
    estimator_name::Union{Nothing, String} = nothing,
    estimator_names::Union{Nothing, String} = nothing, # for backwards compatibility
    use_gpu::Bool = true,
    probs = f32(range(0.01, stop = 0.99, length = 100))
) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

    # Extract the matrix of parameters and check that the parameter names match the dimension of θ
    θ = _extractθ(θ)
    d, K = size(θ)
    if θ isa NamedMatrix
        parameter_names = names(θ, 1)
    end
    @assert length(parameter_names) == d

    # Get the number of data sets and check that it conforms with the number of parameter vectors stored in θ
    KJ = numobs(Z)
    @assert KJ % K == 0 "The number of data sets in `Z` must be a multiple of the number of parameter vectors stored in `θ`"
    J = KJ ÷ K
    if J > 1
        θ = repeat(θ, outer = (1, J))
    end

    # If the data are stored as a vector, get the number of replicates stored in each element 
    if Z isa AbstractVector
        m = numberreplicates(Z)
    else
        m = fill(1, KJ)
    end

    # Get the probability levels 
    if estimator isa Union{QuantileEstimatorDiscrete, Ensemble{<:QuantileEstimatorDiscrete}}
        probs = estimator isa Ensemble{<:QuantileEstimatorDiscrete} ? estimator[1].probs : estimator.probs
    else
        τ = [permutedims(probs) for _ in eachindex(Z)] # convert from vector to vector of matrices
    end
    n_probs = length(probs)

    # Construct input set
    i = estimator.i
    if isnothing(i)
        if estimator isa Union{QuantileEstimatorDiscrete, Ensemble{<:QuantileEstimatorDiscrete}}
            set_info = nothing
        else
            set_info = τ
        end
    else
        θ₋ᵢ = θ[Not(i), :]
        if estimator isa Union{QuantileEstimatorDiscrete, Ensemble{<:QuantileEstimatorDiscrete}}
            set_info = eachcol(θ₋ᵢ)
        else
            # Combine each θ₋ᵢ with the corresponding vector of probability levels, which requires repeating θ₋ᵢ appropriately
            set_info = map(1:K) do k
                θ₋ᵢₖ = repeat(θ₋ᵢ[:, k:k], inner = (1, n_probs))
                vcat(θ₋ᵢₖ, probs')
            end
        end
        θ = θ[i:i, :]
        parameter_names = parameter_names[i:i]
    end

    # Estimates 
    runtime = @elapsed θ̂ = estimate(estimator, Z, set_info, use_gpu = use_gpu)

    # Convert to DataFrame and add information
    d = size(θ, 1)
    runtime = DataFrame(runtime = runtime)
    df = DataFrame(
        parameter = repeat(repeat(parameter_names, inner = n_probs), K),
        truth = repeat(vec(θ), inner = n_probs),
        prob = repeat(repeat(probs, outer = d), K),
        estimate = vec(θ̂),
        m = repeat(m, inner = n_probs*d),
        k = repeat(1:K, inner = n_probs*d),
        j = 1 # just for consistency with other methods
    )

    # Add estimator name if it was provided
    if !isnothing(estimator_names)
        estimator_name = estimator_names
    end # deprecation coercion
    if !isnothing(estimator_name)
        df[!, "estimator"] .= estimator_name
        runtime[!, "estimator"] .= estimator_name
    end

    return Assessment(df, runtime)
end

function assess(
    estimators::Union{AbstractVector, Tuple}, θ::P, Z;
    estimator_names::Union{Nothing, Vector{String}} = nothing,
    use_xi = false,
    use_ξ = false,
    ξ = nothing,
    xi = nothing,
    use_gpu = true,
    verbose::Bool = true,
    kwargs...
) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    E = length(estimators)
    if isnothing(estimator_names)
        estimator_names = ["estimator$i" for i ∈ eachindex(estimators)]
    end
    @assert length(estimator_names) == E
    if use_xi != false
        use_ξ = use_xi
    end  # NB here we check "use_xi != false" since use_xi might be a vector of bools, so it can't be used directly in the if-statement
    @assert eltype(use_ξ) == Bool
    @assert eltype(use_gpu) == Bool
    if use_ξ isa Bool
        use_ξ = repeat([use_ξ], E)
    end
    if use_gpu isa Bool
        use_gpu = repeat([use_gpu], E)
    end
    @assert length(use_ξ) == E
    @assert length(use_gpu) == E

    # Assess the estimators
    assessments = map(1:E) do i
        verbose && println("	Running $(estimator_names[i])...")
        if use_ξ[i]
            assess(estimators[i], θ, Z, ξ = ξ; use_gpu = use_gpu[i], estimator_name = estimator_names[i], kwargs...)
        else
            assess(estimators[i], θ, Z; use_gpu = use_gpu[i], estimator_name = estimator_names[i], kwargs...)
        end
    end

    # Combine the assessment objects 
    if E == 2 && any(isa.(estimators, Union{IntervalEstimator, Ensemble{<:IntervalEstimator}})) && any(isa.(estimators, Union{PointEstimator, Ensemble{<:PointEstimator}}))
        assessment = join(assessments...)
    elseif all(assessment -> names(assessment.df) == names(assessments[1].df), assessments)
        assessment = merge(assessments...)
    else
        assessment = assessments
    end

    return assessment
end

function _merge(θ, θ̂)
    non_measure_vars = [:m, :k, :j]
    if "estimator" ∈ names(θ̂)
        push!(non_measure_vars, :estimator)
    end

    # Transform θ̂ to long form
    θ̂ = stack(θ̂, Not(non_measure_vars), variable_name = :parameter, value_name = :estimate)

    # Merge θ and θ̂ by adding true parameters to θ̂
    θ̂[!, :truth] = θ[:, :truth]

    return θ̂
end

function _merge2(θ, θ̂)
    non_measure_vars = [:m, :k, :j]
    if "estimator" ∈ names(θ̂)
        push!(non_measure_vars, :estimator)
    end

    # Convert θ̂ into appropriate form
    # Lower bounds:
    df = copy(θ̂)
    select!(df, Not(contains.(names(df), "upper")))
    df = stack(df, Not(non_measure_vars), variable_name = :parameter, value_name = :lower)
    df.parameter = replace.(df.parameter, r"_lower$"=>"")
    df1 = df
    # Upper bounds:
    df = copy(θ̂)
    select!(df, Not(contains.(names(df), "lower")))
    df = stack(df, Not(non_measure_vars), variable_name = :parameter, value_name = :upper)
    df.parameter = replace.(df.parameter, r"_upper$"=>"")
    df2 = df
    # Join lower and upper bounds:
    θ̂ = innerjoin(df1, df2, on = [non_measure_vars..., :parameter])

    # Merge θ and θ̂ by adding true parameters to θ̂
    θ̂[!, :truth] = θ[:, :truth]

    return θ̂
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
- `average_over_parameters::Bool = false`: if true, the loss is averaged over all parameters; otherwise (default), the loss is averaged over each parameter separately.
- `average_over_sample_sizes::Bool = true`: if true (default), the loss is averaged over all sample sizes ``m``; otherwise, the loss is averaged over each sample size separately.
"""
risk(assessment::Assessment; args...) = risk(assessment.df; args...)

function risk(df::DataFrame;
    loss = (x, y) -> abs(x - y),
    average_over_parameters::Bool = false,
    average_over_sample_sizes::Bool = true)

    #TODO the default loss should change based on the type of estimator

    grouping_variables = "estimator" ∈ names(df) ? [:estimator] : []
    if !average_over_parameters
        push!(grouping_variables, :parameter)
    end
    if !average_over_sample_sizes
        push!(grouping_variables, :m)
    end
    df = groupby(df, grouping_variables)
    df = combine(df, [:estimate, :truth] => ((x, y) -> loss.(x, y)) => :loss, ungroup = false)
    df = combine(df, :loss => mean => :risk)

    return df
end

@doc raw"""
	bias(assessment::Assessment; ...)

Computes a Monte Carlo approximation of an estimator's bias,

```math
{\textrm{bias}}(\hat{\boldsymbol{\theta}}(\cdot))
\approx
\frac{1}{K} \sum_{k=1}^K \{\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(k)}) - \boldsymbol{\theta}^{(k)}\},
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
{\textrm{rmse}}(\hat{\boldsymbol{\theta}}(\cdot))
\approx
\sqrt{\frac{1}{K} \sum_{k=1}^K \{\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(k)}) - \boldsymbol{\theta}^{(k)}\}^2},
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

Computes a Monte Carlo approximation of an interval estimator's expected coverage,
as defined in [Hermans et al. (2022, Definition 2.1)](https://arxiv.org/abs/2110.06581),
and the proportion of parameters below and above the lower and upper bounds, respectively.

# Keyword arguments
- `average_over_parameters::Bool = false`: if true, the coverage is averaged over all parameters; otherwise (default), it is computed over each parameter separately.
- `average_over_sample_sizes::Bool = true`: if true (default), the coverage is averaged over all sample sizes ``m``; otherwise, it is computed over each sample size separately.
"""
function coverage(assessment::Assessment;
    average_over_parameters::Bool = false,
    average_over_sample_sizes::Bool = true)
    df = assessment.df

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
    df = assessment.df

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
    df = assessment.df

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
