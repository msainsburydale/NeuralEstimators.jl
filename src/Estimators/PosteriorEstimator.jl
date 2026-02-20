
@doc raw"""
	PosteriorEstimator <: NeuralEstimator
	PosteriorEstimator(network, q::ApproximateDistribution)
    PosteriorEstimator(network, d::Integer, dstar::Integer = d; q::ApproximateDistribution = NormalisingFlow, kwargs...)
A neural estimator that approximates the posterior distribution $p(\boldsymbol{\theta} \mid \boldsymbol{Z})$, based on a neural `network` and an approximate distribution `q` (see the available in-built [Approximate distributions](@ref)). 

The neural `network` is a mapping from the sample space to a space determined by the chosen approximate distribution `q`. Often, the output space is the space $\mathcal{K}$ of the approximate-distribution parameters $\boldsymbol{\kappa}$. However, for certain distributions (notably, [`NormalisingFlow`](@ref)), the neural network outputs summary statistics of suitable dimension (e.g., the dimension $d$ of the parameter vector), which are then transformed into parameters of the approximate distribution using conventional multilayer perceptrons (see [`NormalisingFlow`](@ref)).

The convenience constructor `PosteriorEstimator(network, d::Integer, dstar::Integer = d)` builds the approximate distribution automatically, with the keyword arguments passed onto the approximate-distribution constructor.  

# Examples
```julia
using NeuralEstimators, Flux, AlgebraOfGraphics, CairoMakie

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d = 2     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 50    # number of independent replicates in each data set
sample(K) = rand32(d, K)
simulate(θ, m) = [ϑ[1] .+ ϑ[2] .* randn32(n, m) for ϑ in eachcol(θ)]

# Distribution used to approximate the posterior 
q = NormalisingFlow(d, d) 

# Neural network (outputs d summary statistics)
w = 128   
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, w, relu), Dense(w, d))
network = DeepSet(ψ, ϕ)

# Initialise the estimator
estimator = PosteriorEstimator(network, q)

# Train the estimator
estimator = train(estimator, sample, simulate, m = m)

# Assess the estimator
θ_test = sample(500)
Z_test = simulate(θ_test, m);
assessment = assess(estimator, θ_test, Z_test)
plot(assessment)

# Inference with observed data 
θ = [0.8f0 0.1f0]'
Z = simulate(θ, m)
sampleposterior(estimator, Z) # posterior draws 
posteriormean(estimator, Z)   # point estimate
```
"""
struct PosteriorEstimator{Q, N} <: NeuralEstimator
    q::Q
    network::N
end
numdistributionalparams(estimator::PosteriorEstimator) = numdistributionalparams(estimator.q)
logdensity(estimator::PosteriorEstimator, θ, Z) = logdensity(estimator.q, f32(θ), estimator.network(f32(Z)))
(estimator::PosteriorEstimator)(Zθ::Tuple) = logdensity(estimator, Zθ[2], Zθ[1]) # internal method only used during training # TODO not ideal that we assume an ordering here

# Convenience constructor
function PosteriorEstimator(network, d::Integer, dstar::Integer = d; q = NormalisingFlow, kwargs...)

    # Convert string to type if needed
    q = if q isa String
        # Get the type from the string name
        getfield(@__MODULE__, Symbol(q))
    else
        q
    end

    # Distribution used to approximate the posterior 
    q = q(d, dstar; kwargs...)

    # Initialise the estimator
    return PosteriorEstimator(q, network)
end

# Constructor for consistent argument ordering
function PosteriorEstimator(network, q::A) where {A <: ApproximateDistribution}
    return PosteriorEstimator(q, network)
end

function train(estimator::PosteriorEstimator, args...; kwargs...)

    # Get the keyword arguments and assign the loss function
    kwargs = (; kwargs...)
    if haskey(kwargs, :loss)
        @info "The keyword argument `loss` is not required when training a $(typeof(estimator)), since in this case the KL divergence is always used"
    end
    kwargs = merge(kwargs, (loss = (q, θ) -> -mean(q),))
    _train(estimator, args...; kwargs...)
end

function _constructset(estimator::PosteriorEstimator, Z, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    Z = f32(Z)
    θ = f32(_extractθ(θ))

    input = (Z, θ) # combine data and parameters into a single tuple
    output = θ     # irrelevant what we use here, just a placeholder

    _DataLoader((input, output), batchsize)
end

function assess(
    estimator::PosteriorEstimator,
    θ::P, Z;
    parameter_names::Vector{String} = ["θ$i" for i ∈ 1:size(θ, 1)],
    estimator_name::Union{Nothing, String} = nothing,
    estimator_names::Union{Nothing, String} = nothing,
    N::Integer = 1000,
    pointsummary::Function = mean, # TODO document this... just needs to be a summary function that transforms a vector of samples into a single number (acts on the marginals) 
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
    θ = convert(Matrix, θ)

    # Posterior samples 
    runtime = @elapsed samples = sampleposterior(estimator, Z, N; kwargs...)

    # Obtain point estimates 
    estimates = reduce(hcat, map.(pointsummary, eachrow.(samples)))

    # Convert to DataFrame and add information
    runtime = DataFrame(runtime = runtime)
    estimates = DataFrame(estimates', parameter_names)
    estimates[!, "k"] = repeat(1:K, J)
    estimates[!, "j"] = repeat(1:J, inner = K)

    # Add estimator name if it was provided
    if !isnothing(estimator_names) # deprecation coercion
        estimator_name = estimator_names
    end
    if !isnothing(estimator_name)
        estimates[!, "estimator"] .= estimator_name
        runtime[!, "estimator"] .= estimator_name
    end

    # Dataframe containing the true parameters, repeated if necessary 
    θ_df = DataFrame(θ', parameter_names)
    θ_df = repeat(θ_df, outer = nrow(estimates) ÷ nrow(θ_df))
    θ_df = stack(θ_df, variable_name = :parameter, value_name = :truth) # transform to long form

    # Merge true parameters and estimates
    non_measure_vars = [:k, :j]
    if "estimator" ∈ names(estimates)
        push!(non_measure_vars, :estimator)
    end
    estimates = stack(estimates, Not(non_measure_vars), variable_name = :parameter, value_name = :estimate)
    estimates[!, :truth] = θ_df[:, :truth]

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

        if !isnothing(estimator_name)
            df_s[!, "estimator"] .= estimator_name
        end

        sample_dfs[idx] = df_s
    end
    samples_df = vcat(sample_dfs...)

    return Assessment(estimates, runtime, samples_df)
end
