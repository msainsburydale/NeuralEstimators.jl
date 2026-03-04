@doc raw"""
	RatioEstimator <: NeuralEstimator
	RatioEstimator(network)
A neural estimator that estimates the likelihood-to-evidence ratio,
```math
r(\boldsymbol{Z}, \boldsymbol{\theta}) \equiv p(\boldsymbol{Z} \mid \boldsymbol{\theta})/p(\boldsymbol{Z}),
```
where $p(\boldsymbol{Z} \mid \boldsymbol{\theta})$ is the likelihood and $p(\boldsymbol{Z})$
is the marginal likelihood, also known as the model evidence.

For numerical stability, training is done on the log-scale using the relation 
$\log r(\boldsymbol{Z}, \boldsymbol{\theta}) = \text{logit}(c^*(\boldsymbol{Z}, \boldsymbol{\theta}))$, 
where $c^*(\cdot, \cdot)$ denotes the Bayes classifier as described in the [methodology](@ref "Neural ratio estimators") section. 
Hence, the neural network should be a mapping from $\mathcal{Z} \times \Theta$ to $\mathbb{R}$, 
where $\mathcal{Z}$ and $\Theta$ denote the sample and parameter spaces, respectively. 

!!! note "Network input"
    The neural network must implement a method `network(::Tuple)`, where the first element of the tuple contains the data sets and the second element contains the parameter matrices.  

When the neural network is a [`DeepSet`](@ref) (which implements the above method), two requirements must be met. First, the number of input neurons in the first layer of the outer network must equal $d$ plus the number of output neurons in the final layer of the inner network. Second, the number of output neurons in the final layer of the outer network must be one.

When applying the estimator to data, the log of the likelihood-to-evidence ratio is returned. 
The estimated ratio can then be used in various Bayesian
(e.g., [Hermans et al., 2020](https://proceedings.mlr.press/v119/hermans20a.html))
or frequentist
(e.g., [Walchessen et al., 2024](https://doi.org/10.1016/j.spasta.2024.100848))
inferential algorithms.

# Examples
```julia
using NeuralEstimators, Flux, CairoMakie

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d = 2     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 30    # number of independent replicates in each data set
sample(K) = rand32(d, K)
simulate(θ, m) = [ϑ[1] .+ ϑ[2] .* randn32(n, m) for ϑ in eachcol(θ)]

# Neural network
w = 128 
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w + d, w, relu), Dense(w, w, relu), Dense(w, 1))
network = DeepSet(ψ, ϕ)

# Initialise the estimator
r̂ = RatioEstimator(network)

# Train the estimator
r̂ = train(r̂, sample, simulate, simulator_args = m, K = 1000)

# Plot the risk history
plotrisk()

# Assess the estimator
θ_test = sample(500)
Z_test = simulate(θ_test, m);
θ_grid = f32(expandgrid(0:0.01:1, 0:0.01:1))'  # fine gridding of the parameter space
assessment = assess(r̂, θ_test, Z_test; θ_grid = θ_grid)
plot(assessment)

# Generate "observed" data 
θ = sample(1)
z = simulate(θ, 200)

# Grid-based optimization and sampling
estimate(r̂, z, θ_grid)                         # log of likelihood-to-evidence ratios
posteriormode(r̂, z; θ_grid = θ_grid)           # posterior mode 
sampleposterior(r̂, z; θ_grid = θ_grid)         # posterior samples

# Gradient-based optimization
using Optim
θ₀ = [0.5, 0.5]                                # initial estimate
posteriormode(r̂, z; θ₀ = θ₀)                   # posterior mode 
```
"""
struct RatioEstimator{N} <: NeuralEstimator
    network::N
end

#TODO maybe its better to not have a tuple, and just allow the arguments to be passed as normal... Just have to change DeepSet definition to allow two arguments in some places (this is more natural). Can easily allow backwards compat in this case too. 

function (estimator::RatioEstimator)(Z, θ; kwargs...)
    estimator((Z, θ); kwargs...) # "Tupleise" the input and pass to Tuple method
end
function (estimator::RatioEstimator)(Zθ::Tuple)
    logr = estimator.network(Zθ)
    if typeof(logr) <: AbstractVector
        logr = reduce(vcat, logr)
    end
    return logr
end

function _inputoutput(estimator::RatioEstimator, Z, θ::P) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    θ = _extractθ(θ)

    # Create independent pairs
    K = numobs(Z)
    θ̃ = subsetparameters(θ, shuffle(1:K))
    Z̃ = Z

    # Combine dependent and independent pairs
    θ = hcat(θ, θ̃)
    if Z isa AbstractVector
        Z = vcat(Z, Z̃)
    elseif Z isa AbstractMatrix
        Z = hcat(Z, Z̃)
    else
        Z = getobs(joinobs(Z, Z̃), 1:2K)
    end

    # Create class labels for output
    labels = [:dependent, :independent]
    output = onehotbatch(repeat(labels, inner = K), labels)[1:1, :]

    # Shuffle everything in case batching isn't shuffled properly downstream
    idx = shuffle(1:2K)
    Z = getobs(Z, idx)
    θ = getobs(θ, idx)
    output = output[1:1, idx]

    input = (Z, θ)
    return input, output
end

_loss(estimator::RatioEstimator, loss = nothing) = Flux.logitbinarycrossentropy

function sampleposterior(
    est::RatioEstimator, Z, N::Integer = 1000;
    logprior::Function = θ -> 0.0f0,
    θ_grid = nothing, theta_grid = nothing,
    kwargs...
    )

    @assert isnothing(θ_grid) || isnothing(theta_grid) "Only one of `θ_grid` or `theta_grid` should be given"
    if !isnothing(theta_grid)
        θ_grid = theta_grid
    end
    @assert !isnothing(θ_grid) "θ_grid must be provided for RatioEstimator"
    θ_grid = f32(θ_grid)

    # Map over datasets, returning Vector{Matrix}
    #TODO can be made more efficient by applying the network directly to Z (rather than indexing Z_j) and also g summary statistics: 
    #     come back to this once we've settled on the summary-network behaviour
    θ = map(Flux.eachobs(Z)) do Zⱼ
        logrZθ  = vec(estimate(est, Zⱼ, θ_grid; kwargs...))
        logpθ   = logprior.(eachcol(θ_grid))
        weights = exp.(logpθ .+ logrZθ)
        samples = StatsBase.wsample(eachcol(θ_grid), weights, N; replace = true)
        reduce(hcat, samples)
    end

    if length(θ) == 1
        θ = θ[1]
    end

    return θ
end

function posteriormode(
    est::RatioEstimator, Z;
    logprior::Function = θ -> 0.0f0, penalty::Union{Function, Nothing} = nothing,
    θ_grid = nothing, theta_grid = nothing,
    θ₀ = nothing, theta0 = nothing,
    kwargs...
)

    # Check duplicated arguments that are needed so that the R interface uses ASCII characters only
    @assert isnothing(θ_grid) || isnothing(theta_grid) "Only one of `θ_grid` or `theta_grid` should be given"
    @assert isnothing(θ₀) || isnothing(theta0) "Only one of `θ₀` or `theta0` should be given"
    if !isnothing(theta_grid)
        θ_grid = theta_grid
    end
    if !isnothing(theta0)
        θ₀ = theta0
    end

    # Change "penalty" to "prior"
    if !isnothing(penalty)
        logprior = penalty
    end

    # Check that we have either a grid to search over or initial estimates
    @assert !isnothing(θ_grid) || !isnothing(θ₀) "One of `θ_grid` or `θ₀` should be given"
    @assert isnothing(θ_grid) || isnothing(θ₀) "Only one of `θ_grid` and `θ₀` should be given"

    if !isnothing(θ_grid)
        θ_grid = f32(θ_grid)
        logrZθ = vec(estimate(est, Z, θ_grid; kwargs...))
        logpθ = logprior.(eachcol(θ_grid))
        logdensity = logpθ .+ logrZθ
        θ̂ = θ_grid[:, argmax(logdensity), :]   # extra colon to preserve matrix output
    else
        θ̂ = _optimdensity(θ₀, logprior, est)
    end

    return θ̂
end
posteriormode(est::RatioEstimator, Z::AbstractVector; kwargs...) = reduce(hcat, posteriormode.(Ref(est), Z; kwargs...))

InferenceOutput(::RatioEstimator) = ReturnsSamples()