#TODO maybe its better to not have a tuple, and just allow the arguments to be passed as normal... Just have to change DeepSet definition to allow two arguments in some places (this is more natural). Can easily allow backwards compat in this case too. 
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
using NeuralEstimators, Flux

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
r̂ = train(r̂, sample, simulate, m = m)

# Generate "observed" data 
θ = sample(1)
z = simulate(θ, 200)[1]

# Grid-based optimization and sampling
θ_grid = f32(expandgrid(0:0.01:1, 0:0.01:1))'  # fine gridding of the parameter space
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

# function (estimator::RatioEstimator)(Zθ::Tuple; classifier::Bool = false)
#     c = σ(estimator.network(Zθ))
#     if typeof(c) <: AbstractVector
#         c = reduce(vcat, c)
#     end
#     classifier ? c : c ./ (1 .- c)
# end

# function (estimator::RatioEstimator)(Zθ::Tuple; return_log_ratio::Bool = true, return_classifier::Bool = false)
#     log_ratio = estimator.network(Zθ)
#     if return_log_ratio
#         return log_ratio
#     end

#     c = σ(log_ratio)
#     if typeof(c) <: AbstractVector
#         c = reduce(vcat, c)
#     end

#     return_classifier ? c : c ./ (1 .- c)
# end

# # Estimate ratio for many data sets and parameter vectors
# θ = sample(1000)
# Z = simulate(θ, m)
# r̂(Z, θ)                                   # log of the likelihood-to-evidence ratios


function train(estimator::RatioEstimator, args...; kwargs...)

    # Get the keyword arguments and assign the loss function
    kwargs = (; kwargs...)
    if haskey(kwargs, :loss)
        @info "The keyword argument `loss` is not required when training a $(typeof(estimator)), since in this case the binary cross-entropy loss is always used"
    end
    _train(estimator, args...; kwargs...)
end

function _constructset(estimator::RatioEstimator, Z, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    Z = f32(Z)
    θ = f32(_extractθ(θ))

    # Create independent pairs
    K = numobs(Z)
    θ̃ = subsetparameters(θ, shuffle(1:K)) #NB can use getobs here instead of subsetparameters
    Z̃ = Z # NB memory inefficient to replicate the data in this way

    # Combine dependent and independent pairs
    θ = hcat(θ, θ̃)
    if Z isa AbstractVector
        Z = vcat(Z, Z̃)
    elseif Z isa AbstractMatrix
        Z = hcat(Z, Z̃)
    else # general combine along the observation dimension... 
        # NB most of the scenarios are covered above, so the following isn't really tested
        Z = getobs(joinobs(Z, Z̃), 1:2K)
    end

    # Create class labels for output
    labels = [:dependent, :independent]
    output = onehotbatch(repeat(labels, inner = K), labels)[1:1, :]

    # Shuffle everything in case batching isn't shuffled properly downstrean
    idx = shuffle(1:2K)
    Z = getobs(Z, idx)
    θ = getobs(θ, idx)
    output = output[1:1, idx]

    # Combine data and parameters into a single tuple
    input = (Z, θ)

    _DataLoader((input, output), batchsize)
end

function _risk(estimator::RatioEstimator, loss, set::DataLoader, device, optimiser = nothing)
    sum_loss = 0.0f0
    K = 0
    for (input, output) in set
        input, output = input |> device, output |> device
        k = size(output)[end]
        loss_fn = est -> Flux.logitbinarycrossentropy(est.network(input), output)
        if !isnothing(optimiser)
            ls, ∇ = Flux.withgradient(loss_fn, estimator)
            Flux.update!(optimiser, estimator, ∇[1])
        else
            ls = loss_fn(estimator)
        end
        # Convert average loss to a sum and add to total
        sum_loss += ls * k
        K += k
    end

    return cpu(sum_loss/K)
end