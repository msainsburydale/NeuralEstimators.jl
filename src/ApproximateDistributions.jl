"""
	ApproximateDistribution
An abstract supertype for approximate posteriors distributions used in conjunction with a [`PosteriorEstimator`](@ref).

Subtypes `A <: ApproximateDistribution` should provide methods `logdensity(q::A, θ::AbstractMatrix, Z)` and `sampleposterior(q::A, Z, N::Integer)`. 
"""
abstract type ApproximateDistribution end


@doc raw"""
    numdistributionalparams(q::ApproximateDistribution)
    numdistributionalparams(estimator::PosteriorEstimator)
The number of distributional parameters (i.e., the dimension of the space ``\mathcal{K}`` of approximate-distribution parameters ``\boldsymbol{\kappa}``). 
"""
function numdistributionalparams end

# ---- GaussianDistribution ----

@doc raw"""
    GaussianDistribution <: ApproximateDistribution
    GaussianDistribution(d::Integer)
A Gaussian distribution for amortised posterior inference, where `d` is the dimension of the parameter vector. 

When using a `GaussianDistribution` as the approximate distribution of a [`PosteriorEstimator`](@ref), 
the neural `network` of the [`PosteriorEstimator`](@ref) should be a mapping from the sample space to ``\mathbb{R}^{|\mathcal{K}|}``, 
where ``\mathcal{K}`` denotes the space of approximate-distribution parameters ``\boldsymbol{\kappa}`` and its dimension ``|\mathcal{K}|`` 
can be accessed using [`numdistributionalparams()`](@ref). 
"""
struct GaussianDistribution <: ApproximateDistribution
	d::Integer 
    covariancematrix::CovarianceMatrix
end
GaussianDistribution(d::Integer) = GaussianDistribution(d, CovarianceMatrix(d))
numdistributionalparams(q::GaussianDistribution) = q.d + q.covariancematrix.p 

#NB might be able to do this more efficiently using batched operations
function distributionparameters(q::GaussianDistribution, κ::AbstractMatrix)

    # Partition κ into components of μ and Cholesky factor L
    μ = κ[1:q.d, :]
    L = q.covariancematrix(κ[(q.d + 1):end, :], true)

    # Convert μ to a vector of vectors and L to a vector of matrices
    # NB re-using variable names (for something with a different type) confuses Zygote
    # NB Zygote doesn't like eachcol()
    K = size(κ, 2)
    μ2 = [μ[:, k] for k in 1:K]
    L2 = [vectotril(L[:, k]) for k in 1:K]
    
    return μ2, L2
end

function logdensity(q::GaussianDistribution, θ::AbstractMatrix, κ::AbstractMatrix)

    # θ = cpu(θ)

    # Ensure dimensions match
    @assert size(κ, 2) == size(θ, 2)
    @assert q.d == size(θ, 1)

    # Convert distributional parameters into appropriate form for density evaluation 
    μ, L = distributionparameters(q, κ)

    # Compute log densities 
    log_densities = [logdensity(q, θ[:, k], μ[k], L[k]) for k in 1:size(θ, 2)]

    # Return as a matrix 
    return reshape(log_densities, 1, :)
end
function logdensity(q::GaussianDistribution, theta::AbstractVector, mu::AbstractVector, L::AbstractMatrix)

    # Compute the difference
    diff = theta - mu

    # Compute the quadratic term: (L \ diff)' * (L \ diff)
    z = L \ diff
    quad_term = sum(z .* z) # equivalent to norm(z)^2

    # Compute the log determinant using the Cholesky factor
    log_det = 2 * sum(log, diag(L)) 

    # Compute the log density
    d = length(theta)
    log_density = -0.5 * (d * log(2π) + log_det + quad_term)

    return log_density
end
function sampleposterior(q::GaussianDistribution, κ::AbstractMatrix, N::Integer) #TODO change κ to AbstractVector?
    μ, L = distributionparameters(q, κ)
    μ[1] .+ L[1] * randn(q.d, N) 
end

# ---- NormalisingFlow ----

"""
    ActNorm(d::Integer)
Activation normalisation layer (Kingma and Dhariwal, 2018) for an input of dimension `d`. 
"""
struct ActNorm
    scale
    bias
end

# TODO functionality to initialise based on first batch
function ActNorm(d::Int)
    scale = ones(Float32, d)
    bias  = zeros(Float32, d)
    return ActNorm(scale, bias)
end

function forward(actnorm::ActNorm, θ::AbstractMatrix)
    U = actnorm.scale .* θ .+ actnorm.bias
    log_det_J = sum(log.(abs.(actnorm.scale)))
    return U, log_det_J
end

function inverse(actnorm::ActNorm, U::AbstractMatrix)
    return (U .- actnorm.bias) ./ actnorm.scale
end

"""
    Permutation(in::Integer)
A layer that permutes the inputs (of dimension `in`) entering a coupling block. 

Note that the variables need to be permuted between coupling blocks in order for all input components to (eventually) be transformed. Note also that permutations are always invertible with absolute Jacobian determinant equal to 1. 
"""
struct Permutation
    permutation::Vector{Integer}
    inv_permutation::Vector{Integer}
end
function Permutation(d::Integer)
    permutation = randperm(d)                # random permutation of integers 1, …, d
    inv_permutation = sortperm(permutation)  # inverse of the permutation
    return Permutation(permutation, inv_permutation)
end
forward(layer::Permutation, θ::AbstractMatrix) = θ[layer.permutation, :]
inverse(layer::Permutation, U::AbstractMatrix) = U[layer.inv_permutation, :]

"""
    MLP(in::Integer, out::Integer; kwargs...)
A traditional fully-connected multilayer perceptron (MLP) with input dimension `in` and output dimension `out`.

The method `(mlp::MLP)(x, y)` concatenates `x` and `y` along their first dimension before passing the result through the neural `network`. This functionality is used in constructs such as [`AffineCouplingBlock`](@ref). 

# Keyword arguments
- `depth::Integer = 2`: the number of hidden layers.
- `width::Integer = 128`: the width of each hidden layer.
- `activation::Function = relu`: the (non-linear) activation function used in each hidden layer.
- `output_activation::Function = identity`: the activation function used in the output layer.
"""
struct MLP{T <: Chain} # type parameter to avoid type instability
    network::T
end 
function MLP(in::Integer, out::Integer; depth::Integer = 2, width::Integer = 128, activation::Function = Flux.relu, output_activation::Function = identity)

    @assert depth > 0
    @assert width > 0

    layers = []
    push!(layers, Dense(in => width, activation))
	if depth > 1
		push!(layers, [Dense(width => width, activation) for _ ∈ 2:depth]...)
	end
	push!(layers, Dense(width => out, output_activation))

    return MLP(Chain(layers...))
end 
(mlp::MLP)(x, y) = mlp.network(cat(x, y; dims = 1))


# An invertible transformation,
# where the input is partitioned
# into two blocks. One of the
# blocks undergoes an affine
# transformation that depends on
# the other block
@doc raw"""
    AffineCouplingBlock(κ₁::MLP, κ₂::MLP)
    AffineCouplingBlock(d₁::Integer, dstar::Integer, d₂; kwargs...)
An affine coupling block used in a [`NormalisingFlow`](@ref). 

An affine coupling block splits its input $\boldsymbol{\theta}$ into two disjoint components, $\boldsymbol{\theta}_1$ and $\boldsymbol{\theta}_2$, with dimensions $d_1$ and $d_2$, respectively. The block then applies the following transformation: 
```math
\begin{aligned}
    \tilde{\boldsymbol{\theta}}_1 &= \boldsymbol{\theta}_1,\\
    \tilde{\boldsymbol{\theta}}_2 &= \boldsymbol{\theta}_2 \odot \exp\{\boldsymbol{\kappa}_{\boldsymbol{\gamma},1}(\tilde{\boldsymbol{\theta}}_1, \boldsymbol{T}(\boldsymbol{Z}))\} + \boldsymbol{\kappa}_{\boldsymbol{\gamma},2}(\tilde{\boldsymbol{\theta}}_1, \boldsymbol{T}(\boldsymbol{Z})),
\end{aligned}
```
where $\boldsymbol{\kappa}_{\boldsymbol{\gamma},1}(\cdot)$ and $\boldsymbol{\kappa}_{\boldsymbol{\gamma},2}(\cdot)$ are generic, non-invertible multilayer perceptrons (MLPs) that are functions of both the (transformed) first input component $\tilde{\boldsymbol{\theta}}_1$ and the learned $d^*$-dimensional summary statistics $\boldsymbol{T}(\boldsymbol{Z})$ (see [`PosteriorEstimator`](@ref)). 

Additional keyword arguments `kwargs` are passed to the [`MLP`](@ref) constructor when creating `κ₁` and `κ₂`. 
"""
struct AffineCouplingBlock
    scale::MLP
    translate::MLP
    d₁::Integer
    d₂::Integer
 end
 
function AffineCouplingBlock(d₁::Integer, dstar::Integer, d₂::Integer; kwargs...)
     scale = MLP(d₁ + dstar, d₂; kwargs...)
     translate = MLP(d₁ + dstar, d₂; kwargs...) 
     AffineCouplingBlock(scale, translate, d₁, d₂)
end

numdistributionalparams(block::AffineCouplingBlock) = 2 * block.d₂
 
function forward(net::AffineCouplingBlock, X, Y, TZ)
     S = net.scale(Y, TZ)
     T = net.translate(Y, TZ)
     U = X .* exp.(S) .+ T
     log_det_J = sum(S, dims = 1) 
     return U, log_det_J
end 
 
function inverse(net::AffineCouplingBlock, U, V, TZ)
     S = net.scale(U, TZ)
     T = net.translate(U, TZ)
     θ = (V .- T) .* exp.(-S)
     return θ
end 

struct CouplingLayer 
    d::Integer
    d₁::Integer 
    d₂::Integer 
    block1 
    block2
    actnorm::Union{Nothing, ActNorm}
    permutation::Union{Nothing, Permutation}
end

function CouplingLayer(d::Integer, dstar::Integer; use_act_norm::Bool = true, use_permutation::Bool = true, kwargs...)

    d₁ = div(d, 2)
    d₂ = div(d, 2) + (d % 2 != 0 ? 1 : 0)

    # Two-in-one coupling block (i.e., no inactive part after a forward pass)
    block1 = AffineCouplingBlock(d₂, dstar, d₁; kwargs...)
    block2 = AffineCouplingBlock(d₁, dstar, d₂; kwargs...)
    
    actnorm = use_act_norm ? ActNorm(d) : nothing
    permutation = use_permutation ? Permutation(d) : nothing

    CouplingLayer(d, d₁, d₂, block1, block2, actnorm, permutation)
end

numdistributionalparams(layer::CouplingLayer) = numdistributionalparams(layer.block1) + numdistributionalparams(layer.block2)

function forward(layer::CouplingLayer, θ::AbstractMatrix, TZ::AbstractMatrix) 
    
    # Initialise accumulator for log determinant of Jacobian
    log_det_J = zeros(1, size(θ, 2))

    # Normalise activation
    if !isnothing(layer.actnorm)
        θ, log_det_J_act = forward(layer.actnorm, θ) 
        log_det_J = log_det_J .+ log_det_J_act
    end

    # Permutation
    if !isnothing(layer.permutation)
        θ = forward(layer.permutation, θ) 
    end

    # Pass through coupling layer (two-in-one coupling block, i.e., no inactive part after a forward pass)
    θ1 = θ[1:layer.d₁, :] 
    θ2 = θ[layer.d₁+1:end, :]
    U1, log_det_J1 = forward(layer.block1, θ1, θ2, TZ) 
    U2, log_det_J2 = forward(layer.block2, θ2, U1, TZ) 
    U = cat(U1, U2; dims = 1)
    log_det_J_coupling = log_det_J1 + log_det_J2 # log determinant of Jacobians from both splits
    log_det_J += log_det_J_coupling
    
    return U, log_det_J
end

function inverse(layer::CouplingLayer, U::AbstractMatrix, TZ::AbstractMatrix) 

    # Split input along first axis and perform inverse coupling
    U1 = U[1:layer.d₁, :] 
    U2 = U[layer.d₁+1:end, :]
    θ2 = inverse(layer.block2, U1, U2, TZ)
    θ1 = inverse(layer.block1, θ2, U1, TZ)
    θ = cat(θ1, θ2; dims = 1)
    
    # Inverse permutation and normalisation 
    if !isnothing(layer.permutation) θ = inverse(layer.permutation, θ) end
    if !isnothing(layer.actnorm) θ = inverse(layer.actnorm, θ) end
    
    return θ
end


# A normalising flow is a diffeomorphism (i.e., an invertible, differentiable transformation with a differentiable inverse).

@doc raw"""
    NormalisingFlow <: ApproximateDistribution
    NormalisingFlow(d::Integer, dstar::Integer = d; num_coupling_layers::Integer = 6, kwargs...)
A normalising flow for amortised posterior inference, where `d` is the dimension of 
the parameter vector and `dstar` is the dimension of the summary statistics for the data. 

When using a `NormalisingFlow` as the approximate distribution of a [`PosteriorEstimator`](@ref), 
the neural `network` of the [`PosteriorEstimator`](@ref) should be a mapping from the sample space to ``\mathbb{R}^{d^*}``, 
where ``d^*`` is an appropriate number of summary statistics for the given parameter vector (e.g., ``d^* = d``).

`NormalisingFlow` uses affine coupling blocks, with the base distribution taken to be a standard multivariate Gaussian distribution. 
Activation normalisation ([Kingma and Dhariwal, 2018](https://dl.acm.org/doi/10.5555/3327546.3327685)) and permutations are used between each coupling block. 

# Keyword arguments
- `num_coupling_layers::Integer = 6`: number of coupling layers. 
- `kwargs`: additional keyword arguments passed to [`AffineCouplingBlock`](@ref). 
"""
struct NormalisingFlow <: ApproximateDistribution
	d::Integer 
    layers::AbstractVector{CouplingLayer}
end

function NormalisingFlow(d::Integer, dstar::Integer = d; num_coupling_layers::Integer = 6, use_act_norm::Bool = true, kwargs...)
    layers = [CouplingLayer(d, dstar; use_act_norm = use_act_norm, kwargs...) for _ in 1:num_coupling_layers]
    NormalisingFlow(d, layers)
end

numdistributionalparams(q::NormalisingFlow) = sum(numdistributionalparams.(q.layers))

function forward(flow::NormalisingFlow, θ::AbstractMatrix, TZ::AbstractMatrix)
    U = θ
    log_det_J = zeros(eltype(θ), 1, size(θ, 2)) # initialise accumulator
    for layer in flow.layers
        U, log_det = forward(layer, U, TZ) 
        log_det_J += log_det 
    end
    return U, log_det_J
end 

function inverse(flow::NormalisingFlow, U::AbstractMatrix, TZ::AbstractMatrix)
    X = U
    for layer in reverse(flow.layers)
        X = inverse(layer, X, TZ)
    end
    return X
end 

function logdensity(flow::NormalisingFlow, θ::AbstractMatrix, TZ::AbstractMatrix)
    
    d, K = size(θ)
    @assert d == flow.d
    @assert K == size(TZ, 2)

    # Apply the forward transformation from θ to U
    U, log_det_J = forward(flow, θ, TZ)
    
    # Log density using change-of-variables formula under a standard Gaussian base distribution: 
    # log(q) = log pᵤ(U) + log|J(U)| = -0.5*d*log(2π) -0.5||U||^2 + log|J(U)|
    log_densities = -0.5f0 * d * Float32(log(2π)) .- 0.5f0 * sum(U .* U; dims = 1) .+ log_det_J #NB log_det_J must be a 1xK matrix

    # Return as a 1xK matrix 
    return log_densities
end

function sampleposterior(flow::NormalisingFlow, TZ::AbstractMatrix, N::Integer; use_gpu::Bool = true) 

    @assert size(TZ, 2) == 1
    
    # Sample from the base distribution (standard Gaussian) and repeat TZ to match the desired sample size
    U = randn(Float32, flow.d, N)
    TZ = repeat(TZ, 1, N) 
    
    # Determine the device, either the CPU or the GPU 
    device = _checkgpu(use_gpu, verbose = false)
    U = U |> device
    TZ = TZ |> device
    flow = flow |> device

    # Compute samples 
    θ  = inverse(flow, U, TZ) 

    return cpu(θ)
end
sampleposterior(flow::NormalisingFlow, TZ::AbstractVector, N::Integer) = sampleposterior(flow, reshape(TZ, :, 1), N) 