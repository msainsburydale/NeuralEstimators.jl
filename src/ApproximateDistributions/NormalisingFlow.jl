#TODO might be faster to do some of the computations on the CPU (MLPs are generally faster on the CPU)

"""
    ActNorm(d::Integer)
Activation normalisation layer (Kingma and Dhariwal, 2018) for an input of dimension `d`. 
"""
struct ActNorm{T1, T2}
    scale::T1
    bias::T2
end

# TODO functionality to initialise based on first batch
function ActNorm(d::Integer)
    scale = ones(Float32, d)
    bias = zeros(Float32, d)
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

Variables need to be permuted between coupling blocks in order for all input components to (eventually) be transformed. Note also that permutations are always invertible with absolute Jacobian determinant equal to 1. 
"""
struct Permutation{I <: Integer}
    permutation::AbstractVector{I}
    inv_permutation::AbstractVector{I}
end
function Permutation(d::Integer)
    permutation = randperm(d)                # random permutation of integers 1, …, d
    inv_permutation = sortperm(permutation)  # inverse of the permutation
    return Permutation(permutation, inv_permutation)
end
forward(layer::Permutation, θ::AbstractMatrix) = θ[layer.permutation, :]
inverse(layer::Permutation, U::AbstractMatrix) = U[layer.inv_permutation, :]

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

To prevent numerical overflows and stabilise the training of the model, the scaling factors $\boldsymbol{\kappa}_{\boldsymbol{\gamma},1}(\cdot)$ are clamped using the function 
```math
f(\boldsymbol{s}) = \frac{2c}{\pi}\tan^{-1}(\frac{\boldsymbol{s}}{c}),
```
where $c = 1.9$ is a fixed clamping threshold. This transformation ensures that the scaling factors do not grow excessively large.

Additional keyword arguments `kwargs` are passed to the [`MLP`](@ref) constructor when creating `κ₁` and `κ₂`. 
"""
struct AffineCouplingBlock{D, M <: MLP}
    scale::M
    translate::M
    d₁::D
    d₂::D
end

function AffineCouplingBlock(d₁::Integer, dstar::Integer, d₂::Integer; kwargs...)
    scale = MLP(d₁ + dstar, d₂; kwargs...)
    translate = MLP(d₁ + dstar, d₂; kwargs...)
    AffineCouplingBlock(scale, translate, d₁, d₂)
end

numdistributionalparams(block::AffineCouplingBlock) = 2 * block.d₂

const clamp_value = 1.9f0
const softclamp_scale = 2.0f0 * clamp_value / Float32(π)
softclamp(s) = softclamp_scale * atan.(s / clamp_value)

function forward(net::AffineCouplingBlock, X, Y, tz)
    S = softclamp(net.scale(Y, tz))
    T = net.translate(Y, tz)
    U = X .* exp.(S) .+ T
    log_det_J = sum(S, dims = 1)
    return U, log_det_J
end

function inverse(net::AffineCouplingBlock, U, V, tz)
    S = softclamp(net.scale(U, tz))
    T = net.translate(U, tz)
    θ = (V .- T) .* exp.(-S)
    return θ
end

struct CouplingLayer{D}
    d::D
    d₁::D
    d₂::D
    block1::Any
    block2::Any
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

function forward(layer::CouplingLayer, θ::AbstractMatrix, tz::AbstractMatrix)

    # Initialise accumulator for log determinant of Jacobian
    log_det_J = zero(similar(θ, 1, size(θ, 2)))

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
    θ2 = θ[(layer.d₁ + 1):end, :]
    U1, log_det_J1 = forward(layer.block1, θ1, θ2, tz)
    U2, log_det_J2 = forward(layer.block2, θ2, U1, tz)
    U = cat(U1, U2; dims = 1)
    log_det_J_coupling = log_det_J1 + log_det_J2 # log determinant of Jacobians from both splits
    log_det_J += log_det_J_coupling

    return U, log_det_J
end

function inverse(layer::CouplingLayer, U::AbstractMatrix, tz::AbstractMatrix)

    # Split input along first axis and perform inverse coupling
    U1 = U[1:layer.d₁, :]
    U2 = U[(layer.d₁ + 1):end, :]
    θ2 = inverse(layer.block2, U1, U2, tz)
    θ1 = inverse(layer.block1, θ2, U1, tz)
    θ = cat(θ1, θ2; dims = 1)

    # Inverse permutation and normalisation 
    if !isnothing(layer.permutation)
        θ = inverse(layer.permutation, θ)
    end
    if !isnothing(layer.actnorm)
        θ = inverse(layer.actnorm, θ)
    end

    return θ
end

@doc raw"""
    NormalisingFlow <: ApproximateDistribution
    NormalisingFlow(d::Integer, dstar::Integer; num_coupling_layers::Integer = 6, kwargs...)
A normalising flow for amortised posterior inference (e.g., [Ardizzone et al., 2019](https://openreview.net/forum?id=rJed6j0cKX); [Radev et al., 2022](https://ieeexplore.ieee.org/document/9298920)), where `d` is the dimension of 
the parameter vector and `dstar` is the dimension of the summary statistics for the data. 
    
Normalising flows are diffeomorphisms (i.e., invertible, differentiable transformations with differentiable inverses) that map a simple base distribution (e.g., standard Gaussian) to a more complex target distribution (e.g., the posterior). They achieve this by applying a sequence of learned transformations, the forms of which are chosen to be invertible and allow for tractable density computation via the change of variables formula. This allows for efficient density evaluation during the training stage, and efficient sampling during the inference stage. For further details, see the reviews by [Kobyzev et al. (2020)](https://ieeexplore.ieee.org/document/9089305) and [Papamakarios (2021)](https://dl.acm.org/doi/abs/10.5555/3546258.3546315).

`NormalisingFlow` uses affine coupling blocks (see [`AffineCouplingBlock`](@ref)), with activation normalisation ([Kingma and Dhariwal, 2018](https://dl.acm.org/doi/10.5555/3327546.3327685)) and permutations used between each block. The base distribution is taken to be a standard multivariate Gaussian distribution. 

When using a `NormalisingFlow` as the approximate distribution of a [`PosteriorEstimator`](@ref), 
the neural network should be a mapping from the sample space to ``\mathbb{R}^{d^*}``, 
where ``d^*`` is an appropriate number of summary statistics for the given parameter vector (e.g., ``d^* = d``). The summary statistics are then mapped to the parameters of the affine coupling blocks using conventional multilayer perceptrons (see [`AffineCouplingBlock`](@ref)).

# Keyword arguments
- `num_coupling_layers::Integer = 6`: number of coupling layers. 
- `kwargs`: additional keyword arguments passed to [`AffineCouplingBlock`](@ref). 
"""
struct NormalisingFlow{D} <: ApproximateDistribution
    d::D
    layers::Vector{<:CouplingLayer}
end

function NormalisingFlow(d::Integer, dstar::Integer; num_coupling_layers::Integer = 6, use_act_norm::Bool = true, kwargs...)
    layers = [CouplingLayer(d, dstar; use_act_norm = use_act_norm, kwargs...) for _ = 1:num_coupling_layers]
    NormalisingFlow(d, layers)
end

numdistributionalparams(q::NormalisingFlow) = sum(numdistributionalparams.(q.layers))

function forward(flow::NormalisingFlow, θ::AbstractMatrix, tz::AbstractMatrix)
    U = θ

    # Initialise accumulator for log determinant of Jacobian
    log_det_J = zero(similar(θ, 1, size(θ, 2)))

    for layer in flow.layers
        U, log_det = forward(layer, U, tz)
        log_det_J += log_det
    end
    return U, log_det_J
end

function inverse(flow::NormalisingFlow, U::AbstractMatrix, tz::AbstractMatrix)
    X = U
    for layer in reverse(flow.layers)
        X = inverse(layer, X, tz)
    end
    return X
end

function logdensity(flow::NormalisingFlow, θ::AbstractMatrix, tz::AbstractMatrix)
    d, K = size(θ)
    @assert d == flow.d
    @assert K == size(tz, 2)

    # Apply the forward transformation from θ to U
    U, log_det_J = forward(flow, θ, tz)

    # Log density using change-of-variables formula under a standard Gaussian base distribution: 
    # log(q) = log pᵤ(U) + log|J(U)| = -0.5*d*log(2π) -0.5||U||^2 + log|J(U)|
    log_densities = -0.5f0 * d * Float32(log(2π)) .- 0.5f0 * sum(U .* U; dims = 1) .+ log_det_J #NB log_det_J must be a 1xK matrix

    # Return as a 1xK matrix 
    return log_densities
end

function sampleposterior(flow::NormalisingFlow, tz::AbstractMatrix, N::Integer; use_gpu::Bool = true)

    # Dimension of parameter vector and device 
    d = flow.d
    device = _checkgpu(use_gpu, verbose = false)
    flow = device(flow)

    # Sample from the flow given each summary statistic in tz 
    θ = map(eachcol(tz)) do t
        # Sample from base distribution 
        U = randn(Float32, d, N) |> device

        # Expand t to match sample size
        T = repeat(reshape(t, :, 1), inner = (1, N)) |> device

        # Compute and return samples 
        inverse(flow, U, T) |> cpu
    end

    return θ
end

# The following code does it all at once (I found that this isn't faster, but perhaps it can be useful at some point)
# function sampleposterior(flow::NormalisingFlow, tz::AbstractMatrix, N::Integer; use_gpu::Bool = true) 
#     # Number of data sets and dimension of parameter vector 
#     K = size(tz, 2) 
#     d = flow.d
#
#     # Sample from base distribution 
#     U = randn(Float32, d, N * K)
#
#     # Repeat tz to match the desired sample size
#     tz = repeat(tz, inner = (1, N))
#    
#     # Determine the device
#     device = _checkgpu(use_gpu, verbose = false)
#     U = device(U)
#     tz = device(tz)
#     flow = device(flow)
#
#     # Compute samples 
#     θ = inverse(flow, U, tz) 
#
#     # Return to CPU
#     θ = cpu(θ)
#
#     # Split into a vector 
#     θ = [θ[:, (i-1)*N+1:i*N] for i in 1:K]  
#
#     return θ
# end
