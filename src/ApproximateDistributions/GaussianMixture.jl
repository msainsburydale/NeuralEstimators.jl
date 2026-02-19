@doc raw"""
    GaussianMixture <: ApproximateDistribution
    GaussianMixture(d::Integer, dstar::Integer; num_components::Integer = 10, kwargs...)
A mixture of Gaussian distributions for amortised posterior inference, where `d` is the dimension of the parameter vector. 

The density of the distribution is: 
```math 
q(\boldsymbol{\theta}; \boldsymbol{\kappa}) = \sum_{j=1}^{J} \pi_j \cdot \mathcal{N}(\boldsymbol{\theta}; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j), 
```
where the parameters $\boldsymbol{\kappa}$ comprise the mixture weights $\pi_j \in [0, 1]$ subject to $\sum_{j=1}^{J} \pi_j = 1$, the mean vector $\boldsymbol{\mu}_j$ of each component, and the variance parameters of the diagonal covariance matrix $\boldsymbol{\Sigma}_j$.

When using a `GaussianMixture` as the approximate distribution of a [`PosteriorEstimator`](@ref), 
the neural network should be a mapping from the sample space to ``\mathbb{R}^{d^*}``, 
where ``d^*`` is an appropriate number of summary statistics for the parameter vector $\boldsymbol{\theta}$. The summary statistics are then mapped to the mixture parameters using a conventional multilayer perceptron ([MLP](@ref)) with approporiately chosen output activation functions (e.g., [softmax](https://fluxml.ai/Flux.jl/stable/reference/models/nnlib/#NNlib.softmax) for the mixture weights, [softplus](https://fluxml.ai/Flux.jl/stable/reference/models/activation/#NNlib.softplus) for the variance parameters).

# Keyword arguments
- `num_components::Integer = 10`: number of components in the mixture. 
- `kwargs`: additional keyword arguments passed to [`MLP`](@ref). 
"""
struct GaussianMixture{D, M} <: ApproximateDistribution
    d::D
    dstar::D
    num_components::D
    mlp::M
end
function GaussianMixture(d::Integer, dstar::Integer; num_components::Integer = 10, kwargs...)
    in = dstar
    out = (2d + 1) * num_components
    final_layer = Parallel(
        vcat,
        Chain(Dense(out, num_components), softmax),    # ∑wⱼ = 1
        Dense(out, d * num_components, identity),      # μ ∈ ℝ
        Dense(out, d * num_components, softplus)       # σ > 0
    )
    mlp = MLP(in, out; final_layer = final_layer, kwargs...)
    GaussianMixture(d, dstar, num_components, mlp)
end
numdistributionalparams(q::GaussianMixture) = (2 * q.d + 1) * q.num_components

function distributionparameters(q::GaussianMixture, κ::AbstractMatrix)
    end1 = q.num_components
    end2 = end1 + q.d * q.num_components

    w = κ[1:end1, :]
    μ = κ[(end1 + 1):end2, :]
    σ = κ[(end2 + 1):end, :]

    return w, μ, σ
end

function logdensity(q::GaussianMixture, θ::AbstractMatrix, tz::AbstractMatrix)
    d, K = size(θ)
    @assert d == q.d
    @assert K == size(tz, 2)

    # Get the approximate-distribution parameters
    w, μ, σ = distributionparameters(q, q.mlp(tz))

    # Reshape ready for broadcasting 
    J = q.num_components
    θ = reshape(θ, d, 1, K)
    μ = reshape(μ, d, J, K)
    σ = reshape(σ, d, J, K)

    # Compute squared Mahalanobis term: (θ - μ)^2 / σ^2
    diff2 = @. (θ - μ)^2 / (σ^2)   # (d, J, K)
    mahal = sum(diff2, dims = 1)     # (1, J, K)

    # Compute log determinant: sum over log(2πσ²)
    log_det = sum(log.(2π .* σ .^ 2), dims = 1) # (1, J, K)

    # Log-likelihood of each component
    log_normal = @. -0.5f0 * (log_det + mahal)
    log_normal = reshape(log_normal, J, K)

    # Combine with log mixture weights
    log_components = log.(w) .+ log_normal       # (J, K)

    # Log-sum-exp along components
    max_log = maximum(log_components, dims = 1)   # (1, K)
    log_densities = max_log + logsumexp(log_components .- max_log; dims = 1)  # (1, K)

    return log_densities # 1xK matrix 
end

function sampleposterior(q::GaussianMixture, tz::AbstractMatrix, N::Integer; use_gpu::Bool = true)
    d = q.d
    J = q.num_components
    device = _checkgpu(use_gpu, verbose = false)
    q = q |> device
    tz = tz |> device
    κ_all = q.mlp(tz) |> cpu

    θ = map(eachcol(κ_all)) do κ

        # Get the approximate-distribution parameters
        κ = reshape(κ, :, 1)
        w, μ, σ = distributionparameters(q, κ)
        μ = reshape(μ, d, J)
        σ = reshape(σ, d, J)

        # Sample component indices and corresponding samples
        component_indices = wsample(1:J, vec(w), N)
        μ[:, component_indices] .+ σ[:, component_indices] .* randn(d, N)
    end

    return θ
end