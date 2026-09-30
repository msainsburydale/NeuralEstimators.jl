@doc raw"""
    Gaussian <: AbstractApproximateDistribution
    Gaussian(d::Integer, num_summaries::Integer; kwargs...)
A Gaussian distribution for amortised inference with a [`PosteriorEstimator`](@ref), where `d` is the dimension of the parameter vector. 

The density of the distribution is: 
```math 
q(\boldsymbol{\theta}; \boldsymbol{\kappa}) = \mathcal{N}(\boldsymbol{\theta}; \boldsymbol{\mu}, \boldsymbol{\Sigma}), 
```
where the parameters $\boldsymbol{\kappa}$ comprise the mean vector $\boldsymbol{\mu}$ and the
lower-triangular Cholesky factor $\boldsymbol{L}$ of the dense covariance matrix $\boldsymbol{\Sigma} = \boldsymbol{L}\boldsymbol{L}'$.

When using a `Gaussian` distribution as the approximate distribution of a [`PosteriorEstimator`](@ref), the (learned) 
summary statistics are mapped to the distribution parameters $\boldsymbol{\kappa}$ by `depth` hidden layers 
of `width` units, each followed by `activation`, and then by output heads with appropriately chosen 
activation functions (`identity` for $\boldsymbol{\mu}$ and the off-diagonal entries of $\boldsymbol{L}$, [softplus](https://fluxml.ai/Flux.jl/stable/reference/models/activation/#NNlib.softplus) for the 
diagonal entries of $\boldsymbol{L}$).

# Keyword arguments
- `depth::Integer = 2`: the number of hidden layers preceding the output heads.
- `width::Integer = 128`: the width of each hidden layer.
- `activation = relu`: the activation function used in each hidden layer.
- `kwargs`: additional keyword arguments passed to each layer (e.g., `init_weight`, `init_bias`).
"""
@concrete struct Gaussian <: AbstractApproximateDistribution
    d
    num_summaries
    inference_network
    # chol_k
    # chol_j
    chol_bases
end
Optimisers.trainable(dist::Gaussian) = (inference_network = dist.inference_network,)

function Gaussian(d::Integer, num_summaries::Integer; depth::Integer = 2, width::Integer = 128, activation = relu, backend::Union{Nothing, Module} = nothing, kwargs...)
    @assert depth >= 0
    B = _resolvebackend(backend)
    num_cov_matrix_params = d*(d+1)÷2

    # Hidden layers, each followed by `activation`. The output heads below read from
    # the final hidden layer; feeding them from a wider latent layer would add
    # parameters without adding expressive power, since that layer and the heads are
    # both affine and would compose to a single affine map.
    hidden = depth == 0 ? () :
             (B.Dense(num_summaries, width, activation; kwargs...),
        (B.Dense(width, width, activation; kwargs...) for _ ∈ 2:depth)...)
    head_in = depth == 0 ? num_summaries : width

    inference_network = B.Chain(
        hidden..., B.Parallel(vcat,
        B.Dense(head_in, d, identity; kwargs...),    # μ ∈ ℝ     
        B.Chain(                                     # L such that Σ = LL' is pos. def.
            B.Dense(head_in, num_cov_matrix_params, identity; kwargs...),
            LowerCholeskyFactor(d, B)
        )
    )
    )

    # x = d:-1:1
    # j = cumsum(x)
    # k = j .- x .+ 1
    # return Gaussian(d, num_summaries, inference_network, Tuple(k), Tuple(j)) # Tuple to prevent GPU movement when using Flux

    chol_idx = [(i, j) for j = 1:d for i = j:d]
    chol_bases = [((1:d) .== i) * ((1:d) .== j)' for (i, j) in chol_idx]
    chol_bases = stack(chol_bases, dims = 3)
    chol_bases = reshape(chol_bases, d*d, :)
    chol_bases = Matrix{Bool}(chol_bases)
    return Gaussian(d, num_summaries, inference_network, chol_bases)
end

"""
    distributionparameters(q::Gaussian, κ::AbstractMatrix)

Splits the raw network output `κ` into the mean vector `μ` (a `d×K` matrix)
and the lower Cholesky factors `L` (a `d×d×K` array) such that `Σ = LL'`.
"""
function distributionparameters(q::Gaussian, κ::AbstractMatrix)
    # Non-mutating version that works with both Zygote and Enzyme.jl
    d = q.d
    K = size(κ, 2)

    μ = κ[1:d, :]
    L = κ[(d + 1):end, :]

    # Non-mutating version with vcat
    # zero_mat = zero(L[1:d, :])
    # k, j = q.chol_k, q.chol_j
    # L̃ = vcat(L[k[1]:j[1], :], reduce(vcat, [vcat(zero_mat[1:(i-1), :], L[k[i]:j[i], :]) for i in 2:d]))

    # Non-mutating version with matrix multiplies + reshapes only
    # bases = @ignore_derivatives adapt(typeof(L), q.chol_bases) # adapt() since chol_bases doesn't get moved to the device() when using Lux.jl
    bases = @ignore_derivatives convert(typeof(L), q.chol_bases) # convert() since chol_bases doesn't get moved to the device() when using Lux.jl
    L̃ = bases * L

    return μ, reshape(L̃, d, d, K)
end

# Total number of distributional parameters: d (mean) + d(d+1)/2 (lower triangle of Σ)
numdistributionalparams(q::Gaussian) = q.d + q.d*(q.d+1)÷2

# Struct for constructing lower Cholesky factors
@concrete struct LowerCholeskyFactor <: Function # function so that we can use Lux.WrappedFunction
    d
    p
    # diag_idx
    diag_mask
end
function LowerCholeskyFactor(d::Integer)
    p = triangularnumber(d)

    diag_idx = [1]
    for i = 2:d
        push!(diag_idx, diag_idx[i - 1] + d - (i - 1) + 1)
    end

    # return LowerCholeskyFactor(d, p, Tuple(diag_idx)) # Tuple to prevent GPU movement when using Flux

    mask = falses(p)
    mask[diag_idx] .= true
    mask = reshape(mask, :, 1)
    mask = Matrix{Bool}(mask)
    return LowerCholeskyFactor(d, p, mask)
end
LowerCholeskyFactor(d::Integer, backend::Module) = LowerCholeskyFactor(d, Val(nameof(backend)))
LowerCholeskyFactor(d::Integer, ::Val{:Flux}) = LowerCholeskyFactor(d)

function (l::LowerCholeskyFactor)(v)
    # reduce(vcat, [i ∈ l.diag_idx ? softplus.(v[i:i, :]) : v[i:i, :] for i in 1:l.p])

    # mask = @ignore_derivatives adapt(typeof(v), l.diag_mask)
    mask = @ignore_derivatives convert(typeof(v), l.diag_mask)
    # MIN_SCALE floors the diagonal entries, which are softplus-parameterised and can
    # otherwise underflow to zero, making the log-determinant -Inf.
    v .+ (softplus.(v) .+ eltype(v)(MIN_SCALE) .- v) .* mask
end

# ── Stateful (Flux) ────────────────────────────────────────────────────────────

function _logdensity(q::Gaussian, θ::AbstractMatrix, tz::AbstractMatrix)
    d, K = size(θ)

    κ = q.inference_network(tz)
    μ, L = distributionparameters(q, κ)

    Δ = θ .- μ

    x = LowerTriangular.(eachslice(L, dims = 3)) .\ eachcol(Δ) |> stack

    # Enzyme doesn't like CartesianIndex...
    # diag_entries = L[CartesianIndex.(1:d, 1:d), :]
    diag_entries = reshape(L, d*d, K)[1:(d + 1):(d * d), :]
    log_det = sum(log, diag_entries, dims = 1)

    quad = sum(x .^ 2, dims = 1)

    log_densities = -d/2.0f0 * log(2.0f0 * π) .- log_det .- 0.5f0 .* quad

    return log_densities
end

function sampleposterior(q::Gaussian, tz::AbstractMatrix, N::Integer; device = nothing)
    device = cpu_device()
    q = q |> device
    tz = tz |> device

    κ = q.inference_network(tz)
    μ, L = distributionparameters(q, κ)
    d, K = size(μ)

    x = randn(eltype(μ), d, N, K)
    θ = unsqueeze(μ, dims = 2) .+ L ⊠ x  # d × N × K # NB equivalent to:  θ = reshape(μ, d, 1, K) .+ L ⊠ x

    return θ
end

# ── Stateless (Lux) ────────────────────────────────────────────────────────────

function _logdensity(q::Gaussian, θ::AbstractMatrix, tz::AbstractMatrix, ps_q, st_q)
    d, K = size(θ)

    κ, st_net = q.inference_network(tz, ps_q.inference_network, st_q.inference_network)
    μ, L = distributionparameters(q, κ)

    Δ = θ .- μ

    x = LowerTriangular.(eachslice(L, dims = 3)) .\ eachcol(Δ) |> stack

    # Enzyme doesn't like CartesianIndex...
    # diag_entries = L[CartesianIndex.(1:d, 1:d), :]
    diag_entries = reshape(L, d*d, K)[1:(d + 1):(d * d), :]
    log_det = sum(log, diag_entries, dims = 1)

    quad = sum(x .^ 2, dims = 1)

    log_densities = -d/2.0f0 * log(2.0f0 * π) .- log_det .- 0.5f0 .* quad

    st_q = merge(st_q, (inference_network = st_net,))
    return log_densities, st_q
end

function sampleposterior(q::Gaussian, tz::AbstractMatrix, N::Integer, ps_q, st_q; device = nothing)
    device = cpu_device()
    ps_q = ps_q |> device
    st_q = st_q |> device
    tz = tz |> device

    κ, _ = q.inference_network(tz, ps_q.inference_network, st_q.inference_network)
    μ, L = distributionparameters(q, κ)
    d, K = size(μ)

    x = randn(eltype(μ), d, N, K)
    θ = unsqueeze(μ, dims = 2) .+ L ⊠ x  # d × N × K # NB equivalent to:  θ = reshape(μ, d, 1, K) .+ L ⊠ x

    return θ
end
