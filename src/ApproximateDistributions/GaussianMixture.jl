@doc raw"""
    GaussianMixture <: AbstractApproximateDistribution
    GaussianMixture(d::Integer, num_summaries::Integer; num_components::Integer = 10, diagonal::Bool = true, kwargs...)
A mixture of Gaussian distributions for amortised inference with a [`PosteriorEstimator`](@ref), where `d` is the dimension of the parameter vector. 

The density of the distribution is: 
```math 
q(\boldsymbol{\theta}; \boldsymbol{\kappa}) = \sum_{j=1}^{J} \pi_j \cdot \mathcal{N}(\boldsymbol{\theta}; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j), 
```
where the parameters $\boldsymbol{\kappa}$ comprise the mixture weights $\pi_j \in [0, 1]$ subject to $\sum_{j=1}^{J} \pi_j = 1$, the mean vector $\boldsymbol{\mu}_j$ of each component, and the parameters of the covariance matrix $\boldsymbol{\Sigma}_j$ of each component.

By default (`diagonal = true`), each covariance matrix is diagonal, $\boldsymbol{\Sigma}_j = \boldsymbol{D}_j^2$ with $\boldsymbol{D}_j = \mathrm{diag}(\boldsymbol{\sigma}_j)$, and it is parameterised by the standard deviations $\boldsymbol{\sigma}_j$. If `diagonal = false`, each covariance matrix is dense, and it is parameterised through the modified Cholesky decomposition of its inverse,
```math
\boldsymbol{\Sigma}_j^{-1} = \boldsymbol{T}_j' \boldsymbol{D}_j^{-2} \boldsymbol{T}_j,
```
where $\boldsymbol{T}_j$ is a unit lower-triangular matrix. Since $|\boldsymbol{\Sigma}_j| = |\boldsymbol{D}_j|^2$, the density can then be evaluated using matrix multiplication only (i.e., without matrix inversion or triangular solves), and the diagonal case is recovered when $\boldsymbol{T}_j = \boldsymbol{I}$.

When using a `GaussianMixture` as the approximate distribution of a [`PosteriorEstimator`](@ref), the (learned) summary statistics are mapped to the mixture parameters by `depth` hidden layers of `width` units, each followed by `activation`, and then by output heads that use output activations that guarantee valid mixture parameters.

# Keyword arguments
- `num_components::Integer = 10`: number of components in the mixture.
- `diagonal::Bool = true`: whether the covariance matrix of each component is diagonal (`true`) or dense (`false`). Dense covariance matrices allow each component to capture dependence between the parameters, at the cost of $d(d-1)/2$ additional distributional parameters per component.
- `depth::Integer = 2`: the number of hidden layers preceding the output heads.
- `width::Integer = 128`: the width of each hidden layer.
- `activation = relu`: the activation function used in each hidden layer.
- `kwargs`: additional keyword arguments passed to each layer (e.g., `init_weight`, `init_bias`).
"""
struct GaussianMixture{D, M, B} <: AbstractApproximateDistribution
    d::D
    num_summaries::D
    num_components::D
    inference_network::M
    tril_bases::B # used to apply each T (see _trilbases()); `nothing` if the covariance matrices are diagonal
end
GaussianMixture(d, num_summaries, num_components, inference_network) = GaussianMixture(d, num_summaries, num_components, inference_network, nothing)

function GaussianMixture(d::Integer, num_summaries::Integer; num_components::Integer = 10, diagonal::Bool = true, depth::Integer = 2, width::Integer = 128, activation = relu, backend::Union{Nothing, Module} = nothing, kwargs...)
    @assert depth >= 0
    B = _resolvebackend(backend)
    diagonal = diagonal || d == 1 # a 1×1 covariance matrix has no off-diagonal elements

    # Hidden layers, each followed by `activation`. The output heads below read from
    # the final hidden layer; feeding them from a layer of width (2d+1)*num_components
    # would add parameters without adding expressive power, since that layer and the
    # heads are both affine and would compose to a single affine map.
    hidden = depth == 0 ? () :
             (B.Dense(num_summaries, width, activation; kwargs...),
        (B.Dense(width, width, activation; kwargs...) for _ ∈ 2:depth)...)
    head_in = depth == 0 ? num_summaries : width

    heads = (
        B.Dense(head_in, num_components; kwargs...),                     # mixture logits
        B.Dense(head_in, d * num_components, identity; kwargs...),       # μ ∈ ℝ
        B.Dense(head_in, d * num_components, softplus; kwargs...)        # σ > 0
    )
    if !diagonal
        # sub-diagonal elements of each T ∈ ℝ
        heads = (heads..., B.Dense(head_in, triangularnumber(d - 1) * num_components, identity; kwargs...))
    end

    inference_network = B.Chain(hidden..., B.Parallel(vcat, heads...))
    GaussianMixture(d, num_summaries, num_components, inference_network, diagonal ? nothing : _trilbases(d))
end

# One-hot matrices used to apply a d×d unit lower-triangular matrix T to a vector x without forming T:
# Tx = x + scatter * ((gather * x) .* t), where t contains the sub-diagonal elements of T, row by row
function _trilbases(d::Integer)
    idx = [(i, j) for i ∈ 2:d for j ∈ 1:(i - 1)]    # sub-diagonal elements of T, row by row
    gather = [j == c for (_, j) ∈ idx, c ∈ 1:d]      # selects xⱼ for each element tᵢⱼ
    scatter = [i == r for r ∈ 1:d, (i, _) ∈ idx]     # adds tᵢⱼxⱼ to the ith element of Tx
    return (gather = gather, scatter = scatter)
end

function numdistributionalparams(q::GaussianMixture)
    num_tril_params = isnothing(q.tril_bases) ? 0 : triangularnumber(q.d - 1)
    return (1 + 2 * q.d + num_tril_params) * q.num_components
end

# NB The methods used during training (distributionparameters() and _logcomponents()) are defined
# separately for diagonal and dense covariance matrices, and selected by dispatch, so that the code
# differentiated in the diagonal case is unaffected by the dense case. Sharing more code between the
# two cases is not free: Zygote cannot remove a branch on isnothing(q.tril_bases) at compile time,
# and it adds allocations to the reverse pass; and evaluating the Mahalanobis term in a helper
# function, which is passed the constant array θ, makes Enzyme throw an EnzymeRuntimeActivityError
# when Julia is run with --check-bounds=yes (as it is by Pkg.test()).
const DiagonalGaussianMixture = GaussianMixture{<:Any, <:Any, Nothing}

function distributionparameters(q::DiagonalGaussianMixture, κ::AbstractMatrix)
    end1 = q.num_components
    end2 = end1 + q.d * q.num_components

    logits = κ[1:end1, :]
    μ = κ[(end1 + 1):end2, :]
    σ = κ[(end2 + 1):end, :] .+ eltype(κ)(MIN_SCALE)

    return logits, μ, σ, nothing
end

function distributionparameters(q::GaussianMixture, κ::AbstractMatrix)
    end1 = q.num_components
    end2 = end1 + q.d * q.num_components
    end3 = end2 + q.d * q.num_components

    logits = κ[1:end1, :]
    μ = κ[(end1 + 1):end2, :]
    σ = κ[(end2 + 1):end3, :] .+ eltype(κ)(MIN_SCALE)
    t = κ[(end3 + 1):end, :] # sub-diagonal elements of each T

    return logits, μ, σ, t
end

"""Log-density of each mixture component, plus the stable log mixture weights.

The mixture weights are kept in log space throughout. Computing `log.(softmax(logits))`
instead would return -Inf once a weight underflows to zero, which leaves the log-density
finite but makes its gradient NaN, destroying the network on the next optimiser step.
"""
function _logcomponents(q::DiagonalGaussianMixture, κ::AbstractMatrix, θ::AbstractMatrix)
    d, K = size(θ)
    J = q.num_components
    logits, μ, σ = distributionparameters(q, κ)

    θ = reshape(θ, d, 1, K)
    μ = reshape(μ, d, J, K)
    σ = reshape(σ, d, J, K)
    T = eltype(σ)

    # Squared Mahalanobis term, formed as ((θ - μ)/σ)^2 so that σ is never squared
    mahal = sum(((θ .- μ) ./ σ) .^ 2, dims = 1)                      # (1, J, K)

    # log|Σ| = 2 Σᵢ log σᵢ, avoiding the overflow of σ² for large σ
    log_det = T(2) .* sum(log.(σ), dims = 1) .+ T(d) * T(log(2π))    # (1, J, K)

    log_normal = reshape(-T(0.5) .* (log_det .+ mahal), J, K)
    log_w = logsoftmax(logits; dims = 1)                             # stable log-softmax
    return log_w .+ log_normal
end

# As above, but for dense covariance matrices with Σ⁻¹ = T'D⁻²T, where D = diag(σ). The residual is
# first decorrelated, r = T(θ - μ), and then scaled as in the diagonal case; the determinant is also
# as in the diagonal case, since T has unit determinant. Here, T is applied to all components of all
# mixtures at once using only matrix multiplication (see _trilbases()), for efficiency on the GPU.
function _logcomponents(q::GaussianMixture, κ::AbstractMatrix, θ::AbstractMatrix)
    d, K = size(θ)
    J = q.num_components
    logits, μ, σ, t = distributionparameters(q, κ)

    θ = reshape(θ, d, 1, K)
    μ = reshape(μ, d, J, K)
    σ = reshape(σ, d, J, K)
    T = eltype(σ)

    # convert() since the bases are not moved to the device when using Lux.jl
    gather = @ignore_derivatives convert(typeof(t), q.tril_bases.gather)
    scatter = @ignore_derivatives convert(typeof(t), q.tril_bases.scatter)

    # Decorrelated residuals, r = T(θ - μ)
    Δ = reshape(θ .- μ, d, J * K)
    r = Δ .+ scatter * ((gather * Δ) .* reshape(t, :, J * K))

    # Squared Mahalanobis term, formed as (r/σ)^2 so that σ is never squared
    mahal = sum((reshape(r, d, J, K) ./ σ) .^ 2, dims = 1)           # (1, J, K)

    # log|Σ| = 2 Σᵢ log σᵢ, avoiding the overflow of σ² for large σ
    log_det = T(2) .* sum(log.(σ), dims = 1) .+ T(d) * T(log(2π))    # (1, J, K)

    log_normal = reshape(-T(0.5) .* (log_det .+ mahal), J, K)
    log_w = logsoftmax(logits; dims = 1)                             # stable log-softmax
    return log_w .+ log_normal
end

# Draws N samples from the mixture defined by each column of κ_all; returns a d × N × K array
function _samplemixture(q::GaussianMixture, κ_all::AbstractMatrix, N::Integer)
    d = q.d
    J = q.num_components

    θ = map(eachcol(κ_all)) do κ

        # Get the approximate-distribution parameters
        κ = reshape(κ, :, 1)
        logits, μ, σ, t = distributionparameters(q, κ)
        w = softmax(logits; dims = 1)   # the network emits logits; sampling needs weights
        μ = reshape(μ, d, J)
        σ = reshape(σ, d, J)

        # Sample component indices and corresponding samples
        if isnothing(t)
            component_indices = wsample(1:J, vec(w), N)
            μ[:, component_indices] .+ σ[:, component_indices] .* randn(d, N)
        else
            _sampledense(μ, σ, reshape(t, :, J), vec(w), N)
        end
    end

    return stack(θ)
end

# Draws N samples from a mixture with weights w and dense covariance matrices. The samples from each
# component are computed together, so that each component requires a single matrix multiplication.
function _sampledense(μ::AbstractMatrix, σ::AbstractMatrix, t::AbstractMatrix, w::AbstractVector, N::Integer)
    d, J = size(μ)
    z = randn(eltype(μ), d, N) # z ~ N(0, I)

    # With a single component (e.g., Gaussian()), there is no need to sample the component indices
    J == 1 && return _transformnormal(z, μ[:, 1], σ[:, 1], t[:, 1])

    component_indices = wsample(1:J, w, N)
    for j ∈ 1:J
        idx = findall(==(j), component_indices)
        isempty(idx) && continue
        z[:, idx] = _transformnormal(z[:, idx], μ[:, j], σ[:, j], t[:, j]) # overwrite z by the samples
    end
    return z
end

# Transforms z ~ N(0, I), stored in the columns of z, into samples from N(μ, Σ), where Σ = T⁻¹D²T⁻ᵀ with
# D = diag(σ), and where t contains the sub-diagonal elements of the unit lower-triangular matrix T, row
# by row: μ + Az ~ N(μ, Σ), where A = T⁻¹D
function _transformnormal(z::AbstractMatrix, μ::AbstractVector, σ::AbstractVector, t::AbstractVector)
    d = length(μ)
    T = Matrix{eltype(z)}(I, d, d)
    k = 0
    for row ∈ 2:d, col ∈ 1:(row - 1)
        T[row, col] = t[k += 1]
    end
    A = ldiv!(UnitLowerTriangular(T), Matrix{eltype(z)}(Diagonal(σ)))
    θ = A * z
    θ .+= μ
    return θ
end

# Stateful (Flux)
function _logdensity(q::GaussianMixture, θ::AbstractMatrix, tz::AbstractMatrix)
    d, K = size(θ)
    @assert d == q.d
    @assert K == size(tz, 2)

    κ = q.inference_network(tz)
    log_components = _logcomponents(q, κ, θ)                  # (J, K)
    return logsumexp(log_components; dims = 1)                # 1xK matrix
end

function sampleposterior(q::GaussianMixture, tz::AbstractMatrix, N::Integer; device = nothing)
    # NB always use CPU (bottleneck is wsample, which isn't vectorised)
    device = cpu_device()
    q = q |> device
    tz = tz |> device

    κ_all = q.inference_network(tz)
    return _samplemixture(q, κ_all, N)
end

# Stateless (Lux) 
function _logdensity(q::GaussianMixture, θ::AbstractMatrix, tz::AbstractMatrix, ps_q, st_q)
    d, K = size(θ)
    @assert d == q.d
    @assert K == size(tz, 2)

    κ, st_net = q.inference_network(tz, ps_q.inference_network, st_q.inference_network)
    log_components = _logcomponents(q, κ, θ)                  # (J, K)
    log_densities = logsumexp(log_components; dims = 1)       # 1xK matrix

    st_q = merge(st_q, (inference_network = st_net,))
    return log_densities, st_q
end

function sampleposterior(q::GaussianMixture, tz::AbstractMatrix, N::Integer, ps_q, st_q; device = nothing)
    # Always use CPU (bottleneck is wsample, which isn't vectorised)
    device = cpu_device()
    ps_q = ps_q |> device
    st_q = st_q |> device
    tz = tz |> device

    κ_all, _ = q.inference_network(tz, ps_q.inference_network, st_q.inference_network)
    return _samplemixture(q, κ_all, N)
end
