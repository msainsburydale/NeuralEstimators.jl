# This is an internal function used in Flux to check the size of the
# arguments passed to a loss function
function _check_sizes(ŷ::AbstractArray, y::AbstractArray)
    for d = 1:max(ndims(ŷ), ndims(y))
        size(ŷ, d) == size(y, d) || throw(DimensionMismatch(
            "loss function expects size(ŷ) = $(size(ŷ)) to match size(y) = $(size(y))"
        ))
    end
end
_check_sizes(ŷ, y) = nothing  # pass-through, for constant label e.g. y = 1
@non_differentiable _check_sizes(ŷ::Any, y::Any)

# ---- surrogates for 0-1 loss ----

@doc raw"""
    tanhloss(θ̂, θ, κ; joint::Bool = true, scale_by_parameter_dim::Bool = true)
For `κ` > 0, computes the loss function given in [Sainsbury-Dale et al. (2025; Eqn. 14)](https://arxiv.org/abs/2501.04330), namely,
```math
L(\hat{\boldsymbol{\theta}}, \boldsymbol{\theta}) = \tanh\big(\big\|\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}\|_1/\kappa\big),
```
which yields the 0-1 loss function in the limit `κ` → 0.

If `joint = true` (default), the L₁ norm is computed over each parameter vector, so that with `κ` close to zero, the resulting Bayes estimator approximates the mode of the joint posterior distribution. Otherwise, if `joint = false`, the loss function is computed as 
```math
L(\hat{\boldsymbol{\theta}}, \boldsymbol{\theta}) = \sum_{i=1}^d \tanh\big(|\hat{\theta}_i - \theta_i|/\kappa\big),
```
where $d$ denotes the dimension of the parameter vector $\boldsymbol{\theta}$. In this case, with `κ` close to zero, the resulting Bayes estimator approximates the vector containing the modes of the marginal posterior distributions.

Compared with the [`kpowerloss()`](@ref), which may also be used as a continuous approximation of the 0--1 loss function, the gradient of
this loss is bounded as ``\|\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}\|_1 \to 0``, which can improve numerical stability during training. 
"""
function tanhloss(θ̂, θ, κ; joint::Bool = true, scale_by_parameter_dim::Bool = true)
    _check_sizes(θ̂, θ)

    T = eltype(θ)
    κ = T(κ)
    p = size(θ, 1)
    scale = scale_by_parameter_dim ? sqrt(T(p)) : one(T)

    d = @. abs(θ̂ - θ)

    if joint
        d = sum(d, dims = 1) ./ scale
    end

    L = tanh_fast.(d ./ κ)
    return mean(L)
end

@doc raw"""
    kpowerloss(θ̂, θ, κ; agg = mean, safeorigin = true, ϵ = 0.1)
For `κ` > 0, the `κ`-th power absolute-distance loss function,
```math
L(\hat{\boldsymbol{\theta}}, \boldsymbol{\theta}) = \|\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}\|_1^\kappa,
```
contains the squared-error (`κ` = 2), absolute-error (`κ` = 2), and 0--1 (`κ` → 0) loss functions as special
cases. It is Lipschitz continuous if `κ` = 1, convex if `κ` ≥ 1, and strictly convex if `κ` > 1. It is
quasiconvex for all `κ` > 0.

If `safeorigin = true`, the loss function is modified to be piecewise, continuous, and linear in the `ϵ`-interval surrounding the origin, to avoid pathologies around the origin. 

See also [`tanhloss()`](@ref).
"""
function kpowerloss(θ̂, θ, κ; safeorigin::Bool = true, agg = mean, ϵ = ofeltype(θ̂, 0.1), joint::Bool = true)

    #  If `joint = true`, the L₁ norm is computed over each parameter vector, so that, with 
    # `κ` close to zero, the resulting Bayes estimator is the mode of the joint posterior distribution;
    #  otherwise, if `joint = false`, the Bayes estimator is the vector containing the modes of the
    #  marginal posterior distributions.

    _check_sizes(θ̂, θ)

    d = abs.(θ̂ .- θ)
    if joint
        d = sum(d, dims = 1)
    end

    if safeorigin
        b = d .> ϵ
        L = vcat(d[b] .^ κ, _safefunction.(d[.!b], κ, ϵ))
    else
        L = d .^ κ
    end

    return agg(L)
end

function _safefunction(d, κ, ϵ)
    @assert d >= 0
    ϵ^(κ - 1) * d
end

# ---- quantile loss ----

"""
    quantileloss(θ̂, θ, τ; agg = mean)
    quantileloss(θ̂, θ, τ::Vector; agg = mean)

The asymmetric quantile loss function,
```math
  L(θ̂, θ; τ) = (θ̂ - θ)(𝕀(θ̂ - θ > 0) - τ),
```
where `τ` ∈ (0, 1) is a probability level and 𝕀(⋅) is the indicator function.
"""
function quantileloss(θ̂, θ, τ; agg = mean)
    _check_sizes(θ̂, θ)
    d = θ̂ .- θ
    b = d .> 0
    b̃ = .!b
    L₁ = d[b] * (1 - τ)
    L₂ = -τ * d[b̃]
    L = vcat(L₁, L₂)
    agg(L)
end

# NB these methods that takes `τ` as a vector or a matrix are useful for jointly approximating
# several quantiles of the posterior distribution, but are only used internally. In this case, the number of
# rows in `θ̂` is assumed to be ``dr``, where ``d`` is the number of parameters and
# ``r`` is the number probability levels in `τ` (i.e., the length of `τ`).
function quantileloss(θ̂, θ, τ::V; agg = mean) where {T, V <: AbstractVector{T}}
    τ = convert(containertype(θ̂), τ) # convert τ to the gpu (this line means that users don't need to manually move τ to the gpu)

    # Check that the sizes match
    @assert size(θ̂, 2) == size(θ, 2)
    d, K = size(θ)

    #TODO Actually pretty brittle to check like this: breaks if the batchsize (K) is equal to length(τ) but we intended to go to the second branch
    if length(τ) == K # different τ for each training sample => must be training continuous quantile estimator with τ as input
        @ignore_derivatives τ = repeat(τ', d) # just repeat τ to match the number of parameters in the statistical model
        quantileloss(θ̂, θ, τ; agg = agg)
    else # otherwise, we must be training a discrete quantile estimator for some fixed set of probability levels
        rd = size(θ̂, 1)
        @assert rd % d == 0
        r = rd ÷ d
        @assert length(τ) == r

        # repeat the arrays to facilitate broadcasting and indexing
        # note that repeat() cannot be differentiated by Zygote
        @ignore_derivatives τ = repeat(τ, inner = (d, 1), outer = (1, K))
        @ignore_derivatives θ = repeat(θ, r)

        quantileloss(θ̂, θ, τ; agg = agg)
    end
end

function quantileloss(θ̂, θ, τ::M; agg = mean) where {T, M <: AbstractMatrix{T}}
    d = θ̂ .- θ
    b = d .> 0
    b̃ = .!b
    L₁ = d[b] .* (1 .- τ[b])
    L₂ = -τ[b̃] .* d[b̃]
    L = vcat(L₁, L₂)
    agg(L)
end



# ---- interval score ----

"""
    intervalscore(l, u, θ, α; agg = mean)
    intervalscore(θ̂, θ, α; agg = mean)
    intervalscore(assessment::Assessment; average_over_parameters::Bool = false, average_over_sample_sizes::Bool = true)

Given an interval [`l`, `u`] with nominal coverage 100×(1-`α`)%  and true value `θ`, the
interval score ([Gneiting and Raftery, 2007](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437)) is defined as
```math
S(l, u, θ; α) = (u - l) + 2α⁻¹(l - θ)𝕀(θ < l) + 2α⁻¹(θ - u)𝕀(θ > u),
```
where `α` ∈ (0, 1) and 𝕀(⋅) is the indicator function.

The method that takes a single value `θ̂` assumes that `θ̂` is a matrix with ``2d`` rows,
where ``d`` is the dimension of the parameter vector to make inference on. The first
and second sets of ``d`` rows will be used as `l` and `u`, respectively.
"""
function intervalscore(l, u, θ, α; agg = mean)
    b₁ = θ .< l
    b₂ = θ .> u

    S = u - l
    S = S + b₁ .* (2 / α) .* (l .- θ)
    S = S + b₂ .* (2 / α) .* (θ .- u)

    agg(S)
end

function intervalscore(θ̂, θ, α; agg = mean)
    @assert size(θ̂, 1) % 2 == 0
    d = size(θ̂, 1) ÷ 2
    l = θ̂[1:d, :]
    u = θ̂[(d + 1):end, :]

    intervalscore(l, u, θ, α, agg = agg)
end
