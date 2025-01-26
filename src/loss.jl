# This is an internal function used in Flux to check the size of the
# arguments passed to a loss function
function _check_sizes(ŷ::AbstractArray, y::AbstractArray)
  for d in 1:max(ndims(ŷ), ndims(y))
   size(ŷ,d) == size(y,d) || throw(DimensionMismatch(
      "loss function expects size(ŷ) = $(size(ŷ)) to match size(y) = $(size(y))"
    ))
  end
end
_check_sizes(ŷ, y) = nothing  # pass-through, for constant label e.g. y = 1
@non_differentiable _check_sizes(ŷ::Any, y::Any)


# ---- surrogates for 0-1 loss ----

@doc raw"""
    tanhloss(θ̂, θ, k; agg = mean, joint = true)

For `k` > 0, computes the loss function,

```math
L(θ̂, θ) = \textrm{tanh}(|θ̂ - θ|/k),
```

which approximates the 0-1 loss as `k` → 0. Compared with the [`kpowerloss`](@ref), 
which may also be used as a continuous surrogate for the 0-1 loss, the gradient of
the tanh loss is bounded as |θ̂ - θ| → 0, which can improve numerical stability during 
training. 

If `joint = true`, the L₁ norm is computed over each parameter vector, so that, with 
`k` close to zero, the resulting Bayes estimator is the mode of the joint posterior distribution;
otherwise, if `joint = false`, the Bayes estimator is the vector containing the modes of the
marginal posterior distributions.

See also [`kpowerloss`](@ref).
"""
function tanhloss(θ̂, θ, k; agg = mean, joint::Bool = true)

  _check_sizes(θ̂, θ)

  d = abs.(θ̂ .- θ)
  if joint
     d = sum(d, dims = 1)
  end

  L = tanh_fast(d ./ k)

  return agg(L)
end


"""
    kpowerloss(θ̂, θ, k; agg = mean, joint = true, safeorigin = true, ϵ = 0.1)

For `k` > 0, the `k`-th power absolute-distance loss function,

```math
L(θ̂, θ) = |θ̂ - θ|ᵏ,
```

contains the squared-error, absolute-error, and 0-1 loss functions as special
cases (the latter obtained in the limit as `k` → 0). It is Lipschitz continuous
iff `k` = 1, convex iff `k` ≥ 1, and strictly convex iff `k` > 1: it is
quasiconvex for all `k` > 0.

If `joint = true`, the L₁ norm is computed over each parameter vector, so that, with 
`k` close to zero, the resulting Bayes estimator is the mode of the joint posterior distribution;
otherwise, if `joint = false`, the Bayes estimator is the vector containing the modes of the
marginal posterior distributions.

If `safeorigin = true`, the loss function is modified to avoid pathologies
around the origin, so that the resulting loss function behaves similarly to the
absolute-error loss in the `ϵ`-interval surrounding the origin.

See also [`tanhloss`](@ref).
"""
function kpowerloss(θ̂, θ, k; safeorigin::Bool = true, agg = mean, ϵ = ofeltype(θ̂, 0.1), joint::Bool = true)

   _check_sizes(θ̂, θ)

   d = abs.(θ̂ .- θ)
   if joint
      d = sum(d, dims = 1)
   end

   if safeorigin
     b = d .>  ϵ
     L = vcat(d[b] .^ k, _safefunction.(d[.!b], k, ϵ))
   else
     L = d.^k
   end

   return agg(L)
end

function _safefunction(d, k, ϵ)
  @assert d >= 0
  ϵ^(k - 1) * d
end

# ---- quantile loss ----

#TODO write the maths for when we have a vector τ
"""
    quantileloss(θ̂, θ, τ; agg = mean)
    quantileloss(θ̂, θ, τ::Vector; agg = mean)

The asymmetric quantile loss function,
```math
  L(θ̂, θ; τ) = (θ̂ - θ)(𝕀(θ̂ - θ > 0) - τ),
```
where `τ` ∈ (0, 1) is a probability level and 𝕀(⋅) is the indicator function.

The method that takes `τ` as a vector is useful for jointly approximating
several quantiles of the posterior distribution. In this case, the number of
rows in `θ̂` is assumed to be ``pr``, where ``p`` is the number of parameters and
``r`` is the number probability levels in `τ` (i.e., the length of `τ`).

# Examples
```
p = 1
K = 10
θ = rand(p, K)
θ̂ = rand(p, K)
quantileloss(θ̂, θ, 0.1)

θ̂ = rand(3p, K)
quantileloss(θ̂, θ, [0.1, 0.5, 0.9])

p = 2
θ = rand(p, K)
θ̂ = rand(p, K)
quantileloss(θ̂, θ, 0.1)

θ̂ = rand(3p, K)
quantileloss(θ̂, θ, [0.1, 0.5, 0.9])
```
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

function quantileloss(θ̂, θ, τ::V; agg = mean) where {T, V <: AbstractVector{T}}

  τ = convert(containertype(θ̂), τ) # convert τ to the gpu (this line means that users don't need to manually move τ to the gpu)

  # Check that the sizes match
  @assert size(θ̂, 2) == size(θ, 2)
  p, K = size(θ)

  if length(τ) == K # different τ for each training sample => must be training continuous quantile estimator with τ as input
    @ignore_derivatives τ = repeat(τ', p) # just repeat τ to match the number of parameters in the statistical model
    quantileloss(θ̂, θ, τ; agg = agg)
  else # otherwise, we must training a discrete quantile estimator for some fixed set of probability levels

    rp = size(θ̂, 1)
    @assert rp % p == 0
    r = rp ÷ p
    @assert length(τ) == r

    # repeat the arrays to facilitate broadcasting and indexing
    # note that repeat() cannot be differentiated by Zygote
    @ignore_derivatives τ = repeat(τ, inner = (p, 1), outer = (1, K))
    @ignore_derivatives θ = repeat(θ, r)

    quantileloss(θ̂, θ, τ; agg = agg)
  end
end

#NB matrix method is only used internally, and therefore not documented 
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
interval score is defined by

```math
S(l, u, θ; α) = (u - l) + 2α⁻¹(l - θ)𝕀(θ < l) + 2α⁻¹(θ - u)𝕀(θ > u),
```

where `α` ∈ (0, 1) and 𝕀(⋅) is the indicator function.

The method that takes a single value `θ̂` assumes that `θ̂` is a matrix with ``2p`` rows,
where ``p`` is the number of parameters in the statistical model. Then, the first
and second set of ``p`` rows will be used as `l` and `u`, respectively.

For further discussion, see Section 6 of Gneiting, T. and Raftery, A. E. (2007),
"Strictly proper scoring rules, prediction, and estimation",
Journal of the American statistical Association, 102, 359–378.
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
  p = size(θ̂, 1) ÷ 2
  l = θ̂[1:p, :]
  u = θ̂[(p+1):end, :]

  intervalscore(l, u, θ, α, agg = agg)
end