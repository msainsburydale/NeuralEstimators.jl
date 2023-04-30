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


# ---- kpowerloss ----

"""
    kpowerloss(θ̂, y, k; agg = mean, safeorigin = true, ϵ = 0.1)

For `k` ∈ (0, ∞), the `k`-th power absolute-distance loss,

```math
L(θ̂, θ) = |θ̂ - θ|ᵏ,
```

contains the squared-error, absolute-error, and 0-1 loss functions as special
cases (the latter obtained in the limit as `k` → 0).

It is Lipschitz continuous iff `k` = 1, convex iff `k` ≥ 1, and strictly convex
iff `k` > 1. It is quasiconvex for all `k` > 0.

If `safeorigin = true`, the loss function is modified to avoid pathologies
around the origin, so that the resulting loss function behaves similarly to the
absolute-error loss in the `ϵ`-interval surrounding the origin.
"""
function kpowerloss(θ̂, θ, k; safeorigin::Bool = true, agg = mean, ϵ = ofeltype(θ̂, 0.1))

   _check_sizes(θ̂, θ)

   if safeorigin
     d = abs.(θ̂ .- θ)
     b = d .>  ϵ
     L = vcat(d[b] .^ k, _safefunction.(d[.!b], k, ϵ))
   else
     L = abs.(θ̂ .- θ).^k
   end

   return agg(L)
end

function _safefunction(d, k, ϵ)
  @assert d >= 0
  ϵ^(k - 1) * d
end


# ---- quantile loss ----

"""
    quantileloss(θ̂, θ, q; agg = mean)
    quantileloss(θ̂, θ, q::V; agg = mean) where {T, V <: AbstractVector{T}}

The asymmetric loss function whose minimiser is the `q`th posterior quantile; namely,
```math
L(θ̂, θ, q) = (θ̂ - θ)(𝕀(θ̂ - θ > 0) - q),
```
where `q` ∈ (0, 1) and 𝕀(⋅) is the indicator function.

The method that takes `q` as a vector is useful for jointly approximating
several quantiles of the posterior distribution. In this case, the number of
rows in `θ̂` is assumed to be pr, where p is the number of parameters: then,
`q` should be an r-vector.

For further discussion on this loss function, see Equation (7) of
Cressie, N. (2022), "Decisions, decisions, decisions in an uncertain
environment", arXiv:2209.13157.

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
function quantileloss(θ̂, θ, q; agg = mean)
  _check_sizes(θ̂, θ)
  d = θ̂ .- θ
  b = d .> 0
  b̃ = .!b
  L₁ = d[b] * (1 - q)
  L₂ = -q * d[b̃]
  L = vcat(L₁, L₂)
  agg(L)
end


function quantileloss(θ̂, θ, q::M; agg = mean) where {T, M <: AbstractMatrix{T}}

  d = θ̂ .- θ
  b = d .> 0
  b̃ = .!b
  L₁ = d[b] .* (1 .- q[b])
  L₂ = -q[b̃] .* d[b̃]
  L = vcat(L₁, L₂)
  agg(L)
end

function quantileloss(θ̂, θ, q::V; agg = mean) where {T, V <: AbstractVector{T}}

  q = convert(containertype(θ̂), q) # convert q to the gpu (this line means that users don't need to manually move q to the gpu)

  # Check that the sizes match
  @assert size(θ̂, 2) == size(θ, 2)
  p, K = size(θ)
  rp = size(θ̂, 1)
  @assert rp % p == 0
  r = rp ÷ p
  @assert length(q) == r

  # repeat the arrays to facilitate broadcasting and indexing
  # note that repeat() cannot be differentiated by Zygote
  @ignore_derivatives q = repeat(q, inner = (p, 1), outer = (1, K))
  @ignore_derivatives θ = repeat(θ, r)

  quantileloss(θ̂, θ, q; agg = agg)
end


# ---- interval score ----

"""
    intervalscore(l, u, θ, α; agg = mean)
    intervalscore(θ̂, θ, α; agg = mean)

Given a 100×(1-`α`)% confidence interval [`l`, `u`] with true value `θ`, the
interval score is defined by
```math
S(l, u, θ; α) = (u - l) + 2α⁻¹(l - θ)𝕀(θ < l) + 2α⁻¹(θ - u)𝕀(θ > u),
```
where `α` ∈ (0, 1) and 𝕀(⋅) is the indicator function.

The method that takes a single value `θ̂` assumes that `θ̂` is a matrix with 2p rows,
where p is the number of parameters in the statistical model. Then, the first
and second set of p rows will be used as `l` and `u`, respectively.

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
