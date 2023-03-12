# This is an internal function used in Flux to check that the size of the
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

The asymmetric loss function whose minimiser is the `q`th posterior quantile; namely,
```math
L(θ̂, θ, q) = (θ̂ - θ)(𝕀(θ̂ - θ > 0) - q),
```
where `q` ∈ (0, 1) and 𝕀(⋅) is the indicator function.

For further discussion, see Equation (7) of Cressie, N. (2022), "Decisions,
decisions, decisions in an uncertain environment", arXiv:2209.13157.
"""
function quantileloss(θ̂, θ, q; agg = mean)
  _check_sizes(θ̂, θ)
  d = θ̂ .- θ
  b = d .> 0
  L₁ = d[b] * (1 - q)
  L₂ = -q * d[.!b]
  L = vcat(L₁, L₂)
  agg(L)
end


# ---- interval score ----

"""
    intervalscore(l, u, θ, α; agg = mean)

Given a 100×(1-`α`)% confidence interval [`l`, `u`] with true value `θ`, the
interval score is defined by
```math
S(l, u, θ; α) = (u - l) + 2α⁻¹(l - θ)𝕀(θ < l) + 2α⁻¹(θ - u)𝕀(θ > u),
```
where `α` ∈ (0, 1) and 𝕀(⋅) is the indicator function.

For further discussion, see Section 6 of Gneiting, T. and Raftery, A. E. (2007),
"Strictly proper scoring rules, prediction, and estimation",
Journal of the American statistical Association, 102, 359–378.
"""
function intervalscore(l, u, θ, α; agg = mean)

  b₁ = θ .< l
  b₂ = θ .> u
  b₀ = .!(b₁ .| b₂)

  S₀ = (u[b₀] - l[b₀])
  S₁ = (u[b₁] - l[b₁]) + (2 / α) * (l[b₁] .- θ[b₁])
  S₂ = (u[b₂] - l[b₂]) + (2 / α) * (θ[b₂] .- u[b₂])

  S = vcat(S₀, S₁, S₂)
  agg(S)
end
