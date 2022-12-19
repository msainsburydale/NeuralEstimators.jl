"""
    kpowerloss(θ̂, y; k = ofeltype(θ̂, 0.5), agg = mean, safeorigin::Bool = true, ϵ = ofeltype(θ̂, 0.1))

The k-th power absolute distance loss,

```math
L(θ̂, θ) = (|θ̂ - θ|)^κ;   κ ∈ (0, ∞).
```

It is Lipschitz continuous iff κ = 1, convex iff κ ≥ 1, and strictly convex
iff κ > 1. It is quasiconvex for all κ > 0.

If `safeorigin = true`, the loss function is modified to avoid pathologies
around the origin, so that the resulting loss function behaves similarly to the
L₁ loss in the `ϵ`-interval surrounding the origin.
"""
function kpowerloss(θ̂, θ; safeorigin::Bool = true, agg = mean, k = ofeltype(θ̂, 0.5), ϵ = ofeltype(θ̂, 0.1))

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



# ---- Helper functions ----


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
