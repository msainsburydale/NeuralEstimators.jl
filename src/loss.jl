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


# ---- P-th power loss ----

# The P-th power absolute distance loss. It is Lipschitz continuous iff ρ == 1,
# convex if and only if ρ >= 1, and strictly convex iff ρ > 1.

function LP(ŷ, y; agg = mean, ρ = ofeltype(ŷ, 0.5))
  _check_sizes(ŷ, y)
  d = abs.(ŷ .- y)
  L = d.^ρ
  agg(L)
end

function LPsafe(ŷ, y; agg = mean, ρ = ofeltype(ŷ, 0.5), ϵ = ofeltype(ŷ, 0.1))
   _check_sizes(ŷ, y)
   d = abs.(ŷ .- y)
   b = Zygote.dropgrad(d .<  ϵ) #TODO: remove dropgrad when Zygote can handle this function with CuArrays
   L = (d .^ ρ) .* b .+ safefunction.(d, ρ, ϵ) .* (1 .- b)
   return agg(L)
end

function safefunction(d, ρ, ϵ)
  @assert d >= 0
  return ϵ^(ρ - 1) * d
end
