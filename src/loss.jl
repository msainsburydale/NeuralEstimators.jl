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

# The P-th power absolute distance loss. It is Lipschitz continuous iff P == 1,
# convex if and only if P >= 1, and strictly convex iff P > 1.

function LP(ŷ, y; agg = mean, P = ofeltype(ŷ, 0.9))
  _check_sizes(ŷ, y)
  agg(abs.(ŷ .- y).^P)
end

# First attempt at modifying LP() to avoid pathologies around the origin,
# following @mcabbott's suggestion to move slightly away from the origin.
function LPsafe(ŷ, y; agg = mean, P = ofeltype(ŷ, 0.9))
   # _check_sizes(ŷ, y)
   agg((abs.(ŷ .- y) .+ 1f-6).^P)
end

# Second attempt at modifying LP() to avoid pathologies around the origin.
# The idea is to make a piecewise loss function that is equal to LP() if
# |ŷ - y| > ϵ and equal to a "safe" loss function within the ϵ-interval
# surrounding the origin. Here, we choose the safe function to be linear,
# so that the piecewise loss behaves similarly to the L₁ loss near the origin.
function LPsafeII(ŷ, y; agg = mean, P = ofeltype(ŷ, 0.9), ϵ = ofeltype(ŷ, 0.3))
   _check_sizes(ŷ, y)
   d = abs.(ŷ .- y)
   b = Zygote.dropgrad(d .<  ϵ) #TODO: remove dropgrad when Zygote can handle this function with CuArrays
   L = (d .^ P) .* b .+ safefunction.(d, P, ϵ) .* (1 .- b)
   return agg(L)
end

function safefunction(d, P, ϵ)
  @assert d >= 0
  ϵ^(P - 1) * d
end
