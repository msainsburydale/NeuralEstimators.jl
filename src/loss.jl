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

#TODO add these functions to the documentation

"""
The P-th power absolute distance loss. It is Lipschitz continuous iff P == 1,
convex iff P >= 1, and strictly convex iff P > 1. It is quasiconvex for all P > 0.
"""
function LP(ŷ, y; agg = mean, P = ofeltype(ŷ, 0.9))
  _check_sizes(ŷ, y)
  agg(abs.(ŷ .- y).^P)
end


"""
A "safe" version of the loss function `LP()`, designed to avoid pathologies
caused by infinite gradients around the origin.

It is a piecewise loss function equal to `LP()` if
`|ŷ - y| > ϵ` and equal to a "safe" function within the `ϵ`-interval
surrounding the origin. The safe function is chosen to be linear,
so that `LPsafe()` behaves similarly to the L₁ loss near the origin.
"""
function LPsafe(ŷ, y; agg = mean, P = ofeltype(ŷ, 0.9), ϵ = ofeltype(ŷ, 0.3))

   _check_sizes(ŷ, y)
   d = abs.(ŷ .- y)
   b = d .>  ϵ

   L = vcat(d[b] .^ P, safefunction.(d[.!b], P, ϵ))
   # The following code is how another piecewise loss function, huber_loss(), is
   # written; however, it computes d .^ P for all instances, so I have chosen to
   # use the above code instead.
   # L = (d .^ P) .* b .+ safefunction.(d, P, ϵ) .* (1 .- b)

   return agg(L)
end


function safefunction(d, P, ϵ)
  @assert d >= 0
  ϵ^(P - 1) * d
end


# Another attempt at modifying LP() to avoid pathologies around the origin,
# following @mcabbott's suggestion to move slightly away from the origin.
# This function successfully avoids causing NAs in the neural network weights,
# but the loss continuously increases during training. Also, the loss is no
# longer zero when ŷ = y.
# function LPsafe(ŷ, y; agg = mean, P = ofeltype(ŷ, 0.9))
#    _check_sizes(ŷ, y)
#    agg((abs.(ŷ .- y) .+ 1f-6).^P)
# end
