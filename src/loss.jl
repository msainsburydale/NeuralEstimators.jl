# ---- 0-1 loss ----

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

function zeroone(ŷ, y; ρ = 0.5f0, agg = mean)
  _check_sizes(ŷ, y)
  agg((abs.(ŷ .- y)).^ρ)
end
