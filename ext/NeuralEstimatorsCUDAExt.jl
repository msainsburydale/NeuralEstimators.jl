module NeuralEstimatorsCUDAExt

using NeuralEstimators
using CUDA
import NeuralEstimators: _forcegc

function _forcegc(verbose::Bool)
    if verbose
        @info "Forcing garbage collection..."
    end
    GC.gc(true)
    if CUDA.functional()
        CUDA.reclaim()
    end
    return nothing
end

end
