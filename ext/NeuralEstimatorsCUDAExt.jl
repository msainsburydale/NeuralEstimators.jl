module NeuralEstimatorsCUDAExt

using NeuralEstimators
using CUDA
import NeuralEstimators: _forcegc

function _forcegc(verbose::Bool)
    GC.gc(true)
    if CUDA.functional()
        CUDA.reclaim()
    end
    return nothing
end

end
