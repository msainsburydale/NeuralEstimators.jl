module NeuralEstimatorsCUDAExt

using NeuralEstimators
using CUDA
using Flux: gpu, cpu
import NeuralEstimators: _checkgpu

function _checkgpu(use_gpu::Bool; verbose::Bool = true)
    if use_gpu && CUDA.functional()
        if verbose
            @info "Running on CUDA GPU"
        end
        CUDA.allowscalar(false)
        device = gpu
    else
        if verbose
            @info "Running on CPU"
        end
        device = cpu
    end

    return (device)
end

end
