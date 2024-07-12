module NeuralEstimatorsMetalExt 

using NeuralEstimators 
using Metal
using NeuralEstimators: gpu, cpu # load these from NeuralEstimators so that we don't need to also make this extension require Flux to be loaded
import NeuralEstimators: _checkgpu

function _checkgpu(use_gpu::Bool; verbose::Bool = true)

	if use_gpu && Metal.functional()
		if verbose @info "Running on Apple Silicon GPU" end
		device = gpu
	else
		if verbose @info "Running on CPU" end
		device = cpu
	end

	return(device)
end

end