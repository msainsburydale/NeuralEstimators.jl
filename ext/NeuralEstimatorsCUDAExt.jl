module NeuralEstimatorsPlotExt 

using NeuralEstimators 
using CUDA
import NeuralEstimators: _checkgpu, subsetdata

function _checkgpu(use_gpu::Bool; verbose::Bool = true)

	if use_gpu && CUDA.functional()
		if verbose @info "Running on CUDA GPU" end
		CUDA.allowscalar(false)
		device = gpu
	else
		if verbose @info "Running on CPU" end
		device = cpu
	end

	return(device)
end

function subsetdata(Z::G, i) where {G <: AbstractGraph}
	if typeof(i) <: Integer i = i:i end
	sym = collect(keys(Z.ndata))[1]
	if ndims(Z.ndata[sym]) == 3
		GNNGraph(Z; ndata = Z.ndata[sym][:, i, :])
	else
		# @warn "`subsetdata()` is slow for graphical data."
		# TODO getgraph() doesn't currently work with the GPU: see https://github.com/CarloLucibello/GraphNeuralNetworks.jl/issues/161
		# TODO getgraph() doesn’t return duplicates. So subsetdata(Z, [1, 1]) returns just a single graph
		flag = Z.ndata[sym] isa CuArray
		Z = cpu(Z)
		Z = getgraph(Z, i)
		if flag Z = gpu(Z) end
		Z
	end
end


end