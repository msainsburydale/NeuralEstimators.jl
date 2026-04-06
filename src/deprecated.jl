"""
	loadbestweights(path::String)

Returns the weights of the neural network saved as 'best_network.bson' in the given `path`.

!!! warning "Deprecated"
    `loadbestweights` is deprecated and may be removed in a future version.
"""
function loadbestweights(path::String)
    @warn "`loadbestweights` is deprecated and may be removed in a future version." maxlog=1
    loadweights(joinpath(path, "best_network.bson"))
end

function loadweights(path::String)
    @warn "`loadweights` is deprecated and may be removed in a future version." maxlog=1
    load(path, @__MODULE__)[:weights]
end

function simulategaussianprocess(args...; kwargs...)
    @warn "`simulategaussianprocess` is deprecated, use `simulategaussian` instead." maxlog=1
    simulategaussian(args...; kwargs...)
end
export simulategaussianprocess

function estimateinbatches(args...; kwargs...)
    @warn "`estimateinbatches` is deprecated, use `estimate` instead." maxlog=1
    estimate(args...; kwargs...)
end
export estimateinbatches

function _runondevice(θ̂, z, use_gpu::Bool; batchsize::Integer = 32)
    @warn "`_runondevice` is deprecated, use `estimate` instead." maxlog=1
    estimate(θ̂, z; batchsize = batchsize, use_gpu = use_gpu)
end

export subsetparameters
function subsetparameters(parameters, idx)
    @warn "`subsetparameters` is deprecated, use `getindex` instead (i.e., `parameters[idx]`)." maxlog=1
    return getobs(parameters, idx)
end

export subsetdata
function subsetdata(data, idx)
    @warn "`subsetdata` is deprecated, use `subsetreplicates` instead." maxlog=1
    return subsetreplicates(data, idx)
end