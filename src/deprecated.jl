#NB deprecated because it isn't the recommended way of storing models anymore
"""
	loadbestweights(path::String)

Returns the weights of the neural network saved as 'best_network.bson' in the given `path`.
"""
loadbestweights(path::String) = loadweights(joinpath(path, "best_network.bson"))
loadweights(path::String) = load(path, @__MODULE__)[:weights]

# aliases for backwards compatability
mapestimate = posteriormode;
export mapestimate
mlestimate = posteriormode;
export mlestimate
simulategaussianprocess = simulategaussian;
export simulategaussianprocess
estimateinbatches = estimate;
export estimateinbatches
trainx = trainmultiple;
export trainx
_runondevice(θ̂, z, use_gpu::Bool; batchsize::Integer = 32) = estimate(θ̂, z; batchsize = batchsize, use_gpu = use_gpu)

"""
Generic function that may be overloaded to implicitly define a statistical model.
Specifically, the user should provide a method `simulate(parameters, m)`
that returns `m` simulated replicates for each element in the given set of
`parameters`.
"""
function simulate end

"""
	simulate(parameters, m, J::Integer)

Simulates `J` sets of `m` independent replicates for each parameter vector in
`parameters` by calling `simulate(parameters, m)` a total of `J` times,
where the method `simulate(parameters, m)` is provided by the user via function
overloading.

# Examples
```
import NeuralEstimators: simulate

p = 2
K = 10
m = 15
parameters = rand(p, K)

# Univariate Gaussian model with unknown mean and standard deviation
simulate(parameters, m) = [θ[1] .+ θ[2] .* randn(1, m) for θ ∈ eachcol(parameters)]
simulate(parameters, m)
simulate(parameters, m, 2)
```
"""
function simulate(parameters::P, m, J::Integer; args...) where {P <: Union{AbstractMatrix, AbstractParameterSet}}
    v = [simulate(parameters, m; args...) for i ∈ 1:J]
    if typeof(v[1]) <: Tuple
        z = vcat([v[i][1] for i ∈ eachindex(v)]...)
        x = vcat([v[i][2] for i ∈ eachindex(v)]...)
        v = (z, x)
    else
        v = vcat(v...)
    end
    return v
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