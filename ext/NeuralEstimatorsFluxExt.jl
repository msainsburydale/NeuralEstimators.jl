module NeuralEstimatorsFluxExt

using NeuralEstimators
using Flux
using BSON
using BSON: @save

# ---------------------- Utility functions ---------------------

import NeuralEstimators: _applywithdevice
using NeuralEstimators: _uses_deepset, _check_deepset_input, _packbatch, _resolvedevice, _DataLoader, cpu, numobs, _TrainDisplay, _bar!, _SILENT_DISPLAY
using Flux: testmode!

function _applywithdevice(network, z; batchsize::Integer = 32, kwargs...)
    _uses_deepset(network) && (z = _check_deepset_input(z))
    z = f32(z)
    batchsize = min(numobs(z), batchsize)
    device = _resolvedevice(; verbose = false, kwargs...)
    network = network |> device
    Flux.testmode!(network)
    data_loader = _DataLoader(z, batchsize, shuffle = false, partial = true)
    try
        y = map(data_loader) do zᵢ
            cpu(network(_packbatch(zᵢ) |> device))
        end
        return reduce(hcat, y)
    finally
        Flux.testmode!(network, :auto)
    end
end

import NeuralEstimators: _state
_state(ensemble) = Flux.state(Flux.cpu(ensemble))

# ---------------------- Training  ---------------------

# import NeuralEstimators: getestimator, _construct_train_state, _save_trainstate, _train_step, _risk

# function _construct_train_state(estimator::AbstractNeuralEstimator, optimiser::Optimisers.AbstractRule)
#     FluxTrainState(estimator, optimiser, Optimisers.setup(optimiser, estimator))
# end

import NeuralEstimators: getestimator, _save_estimator, _loadestimator, _train_step, _risk

getestimator(trainstate::FluxTrainState) = trainstate.model

function _save_estimator(trainstate::FluxTrainState, savepath, prefix)
    model_state = Flux.state(cpu_device()(trainstate.model))
    !ispath(savepath) && mkpath(savepath)
    @save joinpath(savepath, "$(prefix)_estimator.bson") model_state
    return nothing
end

function _loadestimator(estimator::AbstractNeuralEstimator, path::String)
    saved = BSON.load(path, @__MODULE__)
    if !haskey(saved, :model_state)
        throw(ArgumentError("$(path) does not contain the parameters of a Flux neural network (found keys $(collect(keys(saved)))). If the estimator contains Lux networks, load Lux (`using Lux`) before calling loadestimator(), or pass the corresponding LuxEstimator."))
    end
    # NB Flux.loadmodel! mutates its first argument, so we load into a copy: loadestimator()
    # returns a new estimator and never modifies the estimator it is given
    estimator = deepcopy(cpu_device()(estimator))
    try
        return Flux.loadmodel!(estimator, saved[:model_state])
    catch e
        (e isa ArgumentError || e isa DimensionMismatch) || rethrow()
        throw(ArgumentError("The parameters stored in $(path) do not match the architecture of the given estimator; please ensure that the same architecture is used when loading. Original error: $(e)"))
    end
end

function _risk(trainstate::FluxTrainState, loss, data, device)
    sum_loss = 0.0f0
    K = 0
    for (input, output) in data
        input, output = _packbatch(input) |> device, output |> device
        ls = loss(trainstate.model(input), output)
        num_obs = numobs(input)
        sum_loss += ls * num_obs
        K += num_obs
    end
    return cpu(sum_loss / K), trainstate
end

_train_step(trainstate::FluxTrainState, loss, data, device, adtype = nothing) = _train_step(trainstate, loss, data, device, adtype, _SILENT_DISPLAY, 0, 1)

function _train_step(trainstate::FluxTrainState, loss, data, device, adtype, progress::_TrainDisplay, epoch::Integer, epochs::Integer)
    sum_loss = 0.0f0
    K = 0
    n = length(data)
    i = 0
    for (input, output) in data
        i += 1
        input, output = _packbatch(input) |> device, output |> device
        ls, ∇ = Flux.withgradient(model -> loss(model(input), output), adtype, trainstate.model)
        Optimisers.update!(trainstate.optimizer_state, trainstate.model, ∇[1])
        num_obs = numobs(input)
        sum_loss += ls * num_obs
        K += num_obs
        _bar!(progress, epoch, epochs, i, n)
    end

    return cpu(sum_loss / K), trainstate
end

end
