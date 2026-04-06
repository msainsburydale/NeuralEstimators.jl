module NeuralEstimatorsFluxExt

using NeuralEstimators
using Flux
using BSON
using BSON: @save

# ---------------------- Utility functions ---------------------

import NeuralEstimators: _applywithdevice
using NeuralEstimators: _uses_deepset, _check_deepset_input, _resolvedevice, _DataLoader, cpu, numobs
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
            cpu(network(zᵢ |> device))
        end
        return reduce(hcat, y)
    finally
        Flux.testmode!(network, :auto)
    end
end

import NeuralEstimators: _state
_state(ensemble) = Flux.state(Flux.cpu(ensemble))

# ---------------------- Training  ---------------------

import NeuralEstimators: getestimator, _construct_train_state, _save_trainstate, _train_step, _risk

function _construct_train_state(estimator::NeuralEstimator, optimiser::Optimisers.AbstractRule)
    FluxTrainState(estimator, optimiser, Optimisers.setup(optimiser, estimator))
end

getestimator(trainstate::FluxTrainState) = trainstate.model

function _save_trainstate(trainstate::FluxTrainState, savepath; best::Bool = true)
    isnothing(savepath) && return
    prefix = best ? "best" : "final"
    optimizer = cpu_device()(trainstate.optimizer)
    optimizer_state = cpu_device()(trainstate.optimizer_state)
    # Always save optimiser and optimiser state to both savepath and tempdir
    for path in unique([savepath, tempdir()])
        !ispath(path) && mkpath(path)
        @save joinpath(path, "$(prefix)_optimizer.bson") optimizer optimizer_state
    end
    # Save full checkpoint only to explicit savepath
    if savepath != tempdir()
        model_state = Flux.state(cpu_device()(trainstate.model))
        @save joinpath(savepath, "$(prefix)_trainstate.bson") model_state optimizer optimizer_state
    end
end

function _risk(trainstate::FluxTrainState, loss, data, device, adtype = nothing) # TODO remove adtype argument once compute_gradients has been replaced in Reactant ext
    sum_loss = 0.0f0
    K = 0
    for (input, output) in data
        input, output = input |> device, output |> device
        ls = loss(trainstate.model(input), output)
        num_obs = numobs(input)
        sum_loss += ls * num_obs
        K += num_obs
    end
    return cpu(sum_loss / K), trainstate
end

function _train_step(trainstate::FluxTrainState, loss, data, device, adtype = nothing)
    sum_loss = 0.0f0
    K = 0
    for (input, output) in data
        input, output = input |> device, output |> device
        ls, ∇ = Flux.withgradient(model -> loss(model(input), output), adtype, trainstate.model)
        Optimisers.update!(trainstate.optimizer_state, trainstate.model, ∇[1])
        # Convert average loss to a sum and add to total
        num_obs = numobs(input)
        sum_loss += ls * num_obs
        K += num_obs
    end

    return cpu(sum_loss/K), trainstate
end

end
