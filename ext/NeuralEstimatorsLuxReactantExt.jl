module NeuralEstimatorsLuxReactantExt

using NeuralEstimators
using NeuralEstimators: numobs, cpu, getestimator, _construct_train_state
using Lux
using LuxCore
using Reactant

using Lux.Training: TrainState, compute_gradients

# ---- Training ----

import NeuralEstimators: _risk, _trainstate_to_device

#TODO Here we rely on compute_gradients for compiling and caching... this is wasteful since the gradient information is not used
function _risk(trainstate::TrainState, loss, data, device::ReactantDevice, adtype)

    #TODO Can I remove this, or use Lux.GenericLoss?
    function lux_loss(model, ps, st, (input, output))
        ŷ, st_new = model(input, ps, st)
        return loss(ŷ, output), st_new, (;)
    end

    sum_loss = 0.0f0
    K = 0
    for (input, output) in device(data) #TODO Lux docs suggest device(data); check if memory is ok
        # input, output = input |> device, output |> device
        ∇, ls, _, trainstate = compute_gradients(adtype, lux_loss, (input, output), trainstate)
        sum_loss += ls * numobs(input)
        K += numobs(input)
    end
    return cpu(sum_loss / K), trainstate
end

function _trainstate_to_device(trainstate::TrainState, device::ReactantDevice)
    #TODO Have to reconstruct from scratch when using reactant... unfortunately this discards the optimiser state
    estimator = getestimator(trainstate)
    estimator = LuxEstimator(estimator.estimator, estimator.ps |> device, estimator.st |> device)
    return _construct_train_state(estimator, trainstate.optimizer)
end

end
