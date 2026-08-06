module NeuralEstimatorsReactantExt

using NeuralEstimators
using NeuralEstimators: numobs, cpu, getestimator, _construct_train_state
using Lux
using LuxCore
using Optimisers
using Reactant
using MLDataDevices: CPUDevice

using Lux.Training: TrainState

# ---- Train-state wrapper (cross-epoch compile cache) ----

"""
Mutable wrapper around a Lux `TrainState` that caches Reactant-compiled forward
passes used by `_risk`. Lux replaces `TrainState` each training step, so the
cache must live on a stable outer object for the whole training run.
"""
mutable struct ReactantTrainState
    trainstate::TrainState
    compiled_forward::Dict{Any, Any}  # _compile_key((input, output)) => compiled RiskBatch
end

_forward(model, ps, st, input) = first(model(input, ps, st))

# Callable so forward + loss compile as one XLA graph (loss outside @compile on
# ConcreteRArrays is extremely slow).
struct RiskBatch{L}
    loss::L
end
(r::RiskBatch)(model, ps, st, input, output) = r.loss(_forward(model, ps, st, input), output)

# Cache key for compiled risk: must support RatioEstimator inputs `(Z, θ)`, etc.
_compile_key(x::AbstractArray) = size(x)
_compile_key(x::Tuple) = map(_compile_key, x)
_compile_key(x) = x

# Forward field access used by train.jl (e.g. .optimizer, .optimizer_state)
function Base.getproperty(r::ReactantTrainState, s::Symbol)
    if s === :trainstate || s === :compiled_forward
        return getfield(r, s)
    else
        return getproperty(getfield(r, :trainstate), s)
    end
end

function Base.setproperty!(r::ReactantTrainState, s::Symbol, v)
    if s === :trainstate || s === :compiled_forward
        return setfield!(r, s, v)
    else
        error("Cannot set property `$s` on ReactantTrainState; assign `trainstate` instead")
    end
end

# Share the compile cache: Reactant compiled functions are not meaningfully deepcopied
function Base.deepcopy_internal(r::ReactantTrainState, stack::IdDict)
    haskey(stack, r) && return stack[r]
    y = ReactantTrainState(getfield(r, :trainstate), getfield(r, :compiled_forward))
    stack[r] = y
    y.trainstate = Base.deepcopy_internal(getfield(r, :trainstate), stack)
    return y
end

# ---- Training primitives ----

import NeuralEstimators: _risk, _train_step, _trainstate_to_device, getestimator, _save_trainstate

function _risk(r::ReactantTrainState, loss, data, device::ReactantDevice)
    ts = r.trainstate
    st = Lux.testmode(ts.states)
    ps, model = ts.parameters, ts.model
    risk_batch = RiskBatch(loss)

    sum_loss = 0.0f0
    K = 0
    for (input, output) in device(data)
        # get! only runs the do-block on a cache miss — @compile is not re-invoked
        # every batch/epoch when the key is already present.
        key = (_compile_key(input), _compile_key(output))
        risk_c = get!(r.compiled_forward, key) do
            @compile risk_batch(model, ps, st, input, output)
        end
        ls = risk_c(model, ps, st, input, output)
        sum_loss += Float32(ls) * numobs(input)
        K += numobs(input)
    end
    return cpu(sum_loss / K), r
end

function _train_step(r::ReactantTrainState, loss, data, device, adtype)
    risk, new_ts = _train_step(r.trainstate, loss, data, device, adtype)
    r.trainstate = new_ts
    return risk, r
end

getestimator(r::ReactantTrainState) = getestimator(r.trainstate)
_save_trainstate(r::ReactantTrainState, savepath; best::Bool = true) =
    _save_trainstate(r.trainstate, savepath; best = best)

function Optimisers.adjust!(r::ReactantTrainState, eta::Real)
    r.trainstate = Optimisers.adjust!(r.trainstate, eta)
    return r
end

function Optimisers.adjust!(r::ReactantTrainState; kwargs...)
    r.trainstate = Optimisers.adjust!(r.trainstate; kwargs...)
    return r
end

function Optimisers.adjust(r::ReactantTrainState, eta::Real)
    return ReactantTrainState(Optimisers.adjust(r.trainstate, eta), r.compiled_forward)
end

function Optimisers.adjust(r::ReactantTrainState; kwargs...)
    return ReactantTrainState(Optimisers.adjust(r.trainstate; kwargs...), r.compiled_forward)
end

# ---- Device placement ----

function _trainstate_to_device(trainstate::TrainState, device::ReactantDevice)
    # Have to reconstruct from scratch when using reactant... unfortunately this discards the optimiser state
    estimator = getestimator(trainstate)
    estimator = LuxEstimator(estimator.estimator, estimator.ps |> device, estimator.st |> device)
    ts = _construct_train_state(estimator, trainstate.optimizer)
    return ReactantTrainState(ts, Dict{Any, Any}())
end

function _trainstate_to_device(r::ReactantTrainState, ::ReactantDevice)
    # Already wrapped for Reactant training; keep the compile cache
    return r
end

function _trainstate_to_device(r::ReactantTrainState, device::CPUDevice)
    # Unwrap so train() returns a plain Lux TrainState
    return device(r.trainstate)
end

end
