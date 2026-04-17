module NeuralEstimatorsLuxExt

using NeuralEstimators
using NeuralEstimators: @set, _uses_deepset, numobs, _resolvedevice, _DataLoader, cpu
import NeuralEstimators: estimate
import NeuralEstimators: summarystatistics, _summarystatistics, _applywithdevice

using Lux
using LuxCore
using Optimisers
using Random

# ---- Initialization ----

import LuxCore: initialparameters, initialstates, parameterlength, statelength

# Tell Lux how to traverse estimator structs 
# NB These reuels imply that all neural networks must be full Lux models (can't use DeepSet directly, would need a Lux version)
const LuxTraversable = Union{LuxCore.AbstractLuxLayer, ApproximateDistribution}

function LuxCore.initialparameters(rng::AbstractRNG, estimator::NeuralEstimator)
    NamedTuple(f => LuxCore.initialparameters(rng, getfield(estimator, f))
               for f in fieldnames(typeof(estimator))
               if getfield(estimator, f) isa LuxTraversable)
end
function LuxCore.initialstates(rng::AbstractRNG, estimator::NeuralEstimator)
    NamedTuple(f => LuxCore.initialstates(rng, getfield(estimator, f))
               for f in fieldnames(typeof(estimator))
               if getfield(estimator, f) isa LuxTraversable)
end
function LuxCore.parameterlength(estimator::NeuralEstimator)
    sum(f -> LuxCore.parameterlength(getfield(estimator, f)),
        filter(f -> getfield(estimator, f) isa LuxTraversable, fieldnames(typeof(estimator))))
end
function LuxCore.statelength(estimator::NeuralEstimator)
    sum(f -> LuxCore.statelength(getfield(estimator, f)),
        filter(f -> getfield(estimator, f) isa LuxTraversable, fieldnames(typeof(estimator))),
        init = 0)  # init=0 handles the case where there are no matching fields
end

function LuxCore.initialparameters(rng::AbstractRNG, q::ApproximateDistribution)
    NamedTuple(f => LuxCore.initialparameters(rng, getfield(q, f))
               for f in fieldnames(typeof(q))
               if getfield(q, f) isa LuxCore.AbstractLuxLayer)
end
function LuxCore.initialstates(rng::AbstractRNG, q::ApproximateDistribution)
    NamedTuple(f => LuxCore.initialstates(rng, getfield(q, f))
               for f in fieldnames(typeof(q))
               if getfield(q, f) isa LuxCore.AbstractLuxLayer)
end

function LuxCore.parameterlength(q::ApproximateDistribution)
    sum(LuxCore.parameterlength(getfield(q, f))
        for f in fieldnames(typeof(q))
        if getfield(q, f) isa LuxCore.AbstractLuxLayer)
end

# ---- Utility functions ----

using Lux: testmode

function _applywithdevice(network, z, ps, st; batchsize::Integer = 32, kwargs...)
    _uses_deepset(network) && (z = _check_deepset_input(z))
    z = f32(z)
    batchsize = min(numobs(z), batchsize)
    device = _resolvedevice(; verbose = false, kwargs...)
    data_loader = _DataLoader(z, batchsize, shuffle = false, partial = true)
    st = Lux.testmode(st)
    ps, st = ps |> device, st |> device
    y = map(data_loader) do zᵢ
        cpu(first(network(zᵢ |> device, ps, st)))
    end
    return reduce(hcat, y)
end

import NeuralEstimators: _identity_layer
_identity_layer(::Val{:Lux}) = Lux.WrappedFunction(identity)

import NeuralEstimators: LowerCholeskyFactor
LowerCholeskyFactor(d::Integer, ::Val{:Lux}) = Lux.WrappedFunction(LowerCholeskyFactor(d))

# ---- LuxEstimator ----

import NeuralEstimators: LuxEstimator

# Convenience constructor so that users don't need to manually initialize the network parameters and states (ps and st)
LuxEstimator(estimator::NeuralEstimator; rng::AbstractRNG = Random.default_rng()) = LuxEstimator(estimator, Lux.setup(rng, estimator)...)

# Forward mode that can be used to directly apply a LuxEstimator to data
(estimator::LuxEstimator)(args...) = first(estimator.estimator(args..., estimator.ps, Lux.testmode(estimator.st)))

# ---- Training ----

import NeuralEstimators: getestimator
import NeuralEstimators: _construct_train_state, _risk, _train_step, _save_trainstate
import Lux.Training: TrainState

using NeuralEstimators: @save
function _save_trainstate(trainstate::TrainState, savepath; best::Bool = true)
    isnothing(savepath) && return
    prefix = best ? "best" : "final"
    parameters = cpu_device()(trainstate.parameters)
    states = cpu_device()(trainstate.states)
    optimizer = cpu_device()(trainstate.optimizer)
    optimizer_state = cpu_device()(trainstate.optimizer_state)
    # Always save optimiser and optimiser state to both savepath and tempdir
    for path in unique([savepath, tempdir()])
        !ispath(path) && mkpath(path)
        @save joinpath(path, "$(prefix)_optimizer.bson") optimizer optimizer_state
    end
    # Save full checkpoint only to explicit savepath
    if savepath != tempdir()
        @save joinpath(savepath, "$(prefix)_trainstate.bson") parameters states optimizer optimizer_state
    end
end

TrainState(e::LuxEstimator, optimiser::Optimisers.AbstractRule) = Lux.Training.TrainState(e.estimator, e.ps, e.st, optimiser)
LuxEstimator(trainstate::TrainState) = LuxEstimator(trainstate.model, trainstate.parameters, trainstate.states)
getestimator(trainstate::TrainState) = LuxEstimator(trainstate)
_construct_train_state(estimator::LuxEstimator, optimiser::Optimisers.AbstractRule) = TrainState(estimator, optimiser)

function _risk(trainstate::TrainState, loss, data, device, adtype = nothing) # TODO remove adtype argument once compute_gradients has been replaced in Reactant ext
    st = Lux.testmode(trainstate.states) #|> device
    ps = trainstate.parameters #|> device
    sum_loss = 0.0f0
    K = 0
    for (input, output) in data
        input, output = input |> device, output |> device
        ŷ = first(trainstate.model(input, ps, st))
        ls = loss(ŷ, output)
        sum_loss += ls * numobs(input)
        K += numobs(input)
    end
    return cpu(sum_loss / K), trainstate
end

function _train_step(trainstate::Lux.Training.TrainState, loss, data, device, adtype)

    # Wrap the 2-argument loss into the 4-argument form required by single_train_step!
    function lux_loss(model, ps, st, (input, output))
        ŷ, st_new = model(input, ps, st)
        return loss(ŷ, output), st_new, (;)
    end

    sum_loss = 0.0f0
    K = 0
    for (input, output) in data
        input, output = input |> device, output |> device
        _, loss_val, _, trainstate = Lux.Training.single_train_step!(adtype, lux_loss, (input, output), trainstate)
        sum_loss += loss_val * numobs(input)
        K += numobs(input)
    end
    return cpu(sum_loss / K), trainstate
end

# ---- Methods required to circumvent <: AbstractLuxLayer ----

# Adapted from Lux.jl source (helpers/training.jl) with the type annotation
# `model::AbstractLuxLayer` removed, since our estimators do not subtype AbstractLuxLayer.
# See https://github.com/LuxDL/Lux.jl/issues/1690
import Lux.Training: TrainState
using Lux.Training: get_allocator_cache
using Lux: ReactantCompatibleOptimisers
function TrainState(model, ps, st, optimizer::Optimisers.AbstractRule) #TODO remove this definition once https://github.com/LuxDL/Lux.jl/issues/1690 is resolved
    dev = get_device(ps)
    if dev isa ReactantDevice
        optimizer = ReactantCompatibleOptimisers.make_reactant_compatible(optimizer, dev)
    end
    st_opt = Optimisers.setup(optimizer, ps)
    return TrainState(
        nothing, nothing, get_allocator_cache(dev), model, ps, st, optimizer, st_opt, 0
    )
end

import LuxCore: preserves_state_type
preserves_state_type(l::NeuralEstimator) = true

end
