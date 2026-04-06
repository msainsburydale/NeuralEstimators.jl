"""
	AbstractTrainState
An abstract supertype for training states defined in NeuralEstimators.
"""
abstract type AbstractTrainState end

@concrete struct FluxTrainState <: AbstractTrainState
    model::Any
    optimizer::Any
    optimizer_state::Any
end

# Simple extension to the `adjust!` API
function Optimisers.adjust!(ts::AbstractTrainState, eta::Real)
    Optimisers.adjust!(ts.optimizer_state, eta)
    ts = @set ts.optimizer = Optimisers.adjust(ts.optimizer, eta)
    return ts
end

function Optimisers.adjust!(ts::AbstractTrainState; kwargs...)
    Optimisers.adjust!(ts.optimizer_state; kwargs...)
    ts = @set ts.optimizer = Optimisers.adjust(ts.optimizer; kwargs...)
    return ts
end

function Optimisers.adjust(ts::AbstractTrainState, eta::Real)
    ts = @set ts.optimizer_state = Optimisers.adjust(ts.optimizer_state, eta)
    ts = @set ts.optimizer = Optimisers.adjust(ts.optimizer, eta)
    return ts
end

function Optimisers.adjust(ts::AbstractTrainState; kwargs...)
    ts = @set ts.optimizer_state = Optimisers.adjust(ts.optimizer_state; kwargs...)
    ts = @set ts.optimizer = Optimisers.adjust(ts.optimizer; kwargs...)
    return ts
end
