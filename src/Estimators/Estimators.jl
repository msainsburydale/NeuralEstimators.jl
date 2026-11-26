"""
	AbstractNeuralEstimator
An abstract supertype for all neural estimators.
"""
abstract type AbstractNeuralEstimator end

"""
	AbstractBayesEstimator <: AbstractNeuralEstimator
An abstract supertype for neural Bayes estimators.
"""
abstract type AbstractBayesEstimator <: AbstractNeuralEstimator end

# ---- Identity layer ----

_identity_layer(backend::Module) = _identity_layer(Val(nameof(backend)))
_identity_layer(::Val{:Flux}) = identity  # plain Julia function, valid as a Flux layer
_is_identity(f) = f === identity || f isa typeof(identity) || (hasproperty(f, :func) && f.func === identity)

_resolvesummarynetwork(ψ; backend = nothing, kwargs...) = _is_identity(ψ) ? _identity_layer(_resolvebackend(backend)) : ψ
_dropbackend(kwargs) = Base.structdiff(NamedTuple(kwargs), (; backend = nothing))

# Hook for keeping an internally constructed `network` (with input dimension `in`) consistent with the
# summary network: identity here, with a method for Lux.SimpleChainsLayer summary networks in the Lux extension
_matchsimplechains(network, in::Integer, summary_network, others...) = network

# ---- Summary network helper functions ----

_has_summary_network(e) = hasfield(typeof(e), :summary_network)

"""
	summarynetwork(estimator::AbstractNeuralEstimator)
Returns the summary network of `estimator`.

See also [`summarystatistics`](@ref).
"""
summarynetwork(estimator::AbstractNeuralEstimator) = estimator.summary_network

"""
	setsummarynetwork(estimator::AbstractNeuralEstimator, network)
	setsummarynetwork(estimator::AbstractNeuralEstimator, original::AbstractNeuralEstimator)
Returns a new estimator identical to `estimator` but with the summary network replaced by `network`, or by the summary
network of another estimator `original`. Useful for transfer learning.

The first method leaves the data standardiser of `estimator` unchanged. The second method also copies the data standardiser
of `original` (see [`setdatastandardiser`](@ref)), so that the summary network sees data on the same scale. When both
estimators are a [`LuxEstimator`](@ref), it also copies the parameters and state of the summary network; training the new
estimator with `freeze_summary_network = true` (see [`train`](@ref)) leaves them fixed.

Note that [`RatioEstimator`](@ref) has a second summary network for the parameters, accessible via `estimator.summary_network_θ`, which is not affected by this function.

See also [`summarynetwork`](@ref) and [`datastandardiser`](@ref).
"""
function setsummarynetwork(estimator::AbstractNeuralEstimator, network)
    @set estimator.summary_network = network  # using Accessors
end
function setsummarynetwork(estimator::AbstractNeuralEstimator, original::AbstractNeuralEstimator)
    estimator = setsummarynetwork(estimator, summarynetwork(original))
    setdatastandardiser(estimator, original)
end

"""
	setdatastandardiser(estimator::AbstractNeuralEstimator, standardiser)
	setdatastandardiser(estimator::AbstractNeuralEstimator, original::AbstractNeuralEstimator)
Returns a new estimator identical to `estimator` but with its data standardiser replaced by `standardiser`, or by the data
standardiser of another estimator `original`.

`standardiser` is a value accepted by the keyword argument `standardise_data` of an estimator: `true`, `false` (equivalently
`nothing`), a [`Standardiser`](@ref), or, for data stored as [`DataAndSummaries`](@ref) objects, a named tuple
`(Z = …, S = …)` of these.

See also [`datastandardiser`](@ref) and [`setsummarynetwork`](@ref).
"""
function setdatastandardiser(estimator::AbstractNeuralEstimator, standardiser)
    hasfield(typeof(estimator), :data_standardiser) || throw(ArgumentError("$(nameof(typeof(estimator))) has no data standardiser"))
    @set estimator.data_standardiser = _standardiser(standardiser)
end
function setdatastandardiser(estimator::AbstractNeuralEstimator, original::AbstractNeuralEstimator)
    setdatastandardiser(estimator, _unmark(datastandardiser(original)))
end

"""
	datastandardiser(estimator::AbstractNeuralEstimator)
Returns the data standardiser of `estimator` (see [`Standardiser`](@ref)): a `Standardiser`, `nothing` if the data are
not standardised, or a named tuple `(Z = …, S = …)` of these (for data stored as [`DataAndSummaries`](@ref) objects).

To reuse the summary network of `estimator` in another estimator (e.g., for transfer learning), copy it together with this
standardiser using [`setsummarynetwork`](@ref), or pass the standardiser to [`setdatastandardiser`](@ref), so that the
network sees data on the same scale.
"""
datastandardiser(estimator::AbstractNeuralEstimator) = estimator.data_standardiser

"""
	summarystatistics(estimator::AbstractNeuralEstimator, Z; batchsize = 32, device = nothing, use_gpu = true)
Computes learned summary statistics by applying the summary network of `estimator` to data `Z`, after standardising the
data with the data standardiser of `estimator` (see [`Standardiser`](@ref)).

If `Z` is a [`DataAndSummaries`](@ref) object, the learned summary statistics are concatenated with the
precomputed expert summary statistics stored in `Z.S` (standardised if the data standardiser of `estimator`
standardises them). That is, the result is the input of the inference network.

The device used for computation can be specified via `device` (e.g., `cpu_device()`, `gpu_device()`, or `reactant_device()`, the latter requiring Lux.jl) or inferred automatically by setting `use_gpu = true` (default) to use a GPU if one is available. The `device` argument takes priority over `use_gpu` if both are provided.

See also [`summarynetwork`](@ref).
"""
function summarystatistics end

# NB the data are standardised once, on the host, before being split into batches and moved to the device

# Stateful (Flux)
function summarystatistics(estimator::AbstractNeuralEstimator, Z; kwargs...)
    Z = _standardisedata(_unmark(_datastandardiser(estimator)), Z)
    _applywithdevice(estimator.summary_network, Z; kwargs...)
end
function summarystatistics(estimator::AbstractNeuralEstimator, d::DataAndSummaries; kwargs...)
    d = _standardisedata(_unmark(_datastandardiser(estimator)), d)
    t = _applywithdevice(estimator.summary_network, d.Z; kwargs...)
    vcat(t, d.S)
end

#TODO Perhaps the cleanest approach is to also define a Summaries type, which is always passed through the .expert_network. Expect users to always provide expert summaries as Summaries(T) or DataAndSummaries(T), and the expert_network is then always applied consistently. Users can provide their own expert_network (allows structured statistics, like fields, I guess).
#TODO Might also be clearer to write "embedding_network" rather than "summary_network"
# function summarystatistics(estimator::AbstractNeuralEstimator, d::DataAndSummaries; kwargs...)
#     t = _applywithdevice(estimator.summary_network, d.Z; kwargs...)
#     s = _applywithdevice(estimator.expert_network, d.S; kwargs...)
#     # isnothing(d.S) ? t : vcat(t, d.S) # #TODO Don't think it makes sense to allow DataAndSummaries to have an empty field
#     vcat(t, d.S)
# end

# Stateless (Lux)
# NB these public functions assume that ps/st have not been subsetted
function summarystatistics(estimator::AbstractNeuralEstimator, Z, ps, st; kwargs...)
    Z = _standardisedata(_unmark(_datastandardiser(estimator)), Z)
    _applywithdevice(estimator.summary_network, Z, ps.summary_network, st.summary_network; kwargs...)
end
function summarystatistics(estimator::AbstractNeuralEstimator, d::DataAndSummaries, ps, st; kwargs...)
    d = _standardisedata(_unmark(_datastandardiser(estimator)), d)
    t = _applywithdevice(estimator.summary_network, d.Z, ps.summary_network, st.summary_network; kwargs...)
    vcat(t, d.S)
end

# Internal helper function for applying the summary network to data, used in each estimator's forward pass.
# NOTE: summarystatistics (public) and _summarystatistics (internal) are distinct:
# - summarystatistics handles batching and device placement, for user-facing calls and computing summary statistics from frozen summary networks
# - _summarystatistics is used inside the forward pass during training, where batching and device placement are already handled by _risk
# The data are standardised before _summarystatistics: on the host during training (see _Prestandardised), and otherwise by
# the forward pass of each estimator (see _NoMap)
function summarystatistics end

# Stateful (Flux)
_summarystatistics(estimator, Z) = estimator.summary_network(Z)
_summarystatistics(estimator, d::DataAndSummaries) = vcat(estimator.summary_network(d.Z), d.S)
_summarystatistics(estimator, d::DataAndSummaries{<:Summaries}) = vcat(d.Z.S, d.S)
_summarystatistics(estimator, s::Summaries) = s.S

# Stateless (Lux)
# NB: These internal functions require ps/st to be subsetted already (i.e., ps = ps.summary_network)
# NB: Enzyme's static analyis doesn't like it when we extract data from a custom struct (e.g., Summaries or DataAndSummaries) and then 
#     concatenate it with something else later. So, here we materialise (copy) to break Enzyme's trace.
_summarystatistics(estimator, Z, ps, st) = estimator.summary_network(Z, ps, st)
function _summarystatistics(estimator, d::DataAndSummaries, ps, st)
    t, st_new = estimator.summary_network(d.Z, ps, st)
    vcat(t, copy(d.S)), st_new # NB: materialise (copy) to break Enzyme's trace
end
_summarystatistics(estimator, d::DataAndSummaries{<:Summaries}, ps, st) = vcat(copy(d.Z.S), copy(d.S)), st # NB: materialise (copy) to break Enzyme's trace
_summarystatistics(estimator, s::Summaries, ps, st) = s.S, st
