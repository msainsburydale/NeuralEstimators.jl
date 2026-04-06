"""
	NeuralEstimator
An abstract supertype for all neural estimators.
"""
abstract type NeuralEstimator end

"""
	BayesEstimator <: NeuralEstimator
An abstract supertype for neural Bayes estimators.
"""
abstract type BayesEstimator <: NeuralEstimator end

# ---- Summary network helper functions ----

_has_summary_network(e) = hasfield(typeof(e), :summary_network)

"""
	summarynetwork(estimator::NeuralEstimator)
Returns the summary network of `estimator`.

See also [`summarystatistics`](@ref).
"""
summarynetwork(estimator::NeuralEstimator) = estimator.summary_network

"""
	setsummarynetwork(estimator::NeuralEstimator, network)
Returns a new estimator identical to `estimator` but with the summary network replaced by `network`. Useful for transfer learning.

Note that [`RatioEstimator`](@ref) has a second summary network for the parameters, accessible via `estimator.summary_network_θ`, which is not affected by this function.

See also [`summarynetwork`](@ref).
"""
function setsummarynetwork(estimator::NeuralEstimator, network)
    @set estimator.summary_network = network  # using Accessors
end

"""
	summarystatistics(estimator::NeuralEstimator, Z; batchsize = 32, device = nothing, use_gpu = true)
Computes learned summary statistics by applying the summary network of `estimator` to data `Z`.

If `Z` is a [`DataSet`](@ref) object, the learned summary statistics are concatenated with the
precomputed expert summary statistics stored in `Z.S`.

The device used for computation can be specified via `device` (e.g., `cpu_device()`, `gpu_device()`, or `reactant_device()`, the latter requiring Lux.jl) or inferred automatically by setting `use_gpu = true` (default) to use a GPU if one is available. The `device` argument takes priority over `use_gpu` if both are provided.

See also [`summarynetwork`](@ref).
"""
function summarystatistics end

# Stateful (Flux)
function summarystatistics(estimator::NeuralEstimator, Z; kwargs...)
    _applywithdevice(estimator.summary_network, Z; kwargs...)
end
function summarystatistics(estimator::NeuralEstimator, d::DataSet; kwargs...)
    t = _applywithdevice(estimator.summary_network, d.Z; kwargs...)
    isnothing(d.S) ? t : vcat(t, d.S)
end

# Stateless (Lux)
# NB these public functions assume that ps/st have not been subsetted
function summarystatistics(estimator::NeuralEstimator, Z, ps, st; kwargs...)
    _applywithdevice(estimator.summary_network, Z, ps.summary_network, st.summary_network; kwargs...)
end
function summarystatistics(estimator::NeuralEstimator, d::DataSet, ps, st; kwargs...)
    t = _applywithdevice(estimator.summary_network, d.Z, ps.summary_network, st.summary_network; kwargs...)
    isnothing(d.S) ? t : vcat(t, d.S)
end

# Internal helper function for applying the summary network to data, used in each estimator's forward pass.
# NOTE: summarystatistics (public) and _summarystatistics (internal) are distinct:
# - summarystatistics handles batching and device placement, for user-facing calls and computing summary statistics from frozen summary networks
# - _summarystatistics is used inside the forward pass during training, where batching and device placement are already handled by _risk
function summarystatistics end

# Stateful (Flux)
_summarystatistics(estimator, Z) = estimator.summary_network(Z)
_summarystatistics(estimator, d::DataSet) = vcat(estimator.summary_network(d.Z), d.S)
_summarystatistics(estimator, d::DataSet{Z, Nothing}) where {Z} = estimator.summary_network(d.Z)
_summarystatistics(estimator, s::Summaries) = s.S
_summarystatistics(estimator, d::DataSet{<:Summaries}) = vcat(d.Z.S, d.S)

# Stateless (Lux)
# NB these internal functions require ps/st to be subsetted already (i.e., ps = ps.summary_network)
_summarystatistics(estimator, Z, ps, st) = estimator.summary_network(Z, ps, st)
function _summarystatistics(estimator, d::DataSet, ps, st)
    t, st_new = estimator.summary_network(d.Z, ps, st)
    vcat(t, d.S), st_new
end
_summarystatistics(estimator, d::DataSet{Z, Nothing}, ps, st) where {Z} = estimator.summary_network(d.Z, ps, st)
_summarystatistics(estimator, s::Summaries, ps, st) = s.S, st
_summarystatistics(estimator, d::DataSet{<:Summaries}, ps, st) = vcat(d.Z.S, d.S), st
