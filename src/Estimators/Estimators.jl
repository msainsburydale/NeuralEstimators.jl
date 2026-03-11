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

"""
	summarynetwork(estimator::NeuralEstimator)
Returns the summary network of `estimator`.

See also [`summarystatistics`](@ref).
"""
summarynetwork(estimator::NeuralEstimator) = estimator.summary_network

# Replace the summary network (for transfer learning)
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
	summarystatistics(estimator::NeuralEstimator, Z; batchsize::Integer = 32, use_gpu::Bool = true)
Computes learned summary statistics by applying the summary network of `estimator` to data `Z`.

If `Z` is a [`DataSet`](@ref) object, the learned summary statistics are concatenated with the
precomputed expert summary statistics stored in `Z.S`.

See also [`summarynetwork`](@ref).
"""
function summarystatistics(estimator::NeuralEstimator, Z; kwargs...)
    _applywithdevice_inference(estimator.summary_network, Z; kwargs...)
end
function summarystatistics(estimator::NeuralEstimator, d::DataSet; kwargs...)
    t = _applywithdevice_inference(estimator.summary_network, d.Z; kwargs...)
    isnothing(d.S) ? t : vcat(t, d.S)
end

# Internal version of summarystatistics used during training. Skips the DeepSet
# input convenience check since data format is guaranteed to be correct at that point,
# and avoids unnecessary overhead in the training loop.
function _precomputesummaries(estimator::NeuralEstimator, Z; kwargs...)
    _applywithdevice(estimator.summary_network, Z; kwargs...)
end
function _precomputesummaries(estimator::NeuralEstimator, d::DataSet; kwargs...)
    t = _applywithdevice(estimator.summary_network, d.Z; kwargs...)
    isnothing(d.S) ? t : vcat(t, d.S)
end

# ---- Summaries wrapper type ----

"""
    Summaries(S::AbstractMatrix)

A thin wrapper around a matrix of precomputed summary statistics. Used internally
during training to signal that the summary network has already been applied to the
data, so that `_summarystatistics` can short-circuit and return the matrix directly
rather than re-running the (frozen) summary network on every forward pass.
"""
struct Summaries{T <: AbstractMatrix}
    S::T
end

Base.length(s::Summaries) = size(s.S, 2)
Base.getindex(s::Summaries, i) = Summaries(s.S[:, i])

# ---- _summarystatistics ----

# Internal helper function for applying the summary network to data, used in each estimator's forward pass.

# NOTE: summarystatistics (public) and _summarystatistics (internal) are distinct:
# - summarystatistics handles batching and device placement, for user-facing calls
# - _summarystatistics is used inside the forward pass during training, where batching and device placement are already handled by _risk

# Default: run the summary network on raw data
_summarystatistics(estimator, Z) = estimator.summary_network(Z)

# DataSet with expert summaries: run summary network and concatenate
_summarystatistics(estimator, d::DataSet) = vcat(estimator.summary_network(d.Z), d.S)

# DataSet without expert summaries: run summary network only
_summarystatistics(estimator, d::DataSet{Z, Nothing}) where {Z} = estimator.summary_network(d.Z)

# Precomputed summaries: short-circuit — the summary network is frozen and has
# already been applied, so just return the stored matrix directly
_summarystatistics(estimator, s::Summaries) = s.S

# DataSet where the Z field has already been replaced by Summaries: concatenate
# precomputed learned summaries with the stored expert summaries
_summarystatistics(estimator, d::DataSet{<:Summaries}) = vcat(d.Z.S, d.S)
