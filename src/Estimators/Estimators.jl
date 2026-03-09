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
	summarystatistics(estimator::NeuralEstimator, Z; batchsize::Integer = 32, use_gpu::Bool = true)
Computes learned summary statistics by applying the summary network of `estimator` to data `Z`.

If `Z` is a [`DataSet`](@ref) object, the learned summary statistics are concatenated with the
precomputed expert summary statistics stored in `Z.S`.

See also [`summarynetwork`](@ref).
"""
function summarystatistics(estimator::NeuralEstimator, Z; kwargs...)
    _applywithdevice(estimator.summary_network, Z; kwargs...)
end
function summarystatistics(estimator::NeuralEstimator, d::DataSet; kwargs...)
    t = _applywithdevice(estimator.summary_network, d.Z; kwargs...)
    isnothing(d.S) ? t : vcat(t, d.S)
end

# Internal helper function for applying the summary network to data, used in each estimator's forward pass 
_summarystatistics(estimator, Z) = estimator.summary_network(Z)
_summarystatistics(estimator, d::DataSet) = vcat(estimator.summary_network(d.Z), d.S)
_summarystatistics(estimator, d::DataSet{Z, Nothing}) where Z = estimator.summary_network(d.Z)

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
