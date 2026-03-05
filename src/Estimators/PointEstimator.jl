"""
	PointEstimator <: BayesEstimator
    PointEstimator(network)
A neural point estimator, where the neural `network` is a mapping from the sample space to the parameter space.
"""
struct PointEstimator{N} <: BayesEstimator
    network::N
end
(estimator::PointEstimator)(Z) = estimator.network(Z)