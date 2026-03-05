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