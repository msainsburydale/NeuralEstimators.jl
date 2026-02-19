@doc raw"""
	ApproximateDistribution
An abstract supertype for approximate posterior distributions used in conjunction with a [`PosteriorEstimator`](@ref). 

Subtypes `A <: ApproximateDistribution` must implement the following methods: 
 - `logdensity(q::A, θ::AbstractMatrix, tz::AbstractMatrix)` 
    - Used during training and therefore must support automatic differentiation.
    - `θ` is a `d × K` matrix of parameter vectors.
    - `tz` is a `dstar × K` matrix of summary statistics obtained by applying the neural network in the `PosteriorEstimator` to a collection of `K` data sets. 
    - Should return a `1 × K` matrix, where each entry is the log density `log q(θₖ | tₖ)` for the `k`-th data set evaluated at the `k`-th parameter vector `θ[:, k]`.
 - `sampleposterior(q::A, tz::AbstractMatrix, N::Integer)`
    - Used during inference and therefore does not need to be differentiable.
    - Should return a `Vector` of length `K`, where each element is a `d × N` matrix containing `N` samples from the approximate posterior `q(θ | tₖ)` for the `k`-th data set.
"""
abstract type ApproximateDistribution end

@doc raw"""
    numdistributionalparams(q::ApproximateDistribution)
    numdistributionalparams(estimator::PosteriorEstimator)
The number of distributional parameters (i.e., the dimension of the space ``\mathcal{K}`` of approximate-distribution parameters ``\boldsymbol{\kappa}``). 
"""
function numdistributionalparams end

# Catch the case that tz is a vector
sampleposterior(q::ApproximateDistribution, tz::AbstractVector, N::Integer; kwargs...) = sampleposterior(q, reshape(tz, :, 1), N; kwargs...)