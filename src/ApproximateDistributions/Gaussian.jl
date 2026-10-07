@doc raw"""
    Gaussian(d::Integer, num_summaries::Integer; diagonal::Bool = false, kwargs...)
A Gaussian distribution for amortised inference with a [`PosteriorEstimator`](@ref), where `d` is the dimension of the parameter vector. 

The density of the distribution is: 
```math 
q(\boldsymbol{\theta}; \boldsymbol{\kappa}) = \mathcal{N}(\boldsymbol{\theta}; \boldsymbol{\mu}, \boldsymbol{\Sigma}), 
```
where the parameters $\boldsymbol{\kappa}$ comprise the mean vector $\boldsymbol{\mu}$ and the parameters of the covariance matrix $\boldsymbol{\Sigma}$, which is dense by default.

This is a convenience constructor for a [`GaussianMixture`](@ref) with a single component and, by default, a dense covariance matrix (`diagonal = false`). See [`GaussianMixture`](@ref) for the parameterisation of the covariance matrix, and for the neural network that maps the (learned) summary statistics to the distributional parameters when using a `Gaussian` distribution as the approximate distribution of a [`PosteriorEstimator`](@ref).

# Keyword arguments
- `diagonal::Bool = false`: whether the covariance matrix is diagonal (`true`) or dense (`false`).
- `kwargs`: additional keyword arguments passed to [`GaussianMixture`](@ref) (e.g., `depth`, `width`, `activation`), with the exception of `num_components`, which is fixed to one.
"""
Gaussian(d::Integer, num_summaries::Integer; diagonal::Bool = false, kwargs...) = GaussianMixture(d, num_summaries; diagonal = diagonal, kwargs..., num_components = 1)

# Allows num_summaries to be passed as a keyword argument, as for the subtypes of AbstractApproximateDistribution
Gaussian(d::Integer; num_summaries::Integer, kwargs...) = Gaussian(d, num_summaries; kwargs...)
