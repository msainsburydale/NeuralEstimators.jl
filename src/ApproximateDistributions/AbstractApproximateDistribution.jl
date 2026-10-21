@doc raw"""
	AbstractApproximateDistribution
An abstract supertype for approximate distributions used in conjunction with a [`PosteriorEstimator`](@ref). 

Subtypes `A <: AbstractApproximateDistribution` must implement the following methods: 
 - `_logdensity(q::A, θ::AbstractMatrix, t::AbstractMatrix)` 
    - Used during training and therefore must support automatic differentiation.
    - `θ` is a `d × K` matrix of parameter vectors.
    - `t` is a `dstar × K` matrix of learned summary statistics obtained by applying the neural network in the `PosteriorEstimator` to a collection of `K` data sets. 
    - Should return a `1 × K` matrix, where each entry is the log density `log q(θₖ | tₖ)` for the `k`-th data set evaluated at the `k`-th parameter vector `θ[:, k]`.
 - `sampleposterior(q::A, t::AbstractMatrix, N::Integer)`
    - Used during inference and therefore does not need to be differentiable.
    - Should return a `d × N × K` array containing `N` samples from the approximate posterior `q(θ | tₖ)` for each of the `K` data sets.

A distribution constructed with `num_summaries = 0` is unconditional (e.g., when used in a [`PriorEstimator`](@ref)), and the methods above are then given an empty `0 × K` matrix `t`.
"""
abstract type AbstractApproximateDistribution end

@doc raw"""
    numdistributionalparams(q::AbstractApproximateDistribution)
    numdistributionalparams(estimator::PosteriorEstimator)
The number of distributional parameters (i.e., the dimension of the space ``\mathcal{K}`` of approximate-distribution parameters ``\boldsymbol{\kappa}``). 
"""
function numdistributionalparams end

# Generic keyword constructor for all AbstractApproximateDistribution subtypes;
# allows num_summaries to be passed as either a positional or keyword argument
(::Type{T})(num_parameters::Integer; num_summaries::Integer, kwargs...) where {T <: AbstractApproximateDistribution} =
    T(num_parameters, num_summaries; kwargs...)

"""Lower bound on scale parameters (standard deviations, Cholesky diagonals).

Scale parameters are parameterised by a softplus, which can underflow to zero during
training. A log-density involving log(σ²) = -Inf together with a quadratic form
(θ - μ)² / σ² = +Inf then evaluates to NaN, from which training cannot recover. This
floor is far below the scale of any parameter of practical interest, so it does not
constrain the fitted model.
"""
const MIN_SCALE = 1.0f-6

# ---- Unconditional distributions (num_summaries = 0) ----

# An unconditional distribution receives an empty 0 × K matrix of summary statistics. A network that would see only the
# summary statistics (e.g., that of a GaussianMixture) is then replaced by the distributional parameters themselves,
# learned directly: each head is a bias-free Dense layer applied to a row of ones, so that its weights are the
# distributional parameters (and its activation keeps them valid). The random initialisation of the weights (by
# glorot_uniform with either backend; see _fluxinit()) ensures that the components of a mixture are distinct, so that
# they can be learned. Parameters learned directly need a larger
# learning rate than the weights of a neural network (see _defaultoptimiser() in PriorEstimator.jl).
# NB a Dense layer with no inputs (and a bias) cannot be used instead: Lux initialises its biases to ±Inf (the default
# bias initialisation scales with 1/√fan_in), Flux initialises them to zero (so that the components of a mixture are
# identical and remain so during training), and Reactant cannot differentiate its empty weight matrix.
_onesrow(t) = fill!(similar(t, 1, size(t, 2)), 1)
@non_differentiable _onesrow(::Any)

# heads: pairs `out => activation`, whose outputs are concatenated
function _learnedconstant(B::Module, heads::Pair...)
    lux = get(Base.loaded_modules, _LUX_UUID, nothing)
    kwargs = _fluxinit(B, !isnothing(lux) && B === lux ? (use_bias = false,) : (bias = false,))
    return B.Chain(_onesrow, B.Parallel(vcat, (B.Dense(1 => out, activation; kwargs...) for (out, activation) in heads)...))
end


