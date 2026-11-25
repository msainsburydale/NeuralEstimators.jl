@doc raw"""
	PointEstimator <: AbstractBayesEstimator
    PointEstimator(network; lower = nothing, upper = nothing, standardise_data = true, standardise_parameters = :auto)
    PointEstimator(summary_network, inference_network; lower = nothing, upper = nothing, standardise_data = true, standardise_parameters = :auto)
    PointEstimator(num_parameters, summary_network = identity; num_summaries, lower = nothing, upper = nothing, standardise_data = true, standardise_parameters = :auto, kwargs...)
A neural point estimator mapping data to a point summary of the posterior distribution.

The neural network can be provided in two ways:
- As a single `network` that maps data directly to the parameter space.
- As a `summary_network` that maps data to a vector of summary statistics, with the `inference_network` constructed internally based on `num_parameters` and `num_summaries`.

# Keyword arguments
- `num_summaries::Integer`: the number of summary statistics output by `summary_network`.
- `lower = nothing`, `upper = nothing`: lower and upper bounds of the parameters, each given as a vector of length $d$ or as a scalar that applies to all parameters, for the loss `mae` or a quantile loss only (see Parameter constraints below). Bounds may be infinite, and `nothing` leaves the parameters unbounded on that side.
- `standardise_data = true`: whether, or how, to standardise the data before they enter the neural network (see [`Standardiser`](@ref)). This does not change the quantity being estimated.
- `standardise_parameters = :auto`: whether, or how, to standardise the parameters, in which case the neural network estimates the standardised parameters and its output is mapped back to the original scale. By default (`:auto`), the parameters are standardised only if this leaves the estimator unchanged (see Standardisation below).
- `kwargs...`: additional keyword arguments passed to the MLP constructor of the inference network.

# Parameter constraints
Use `lower` and `upper` only with the loss `mae` (the default) or a quantile loss (see [`quantileloss`](@ref)). With any other loss (e.g., `mse` or [`tanhloss`](@ref)), impose the constraints within the neural network itself, with an output layer such as [`Compress`](@ref) or an activation function such as `softplus`, and do not give `lower` and `upper`.

Given `lower` and `upper`, the parameters are mapped to the real line by [`scaledlogit`](@ref) and then standardised (see Standardisation below), so that the neural network is trained to estimate the transformed parameters, and its estimates are mapped back by [`scaledlogistic`](@ref), so that they lie within the bounds. The loss is therefore evaluated on the transformed scale. Posterior medians and quantiles commute with increasing transformations of each parameter, such as `scaledlogit`, so the Bayes estimator under `mae` or a quantile loss is unchanged. Under other losses it is not: for example, under `mse`, the posterior mean ``E(\theta_i \mid \boldsymbol{Z})`` would become ``\mathrm{scaledlogistic}[E\{\mathrm{scaledlogit}(\theta_i) \mid \boldsymbol{Z}\}]``. Constraints other than bounds (e.g., an ordering of the parameters, or a covariance matrix that must be positive definite) are always imposed within the neural network (see [Output layers](@ref)).

# Standardisation
When the parameters are standardised, each parameter ``\theta_i`` is standardised as ``(\theta_i - \mu_i)/\sigma_i``, with the mean ``\mu_i`` and standard deviation ``\sigma_i`` computed from the training parameters (see [`Standardiser`](@ref)). Standardisation leaves the estimator unchanged if two conditions hold:
1. The loss is a sum over the parameters of terms that each depend on the estimate and the true value of a parameter only through their difference, as a positively homogeneous function of it (e.g., ``|x|^k``), so that standardisation only rescales each term. This holds for `mae` (posterior median), `mse` (posterior mean), and [`quantileloss`](@ref), but not for [`tanhloss`](@ref) or for [`kpowerloss`](@ref).
2. The neural network does not constrain its output, since an output layer that constrains the estimates (e.g., the activation function `softplus` or `exp`) would then constrain them on the standardised scale instead.

By default (`standardise_parameters = :auto`), the parameters are standardised exactly when [`train`](@ref) finds that both conditions hold. It checks the first numerically for the loss function it is given, and the second by requiring that the neural network ends in a `Dense` layer without activation function, with one output per parameter. This decision is made the first time that the estimator is trained, and is kept when it is trained further, whatever the loss, since the neural network then depends on it. Set `standardise_parameters = true` or `false` to override the decision.

# Examples
```julia
using NeuralEstimators, Flux, CairoMakie

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ N(0, 1) and σ ~ U(0, 1)
d, m = 2, 100  # dimension of θ and number of replicates
sampler(K) = NamedMatrix(μ = randn(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(m))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

# Neural network, an MLP mapping m inputs into d outputs
network = Chain(Dense(m, 64, gelu), Dense(64, 64, gelu), Dense(64, d))

# Initialise a neural point estimator, with the estimates constrained to the support of the prior, μ ∈ ℝ and σ ∈ (0, 1)
estimator = PointEstimator(network; lower = [-Inf, 0], upper = [Inf, 1])

# Train the estimator
estimator = train(estimator, sampler, simulator)

# Plot the risk history
plotrisk()

# Assess the estimator
θ_test = sampler(1000)
Z_test = simulator(θ_test)
assessment = assess(estimator, θ_test, Z_test)
bias(assessment)
rmse(assessment)
plot(assessment)

# Apply to observed data (here, simulated as a stand-in)
θ = sampler(1)           # ground truth (not known in practice)
Z = simulator(θ)         # stand-in for real observations
estimate(estimator, Z)   # point estimate
```
"""
@concrete struct PointEstimator <: AbstractBayesEstimator
    summary_network
    inference_network
    data_standardiser      # see Standardiser
    parameter_standardiser # the inference network outputs the standardised parameters, which are mapped back
    transform              # maps the parameters to the unbounded scale on which the networks are trained (identity without bounds)
    invtransform           # maps the estimates back to the parameter space
end

# The standardisers and bounds are applied outside the neural networks and saved separately (see _transformations), so
# they are kept out of the functor: they are never moved to a device or optimised, and the Flux state (and hence the
# loading of estimators saved by earlier versions) is unchanged
@functor PointEstimator (summary_network, inference_network)

# Constructor: summary network and inference network
function PointEstimator(summary_network, inference_network; lower::Union{Nothing, Real, AbstractVector} = nothing, upper::Union{Nothing, Real, AbstractVector} = nothing, standardise_data = true, standardise_parameters = :auto)
    parameter_standardiser = standardise_parameters === :auto ? _AutoStandardiser() : _standardiser(standardise_parameters)
    PointEstimator(summary_network, inference_network, _standardiser(standardise_data), parameter_standardiser, _transforms(lower, upper)...)
end

# Constructor: number of parameters and optional summary network => MLP inference network
function PointEstimator(num_parameters::Integer, summary_network = identity; num_summaries::Integer, lower = nothing, upper = nothing, standardise_data = true, standardise_parameters = :auto, kwargs...)
    summary_network = _resolvesummarynetwork(summary_network; kwargs...)
    backend = _backendof(summary_network)
    inference_network = MLP(num_summaries, num_parameters; backend = backend, _dropbackend(kwargs)...)
    inference_network = _matchsimplechains(inference_network, num_summaries, summary_network)
    @info "PointEstimator: num_summaries = $num_summaries."
    PointEstimator(summary_network, inference_network; lower, upper, standardise_data, standardise_parameters)
end

# Constructor: consistent argument ordering 
PointEstimator(summary_network, num_parameters::Integer; kwargs...) = PointEstimator(num_parameters, summary_network; kwargs...)

# Constructor: simple workflow, single network
function PointEstimator(network; lower = nothing, upper = nothing, standardise_data = true, standardise_parameters = :auto)
    @info "Constructing PointEstimator with a single network. Consider separating the summary and inference networks as PointEstimator(summary_network, inference_network), which enables additional functionality." # such as transfer learning and model-misspecification detection."
    backend = _backendof(network)
    PointEstimator(network, _identity_layer(backend); lower, upper, standardise_data, standardise_parameters)
end

# Parameter bounds: saved with the estimator and restored by loadestimator() (see _transformations)
_hasbounds(::PointEstimator) = true

# The number of outputs of the neural network if its final layer is linear (see _linearoutputs)
_linearoutputs(e::PointEstimator) = _linearoutputs(_is_identity(e.inference_network) ? e.summary_network : e.inference_network)

# Forward pass: Stateful (Flux)
# NB the original forward pass when the standardisers and the inverse transform apply no map (e.g., during training, when
# the data and parameters are mapped on the host); otherwise, the data are standardised and the estimates mapped back
# around it (see _NoMap)
const _PointEstimatorNoMap = PointEstimator{<:Any, <:Any, <:_NoMap, <:_NoMap, <:Any, <:Union{typeof(identity), _NoMap}}
function (e::_PointEstimatorNoMap)(Z)
    e.inference_network(_summarystatistics(e, Z))
end
(e::PointEstimator)(Z) = e.invtransform(_unstandardise(e.parameter_standardiser, _withoutstandardisers(e)(_standardisedata(e.data_standardiser, Z))))

# Forward pass: Stateless (Lux)
function (e::_PointEstimatorNoMap)(Z, ps, st)
    t, st_s = _summarystatistics(e, Z, ps.summary_network, st.summary_network)
    y, st_i = e.inference_network(t, ps.inference_network, st.inference_network)
    return y, (summary_network = st_s, inference_network = st_i)
end
function (e::PointEstimator)(Z, ps, st)
    y, st = _withoutstandardisers(e)(_standardisedata(e.data_standardiser, Z), ps, st)
    return e.invtransform(_unstandardise(e.parameter_standardiser, y)), st
end
