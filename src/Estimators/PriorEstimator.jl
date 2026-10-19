@doc raw"""
	PriorEstimator <: AbstractNeuralEstimator
	PriorEstimator(num_parameters::Integer; q = NormalisingFlow, kwargs...)
A neural estimator that approximates a probability distribution $p(\boldsymbol{\theta})$ from samples alone, without conditioning on data (i.e., unconditional density estimation). A typical use is to learn a prior distribution from samples that encode prior knowledge, such as parameter estimates from many previous analyses or posterior draws from an earlier analysis.

The distribution is approximated by `q`, which may be any of the in-built [Approximate distributions](@ref) (e.g., [`NormalisingFlow`](@ref), the default, or [`GaussianMixture`](@ref)). It is constructed given `num_parameters` and `num_summaries = 0` (i.e., it is unconditional), with any additional keyword arguments passed to the constructor of `q`. The parameters of a [`GaussianMixture`](@ref), [`Gaussian`](@ref), or [`SpikeAndSlab`](@ref) distribution are then learned directly, so that keyword arguments specifying a neural network (e.g., `depth` and `width`) apply only to a [`NormalisingFlow`](@ref).

The estimator is trained with [`train`](@ref), given either fixed training and validation samples, `train(estimator, θ_train, θ_val)`, or a function `sampler` such that `sampler(K)` returns `K` samples, `train(estimator, sampler)`, in which case new samples are drawn during training. In both cases, the samples are stored as a ``d`` × ``K`` matrix (or a [`NamedMatrix`](@ref)), where ``d`` is the dimension of the parameter vector.

Once trained, the learned distribution can be sampled with [`sample`](@ref sample(::PriorEstimator, ::Integer)) and its density evaluated with [`logdensity`](@ref). When `q` is a [`NormalisingFlow`](@ref), [`forward`](@ref) and [`inverse`](@ref) map between the parameter space and the latent space of the flow, in which the learned distribution is standard Gaussian.

!!! note "Standardisation"
    The parameters are not standardised internally. When they are on very different scales, standardise each parameter before training (e.g., by subtracting its mean and dividing by its standard deviation), and apply the inverse transformation to the samples.

!!! note "Single parameter"
    With a single parameter, the coupling layers of a [`NormalisingFlow`](@ref) have nothing to condition on, so the flow reduces to an affine transformation of its Gaussian base distribution. Hence, when `num_parameters = 1`, a [`GaussianMixture`](@ref) is used in place of a `NormalisingFlow`.

!!! note "Learning rate"
    Since the parameters of a [`GaussianMixture`](@ref), [`Gaussian`](@ref), or [`SpikeAndSlab`](@ref) distribution are learned directly, they require a larger learning rate than the weights of a neural network. Hence, in this case, [`train`](@ref) uses the optimiser `Adam(2e-2)` by default. 

# Examples
```julia
using NeuralEstimators, Flux

# Samples from the target distribution (e.g., existing parameter estimates)
sampler(K) = NamedMatrix(μ = randn(K), σ = rand(K))
θ_train = sampler(10_000)
θ_val   = sampler(10_000)

# Initialise and train the estimator
estimator = PriorEstimator(2)
estimator = train(estimator, θ_train, θ_val)

# Sample from the learned distribution and evaluate its density
θ = sample(estimator, 1000)
logdensity(estimator, θ)

# Map to and from the latent space of the normalising flow
U = forward(estimator, θ)  # approximately standard Gaussian
inverse(estimator, U)      # recovers θ
```
"""
@concrete struct PriorEstimator <: AbstractNeuralEstimator
    q
end

# Constructor: number of parameters => build approximate distribution automatically
function PriorEstimator(num_parameters::Integer; q = NormalisingFlow, kwargs...)
    # Convert string to type if needed
    if q isa String
        q = getfield(@__MODULE__, Symbol(q))
    end
    if num_parameters == 1 && q === NormalisingFlow
        @info "PriorEstimator: a NormalisingFlow with a single parameter reduces to a Gaussian distribution, so a GaussianMixture is used instead."
        q = GaussianMixture
    end
    PriorEstimator(q(num_parameters, 0; kwargs...))
end

# ---- Training ----

# The distribution is not conditioned on data (num_summaries = 0), so its summary statistics are an empty 0 × K matrix
_emptysummary(θ) = similar(θ, 0, size(θ, 2))
@non_differentiable _emptysummary(::Any)

_loss(estimator::PriorEstimator, loss = nothing) = (ld, θ) -> -mean(ld)

# No data are simulated (Z = nothing), so the parameters are both the input and the output placeholder
_inputoutput(estimator::PriorEstimator, ::Nothing, θ) = (θ, θ)
_inputoutput(estimator::PriorEstimator, Z, θ) = throw(ArgumentError("A PriorEstimator is trained with parameter samples alone, without data: use `train(estimator, θ_train, θ_val)` or `train(estimator, sampler)`"))

# Forward pass: Stateful (Flux)
(estimator::PriorEstimator)(θ) = _logdensity(estimator.q, θ, _emptysummary(θ))

# Forward pass: Stateless (Lux)
function (estimator::PriorEstimator)(θ, ps, st)
    ld, st_q = _logdensity(estimator.q, θ, _emptysummary(θ), ps.q, st.q)
    return ld, (q = st_q,)
end

# Both methods use the generic methods of train() with no data (Z = nothing). LuxEstimator is included
# so that an estimator containing Lux networks can be trained further after training or loading.
const _PriorEstimators = Union{PriorEstimator, LuxEstimator{<:PriorEstimator}}
function train(estimator::_PriorEstimators, θ_train::P, θ_val::P; optimiser::Optimisers.AbstractRule = _defaultoptimiser(estimator), kwargs...) where {P <: Union{AbstractMatrix, AbstractParameterSet}}
    train(estimator, θ_train, θ_val, nothing, nothing; optimiser = optimiser, kwargs...)
end
train(estimator::_PriorEstimators, sampler; optimiser::Optimisers.AbstractRule = _defaultoptimiser(estimator), kwargs...) = train(estimator, sampler, _nosimulator; optimiser = optimiser, kwargs...)
_nosimulator(θ, args...; kwargs...) = nothing

# Default optimiser. The parameters of a GaussianMixture (including a Gaussian) or SpikeAndSlab distribution are learned
# directly, since an unconditional distribution has no summary statistics to map to them (see _learnedconstant()). They
# need a larger learning rate than the weights of a neural network, many of which contribute to each of its outputs.
_defaultoptimiser(estimator::PriorEstimator) = _defaultoptimiser(estimator.q)
_defaultoptimiser(estimator::LuxEstimator) = _defaultoptimiser(estimator.estimator)
_defaultoptimiser(::Any) = Adam(5e-4) # the default of train()
_defaultoptimiser(::Union{GaussianMixture, SpikeAndSlab}) = Adam(2e-2)

# ---- Inference ----

# Applies `f` to parameters conditioned on the empty summary statistics. This is a callable
# struct so that _applywithdevice() can be reused for its batching, device placement, and test mode:
# for Flux, `f(q, θ, t)` returns an array; for Lux, `f(q, θ, t, ps, st)` returns a tuple whose first
# element is that array.
@concrete struct _PriorMap
    f
    estimator
end
(m::_PriorMap)(θ) = m.f(m.estimator.q, θ, _emptysummary(θ))
(m::_PriorMap)(θ, ps, st) = m.f(m.estimator.q, θ, _emptysummary(θ), ps.q, st.q)

# Maps parameters to the latent space of a normalising flow, dropping the log-determinant of the Jacobian
_tolatent(flow, θ, t) = first(forward(flow, θ, t))
_tolatent(flow, θ, t, ps, st) = forward(flow, θ, t, ps, st) # returns (U, log_det_J, st), of which _applywithdevice() keeps U

# Applies `f` to a d × K matrix or a d-dimensional vector θ, returning a matrix with K columns
function _applyprior(f, estimator::PriorEstimator, θ, args...; batchsize::Integer = 10_000, kwargs...)
    θ = _stripnames(_extractθ(θ))
    if θ isa AbstractVector
        θ = reshape(θ, :, 1)
    end
    _applywithdevice(_PriorMap(f, estimator), θ, args...; batchsize = batchsize, kwargs...)
end

function _checkflow(estimator::PriorEstimator, f)
    estimator.q isa NormalisingFlow || throw(ArgumentError("`$f` is only defined for a `PriorEstimator` whose approximate distribution is a `NormalisingFlow`"))
end

@doc raw"""
	sample(estimator::PriorEstimator, N::Integer; device = nothing, use_gpu = true)
	sample(estimator::PriorEstimator; N::Integer = 1000, device = nothing, use_gpu = true)
Draws `N` samples from the distribution learned by a [`PriorEstimator`](@ref), returned as a ``d`` × `N` matrix.

The number of samples `N` can be given as either a positional or a keyword argument.

The compute `device` (e.g., `cpu_device()` or `gpu_device()`) takes precedence over the flag `use_gpu`.

See also [`logdensity`](@ref) and [`inverse`](@ref).
"""
function sample(estimator::PriorEstimator, N::Integer; device = nothing, use_gpu::Bool = true)
    device = _resolvedevice(device = device, use_gpu = use_gpu, verbose = false)
    θ = sampleposterior(estimator.q, zeros(Float32, 0, 1), N; device = device)
    return θ[:, :, 1]
end
sample(estimator::PriorEstimator; N::Integer = 1000, kwargs...) = sample(estimator, N; kwargs...)

# Stateless (Lux)
function sample(estimator::PriorEstimator, N::Integer, ps, st; device = nothing, use_gpu::Bool = true)
    device = _resolvedevice(device = device, use_gpu = use_gpu, verbose = false)
    θ = sampleposterior(estimator.q, zeros(Float32, 0, 1), N, ps.q, st.q; device = device)
    return θ[:, :, 1]
end
# NB ps and st are typed, since a method with untyped arguments would be ambiguous with AbstractMCMC's sample(model, ::AbstractSampler, N)
sample(estimator::PriorEstimator, ps::NamedTuple, st::NamedTuple; N::Integer = 1000, kwargs...) = sample(estimator, N, ps, st; kwargs...)

@doc raw"""
	logdensity(estimator::PriorEstimator, θ; batchsize = 10_000, device = nothing, use_gpu = true)
Evaluates the log density of the distribution learned by a [`PriorEstimator`](@ref) at `θ`, a ``d`` × ``K`` matrix of parameter vectors or a single ``d``-dimensional parameter vector.

Returns a vector of ``K`` log densities, or a scalar when `θ` is a vector.

The compute `device` (e.g., `cpu_device()` or `gpu_device()`) takes precedence over the flag `use_gpu`.

See also [`sample`](@ref sample(::PriorEstimator, ::Integer)) and [`forward`](@ref).
"""
function logdensity end

# Stateful (Flux)
function logdensity(estimator::PriorEstimator, θ; kwargs...)
    ld = _applyprior(_logdensity, estimator, θ; kwargs...)
    return θ isa AbstractVector ? only(ld) : vec(ld)
end

# Stateless (Lux)
function logdensity(estimator::PriorEstimator, θ, ps, st; kwargs...)
    ld = _applyprior(_logdensity, estimator, θ, ps, st; kwargs...)
    return θ isa AbstractVector ? only(ld) : vec(ld)
end

@doc raw"""
	forward(estimator::PriorEstimator, θ; batchsize = 10_000, device = nothing, use_gpu = true)
For a [`PriorEstimator`](@ref) whose approximate distribution is a [`NormalisingFlow`](@ref), maps `θ`, a ``d`` × ``K`` matrix of parameter vectors or a single ``d``-dimensional parameter vector, to the latent space of the flow. The result has the same dimensions as `θ`.

If `θ` follows the learned distribution, the result $\boldsymbol{U}$ is standard Gaussian. Hence, for example, the squared norm of each column of $\boldsymbol{U}$ approximately follows a $\chi^2_d$ distribution, which can be used to flag parameter vectors that are unusual under the learned distribution.

Keyword arguments are as in [`logdensity`](@ref).

See also [`inverse`](@ref).
"""
function forward(estimator::PriorEstimator, θ; kwargs...)
    _checkflow(estimator, forward)
    U = _applyprior(_tolatent, estimator, θ; kwargs...)
    return θ isa AbstractVector ? vec(U) : U
end

# Stateless (Lux)
function forward(estimator::PriorEstimator, θ, ps, st; kwargs...)
    _checkflow(estimator, forward)
    U = _applyprior(_tolatent, estimator, θ, ps, st; kwargs...)
    return θ isa AbstractVector ? vec(U) : U
end

@doc raw"""
	inverse(estimator::PriorEstimator, U; batchsize = 10_000, device = nothing, use_gpu = true)
For a [`PriorEstimator`](@ref) whose approximate distribution is a [`NormalisingFlow`](@ref), maps `U`, a ``d`` × ``K`` matrix or a single ``d``-dimensional vector in the latent space of the flow, to the parameter space. The result has the same dimensions as `U`.

This is the inverse of [`forward`](@ref): if `U` is standard Gaussian, the result follows the learned distribution.

Keyword arguments are as in [`logdensity`](@ref).
"""
function inverse(estimator::PriorEstimator, U; kwargs...)
    _checkflow(estimator, inverse)
    θ = _applyprior(inverse, estimator, U; kwargs...)
    return U isa AbstractVector ? vec(θ) : θ
end

# Stateless (Lux)
function inverse(estimator::PriorEstimator, U, ps, st; kwargs...)
    _checkflow(estimator, inverse)
    θ = _applyprior(inverse, estimator, U, ps, st; kwargs...)
    return U isa AbstractVector ? vec(θ) : θ
end
