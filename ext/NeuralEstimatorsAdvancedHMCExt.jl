# NUTS posterior sampling for the RatioEstimator.
#
# Design notes:
#  * The target is log r(θ, Z) + log prior + log |J|, as a function of the
#    UNCONSTRAINED u, with θ = lower + (upper - lower) * sigmoid(u). The sigmoid map
#    maps to the prior box; log |J| is its Jacobian correction.
#  * The data summaries tz are computed once per
#    dataset Z and cached; each NUTS step then costs MLP passes (plus the ForwardDiff duals)
#  * The chain runs in Float64 (leapfrog likes Float64); the Float32 networks promote
#    automatically . Code can be performance tuned.
#  * One chain per data set. NUTS trajectories are adaptive in the number of leapfrog steps,
#    and to the best of my knowledge, there is no way to efficiently vectorize this. Recommend TRE sampling via Chebyshev approximations instead.

module NeuralEstimatorsAdvancedHMCExt

using NeuralEstimators
using NeuralEstimators: RatioEstimator, summarystatistics
using AdvancedHMC
using ForwardDiff
using LogDensityProblems

# The box map
_toθ(lower, upper, u) = lower .+ (upper .- lower) ./ (1 .+ exp.(-u))

struct _NRELogDensity{Fθ, Finf, Fp}
    tz::Matrix{Float32}       # cached data summaries, one column
    apply_θ::Fθ               # θ-matrix -> parameter summaries
    apply_inf::Finf           # vcat(tz, tθ) -> log-ratio
    lower::Vector{Float64}
    upper::Vector{Float64}
    logprior::Fp
end

LogDensityProblems.dimension(t::_NRELogDensity) = length(t.lower)
LogDensityProblems.capabilities(::Type{<:_NRELogDensity}) = LogDensityProblems.LogDensityOrder{0}()

function LogDensityProblems.logdensity(t::_NRELogDensity, u)
    θ = _toθ(t.lower, t.upper, u)
    # Jacobian of the sigmoid box map, written in θ: (b - a) s (1 - s) = (θ - a)(b - θ) / (b - a)
    logJ = sum(log.((θ .- t.lower) .* (t.upper .- θ) ./ (t.upper .- t.lower)))
    tθ = t.apply_θ(reshape(θ, :, 1))
    return only(t.apply_inf(vcat(t.tz, tθ))) + t.logprior(θ) + logJ
end

function _nuts_samples(summary_stats_Z, apply_θ, apply_inf, lower, upper, logprior, N, warmup)
    d = length(lower)
    lo, hi = Float64.(collect(lower)), Float64.(collect(upper))

    samples = map(1:size(summary_stats_Z, 2)) do k
        t = _NRELogDensity(summary_stats_Z[:, k:k], apply_θ, apply_inf, lo, hi, logprior)
        u0 = zeros(d)                                  # box midpoint after the transform

        # Standard AdvancedHMC: NUTS with multinomial sampling, generalised
        # no-U-turn and Stan-style joint step-size and mass-matrix adaptation
        metric = DiagEuclideanMetric(d)
        hamiltonian = Hamiltonian(metric, t, ForwardDiff)
        integrator = Leapfrog(find_good_stepsize(hamiltonian, u0))
        kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
        adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
        draws, stats = AdvancedHMC.sample(hamiltonian, kernel, u0, N + warmup, adaptor, warmup;
                                          verbose = false, progress = false)

        # Keep the last N regardless of whether the installed version drops warmup draws
        kept = draws[(end - N + 1):end]
        keptstats = stats[(end - N + 1):end]
        if !isempty(keptstats) && hasproperty(first(keptstats), :numerical_error)
            ndivergent = count(s -> s.numerical_error, keptstats)
            ndivergent > 0 && @warn "NUTS reported $ndivergent divergent transition(s) for data set $k; treat these samples with caution"
        end

        Float32.(_toθ(lo, hi, reduce(hcat, kept)))     # d × N, back on the box
    end

    return length(samples) == 1 ? samples[1] : samples
end

function NeuralEstimators._sampleposterior_hmc(
    estimator::RatioEstimator, Z;
    N::Integer,
    lower::AbstractVector,
    upper::AbstractVector,
    logprior::Function,
    warmup::Integer,
    kwargs...
)
    summary_stats_Z = summarystatistics(estimator, Z; kwargs...)
    apply_θ = θ -> estimator.summary_network_θ(θ)
    apply_inf = x -> estimator.inference_network(x)
    _nuts_samples(summary_stats_Z, apply_θ, apply_inf, lower, upper, logprior, N, warmup)
end

function NeuralEstimators._sampleposterior_hmc(
    estimator::RatioEstimator, Z, ps, st;
    N::Integer,
    lower::AbstractVector,
    upper::AbstractVector,
    logprior::Function,
    warmup::Integer,
    kwargs...
)
    summary_stats_Z = summarystatistics(estimator, Z, ps, st; kwargs...)
    apply_θ = θ -> first(estimator.summary_network_θ(θ, ps.summary_network_θ, st.summary_network_θ))
    apply_inf = x -> first(estimator.inference_network(x, ps.inference_network, st.inference_network))
    _nuts_samples(summary_stats_Z, apply_θ, apply_inf, lower, upper, logprior, N, warmup)
end

end
