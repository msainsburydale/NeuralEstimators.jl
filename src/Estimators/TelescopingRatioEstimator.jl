@doc raw"""
	TelescopingRatioEstimator <: AbstractNeuralEstimator
	TelescopingRatioEstimator(summary_network, num_parameters; num_summaries, sampler, kwargs...)
A neural estimator that factorises the likelihood-to-evidence ratio sequentally accross parameter diemsnions, with
one classifier MLP head per parameter coordinate. Currently, the implementation only supports the case in which
the prior factorizes p(\boldsymbol{\theta}) = \prod_{i=1}^d p(\theta^i). In this case, we have that
```math
r(\boldsymbol{Z}, \boldsymbol{\theta}) = \prod_{i=1}^{d} r_i(\boldsymbol{Z}, \theta_i \mid \theta_1, \dots, \theta_{i-1}),
```
and head $i$ discriminates the true $\theta_i$ from a fresh prior draw, holding the
prefix $\theta_1, \dots, \theta_{i-1}$ fixed and conditioning on it. Summing the per-head
log-ratios recovers the joint log-ratio estimated by [`RatioEstimator`](@ref), but the
factorisation also exposes each one-dimensional conditional
$p(\theta_i \mid \theta_1, \dots, \theta_{i-1}, \boldsymbol{Z})$ individually, which allows for efficient posterior sampling via
approximate Chebyshev polynomial fitting.

Indeed, [`sampleposterior`](@ref) draws $\theta_1 \mid \boldsymbol{Z}$, then $\theta_2 \mid \theta_1, \boldsymbol{Z}$, 
and so on, approximating each conditional with a Chebyshev polynomial and inverting its CDF exactly.
No MCMC, no grid, and the cost grows linearly rather than exponentially in $d$.

The data are summarised by `summary_network`; the heads are a `MultiHeadMLP` with growing
inputs, head $i$ taking the summaries together with $\theta_1, \dots, \theta_i$.

# Keyword arguments
- `num_summaries::Integer`: the number of summaries output by `summary_network`. Must match the output dimension of `summary_network`.
- `sampler::Function`: a function returning `K` independent draws from the prior as a `d × K` matrix, the same function passed as `sampler` to [`train`](@ref). Required: the independent pairs of each head are formed from fresh prior draws of the focus coordinate.
- `kwargs...`: additional keyword arguments passed to the `MultiHeadMLP` constructor for the heads.

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d, m = 2, 10  # dimension of θ and number of replicates
sampler(K) = NamedMatrix(μ = rand(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(m))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

# Neural network
num_summaries = 3d
summary_network = Chain(Dense(m, 32, gelu), Dense(32, 16, gelu), Dense(16, num_summaries))

# Initialise the estimator
estimator = TelescopingRatioEstimator(summary_network, d; num_summaries = num_summaries, sampler = sampler)

# Train the estimator
estimator = train(estimator, sampler, simulator, K = 10000)

# Generate "observed" data
θ = sampler(1)
z = simulator(θ)

# Evaluation and sequential posterior sampling
grid = expandgrid(0:0.01:1, 0:0.01:1)'  # evaluation points for the log-ratio
logratio(estimator, z; grid = grid)     # log of likelihood-to-evidence ratios
logposterior(estimator, grid, z; lower = [0.0, 0.0], upper = [1.0, 1.0])  # log posterior density
sampleposterior(estimator, z; lower = [0.0, 0.0], upper = [1.0, 1.0])     # posterior sample
```
"""
@concrete struct TelescopingRatioEstimator <: AbstractNeuralEstimator
    summary_network # summary network for data Z (called summary_network for consistency with other estimators)
    heads           # MultiHeadMLP with growing inputs; head i takes the summaries and the first i coordinates θ1,..., θi of θ
    sampler         # same sampler that generates theta
end
 
# The sampler is intentionally kept out of the network, same as in RatioEstimator.jl
# e.g., to prevent silent transformations from cpu to gpu before _inputoutput invokes it
# e.g., to not make the optimizer compute gradients w.r.t. sampler
@functor TelescopingRatioEstimator (summary_network, heads)
 
# Constructor: summary network, number of parameters, number of summaries => one classifier head per parameter
function TelescopingRatioEstimator(
    summary_network, num_parameters::Integer, num_summaries::Integer;
    sampler::Function,
    kwargs...
)
    backend = _backendof(summary_network)
    heads = MultiHeadMLP(num_summaries, 1, num_parameters; backend = backend, growing = true, output_activation = identity, kwargs...)
    @info "TelescopingRatioEstimator: num_summaries = $num_summaries, num_heads = $num_parameters."
    TelescopingRatioEstimator(summary_network, heads, sampler)
end
 
# Constructor: keyword num_summaries
TelescopingRatioEstimator(summary_network, num_parameters::Integer; num_summaries::Integer, kwargs...) = TelescopingRatioEstimator(summary_network, num_parameters, num_summaries; kwargs...)
 
# Number of heads (= number of parameters); both Flux and Lux store the branches of Parallel in `layers`; 
_numheads(estimator::TelescopingRatioEstimator) = length(estimator.heads.layers)
 
# Evaluate a single head, without running the other heads through Parallel, which would be wastfeul, particularly when
# Used by the sequential posterior sampler, which needs head i alone at many synthetic inputs.
# Flux stores the branches in a Tuple, Lux in a NamedTuple; ps/st mirror the NamedTuple order,
# positional indexing lines up across all three.
# to double check with Matt around here, and also flag the summary statsitics being transafered to CPU;
# we have the kernel abstraction that works faster on GPU
# important becaues it gives speed of sampling close to normalizing flows
_head(estimator::TelescopingRatioEstimator, i::Integer, X) = estimator.heads.layers[i](X)
_head(estimator::TelescopingRatioEstimator, i::Integer, X, ps, st) = first(estimator.heads.layers[i](X, ps.heads[i], st.heads[i]))
 
function _inputoutput(estimator::TelescopingRatioEstimator, Z, θ)
    d, K = size(θ)
    @assert d == _numheads(estimator) "θ has $d rows but the estimator has $(_numheads(estimator)) heads"
 
    # Fresh prior draws: row i provides the focus coordinate of head i's independent pairs
    θ̃ = _stripnames(_extractθ(estimator.sampler(K)))
    @assert size(θ̃) == (d, K) "sampler(K) must return a $d × $K parameter matrix; got $(size(θ̃))"
 
    # Binary class labels: rows 1:d for the dependent pairs, rows d+1:2d for the independent pairs.
    # Positives and negatives are stacked along rows (not columns) so that all components of the
    # input share numobs = K, as required by the data loader; this also keeps each (Z, θ, θ̃)
    # aligned under shuffling, which is importntt bc independent pairs reuse the
    # parameter prefix (i.e. first coodinates) of their dependent prefix.
    output = vcat(ones(Float32, d, K), zeros(Float32, d, K))
 
    input = (Z, θ, θ̃)
    return input, output
end
 
_loss(estimator::TelescopingRatioEstimator, loss = nothing) = logitbinarycrossentropy

# Let θ1,...,θd, Z be the data. Head i is a classifier that inputs the data summaries,
# the prefix θ1,...,θi-1 which it hold fixed, and a new value for θi
# this last value is either the true one θi (label 1), or a fresh draw from the prior (label = 0).
# NB: map |> Tuple rather than ntuple: Zygote has no rule for ntuple, whose Base implementation
#     switches from unrolled tuples to a generator for n > 10; map over a range is reliably
#     differentiable for any number of heads
_headinputs(tz, θ) = map(i -> vcat(tz, θ[1:i, :]), 1:size(θ, 1)) |> Tuple
_headinputs(tz, θ, θ̃) = map(i -> vcat(tz, θ[1:(i - 1), :], θ̃[i:i, :]), 1:size(θ, 1)) |> Tuple
 
# Forward pass: Stateful (Flux)
# Returns the d × K matrix of per-head logits; the total log-ratio is the sum over rows
function (estimator::TelescopingRatioEstimator)(Z, θ)
    tz = _summarystatistics(estimator, Z)
    estimator.heads(_headinputs(tz, θ))
end
 
# Training forward pass: 2d × K logits, matching the labels constructed in _inputoutput
function (estimator::TelescopingRatioEstimator)(Z, θ, θ̃)
    tz = _summarystatistics(estimator, Z)
    pos = estimator.heads(_headinputs(tz, θ))
    neg = estimator.heads(_headinputs(tz, θ, θ̃))
    vcat(pos, neg)
end
 
# Forward pass: Stateless (Lux)
function (e::TelescopingRatioEstimator)(Z, θ, ps, st)
    tz, st_s = _summarystatistics(e, Z, ps.summary_network, st.summary_network)
    tz = tz |> copy # materialise to break Enzyme's trace (see the note in RatioEstimator.jl) # to check with Matt
    logits, st_h = e.heads(_headinputs(tz, θ), ps.heads, st.heads)
    return logits, (summary_network = st_s, heads = st_h)
end
 
function (e::TelescopingRatioEstimator)(Z, θ, θ̃, ps, st)
    tz, st_s = _summarystatistics(e, Z, ps.summary_network, st.summary_network)
    tz = tz |> copy # materialise to break Enzyme's trace (see the note in RatioEstimator.jl) # to check with Matt
    pos, st_h = e.heads(_headinputs(tz, θ), ps.heads, st.heads)
    neg, st_h = e.heads(_headinputs(tz, θ, θ̃), ps.heads, st_h)
    return vcat(pos, neg), (summary_network = st_s, heads = st_h)
end
 
# Tuple methods used internally during training
# bridge between the generic training loop and the TRE's specific forward-pass
(estimator::TelescopingRatioEstimator)(input::Tuple) = estimator(input...)
(estimator::TelescopingRatioEstimator)(input::Tuple, ps, st) = estimator(input..., ps, st)
 
# ---- Inference: Stateful (Flux) ----
 
function logratio(estimator::TelescopingRatioEstimator, Z; grid, kwargs...)
    grid = f32(grid)
    summary_stats_Z = summarystatistics(estimator, Z; kwargs...)
    _gridlogratio(estimator, summary_stats_Z, grid)
end
 
function _gridlogratio(estimator::TelescopingRatioEstimator, summary_stats_Z, grid::AbstractMatrix)
    K = size(summary_stats_Z, 2)    # number of data sets
    G = size(grid, 2)               # number of grid points
    # Repeat so that the summaries and the grid both have GxK columns
    summary_stats_Z_rep = repeat(summary_stats_Z, inner = (1, G))
    grid_rep = repeat(grid, outer = (1, K))
    logits = estimator.heads(_headinputs(summary_stats_Z_rep, grid_rep))
    log_ratios = sum(logits, dims = 1)  # total log-ratio: sum of the per-head conditional log-ratios
    return permutedims(reshape(log_ratios, G, K))  # K × G matrix
end

#NB The sampling from this Chebyshev approximation is exact up to machine precision, not the approximation itself. The latter
# is harder to guarantee, although these approximations have excellent convergence properties, e.g.,
# super-polynomial if the activation functions are analytic.
# NB The first coordinate uses a single envelope for all `N` samples; every other coordinate
# must build one envelope per sample, because each sample has a different prefix;
# the procedure is fitted and inverted in a single batched pass for efficiency.
function sampleposterior(
    estimator::TelescopingRatioEstimator, Z;
    lower::AbstractVector,
    upper::AbstractVector,
    N::Integer = 1000,
    degree::Integer = 128,
    logpriors::Union{Nothing, AbstractVector} = nothing,
    batchsize::Integer = 1,
    kwargs...
)
    summary_stats_Z = summarystatistics(estimator, Z; kwargs...)
    headfun = (i, X) -> _head(estimator, i, X)
    _sampleposterior_blocks(estimator, headfun, summary_stats_Z, lower, upper, N, degree, logpriors, batchsize)
end
 
# Chunk the data sets and run each block through the fused core. batchsize = 1 defaults to
# processing the data sets in a one-at-a-time fashion.
function _sampleposterior_blocks(estimator::TelescopingRatioEstimator, headfun, summary_stats_Z, lower, upper, N::Integer, degree::Integer, logpriors, batchsize::Integer)
    K = size(summary_stats_Z, 2)
    samples = Vector{Matrix{eltype(summary_stats_Z)}}(undef, K)
    for block in Iterators.partition(1:K, batchsize)
        θdrawn, _, _ = _sequential_core(estimator, headfun, summary_stats_Z[:, block], lower, upper, N, nothing, degree, logpriors)
        for (j, k) in enumerate(block)
            samples[k] = θdrawn[:, ((j - 1) * N + 1):(j * N)]
        end
    end
    return K == 1 ? samples[1] : samples
end
 
# Backend-agnostic sequential core shared by sampleposterior, logposterior, and the
# coverage checks in calibration.jl; `headfun(i, X)` evaluates head i on the input
# matrix X and returns a 1 × size(X, 2) matrix of logits. All cheb envelope conventions
# live here and nowhere else: node ordering, prefix-major column layout, max-shift.
#
# Processes a block of B data sets (columns of tzs) in one fused pass, for improved efficiency. Per data set
# there are N drawn columns and F fixed columns:
#   * drawn columns are posterior draws, sampled coordinate by coordinate;
#   * fixed columns are never sampled; their coordinates are given by θfixed (d × F*B,
#     data-set-major) and only their log-density under the sampling law is accumulated.
# logposterior is the N = 0 case; plain sampling is F = 0; the joint coverage check
# uses N = M draws plus F = 1 (the true parameter).
#
# Global column layout, drawn block then fixed block, both data-set-major:
#     [ds1 draws ... dsB draws | ds1 fixed ... dsB fixed]
# so envelopes, head inputs, uniforms and outputs all reshape consistently.
#
# With logq = true, log-densities are accumulated from the same fitted
# polynomial the samples are drawn from: log chebval - log chebdefinite per coordinate.
# The max-shift inside _chebdensity cancels between the value and the integral, and a
# non-uniform logpriors enters both, so the result is exactly the (normalised)
# log-density of the sampling law (floors and cancellation: see cheblogq).
function _sequential_core(estimator::TelescopingRatioEstimator, headfun, tzs, lower, upper, N::Integer, θfixed, degree::Integer, logpriors; logq::Bool = false)
    d = _numheads(estimator)
    @assert length(lower) == d && length(upper) == d "lower and upper must have one entry per parameter; expected length $d"
    @assert all(lower .< upper) "lower bounds must be strictly below upper bounds"
    isnothing(logpriors) || @assert length(logpriors) == d "logpriors must have one entry per parameter; expected length $d"
 
    # Match the network's element type (typically Float32) throughout: the Chebyshev plans,
    # the node inputs, and the uniforms. Mixing in Float64 anywhere silently promotes the
    # whole pipeline.
    T = eltype(tzs)
    L = degree + 1
    B = size(tzs, 2)
 
    # Fixed columns may lie outside the box (logposterior at arbitrary points): clamp
    # them for the polynomial work, remember which, and overwrite with -Inf at the end
    # (zero density under the truncated sampling law).
    F = 0
    θf = nothing
    inbox = nothing
    if !isnothing(θfixed)
        θf = T.(_stripnames(_extractθ(θfixed)))
        @assert size(θf, 1) == d "θfixed must have one row per parameter; expected $d rows, got $(size(θf, 1))"
        @assert size(θf, 2) % B == 0 "θfixed must hold the same number of fixed columns for each of the $B data sets"
        F = size(θf, 2) ÷ B
        lo, hi = T.(collect(lower)), T.(collect(upper))
        inbox = vec(all((θf .>= lo) .& (θf .<= hi), dims = 1))
        θf = clamp.(θf, lo, hi)
    end
    @assert N > 0 || F > 0 "nothing to do: no drawn and no fixed columns"
 
    θdrawn = Matrix{T}(undef, d, N * B)
    lq_drawn = logq && N > 0 ? zeros(T, N * B) : nothing
    lq_fixed = logq && F > 0 ? zeros(T, F * B) : nothing
 
    for i in 1:d
        plan = ChebPlan(lower[i], upper[i]; degree = degree, T = T)
        # NB: plan.nodes runs from upper[i] down to lower[i]; chebfit assumes function
        #     values in exactly that order, so evaluate the head at plan.nodes verbatim
        #     and never reorder
        nodesrow = reshape(plan.nodes, 1, L)
        logp = isnothing(logpriors) ? nothing : T.(logpriors[i].(plan.nodes))
        if i == 1
            # One envelope per data set, shared by all of its drawn and fixed columns
            X = vcat(repeat(tzs, inner = (1, L)), repeat(nodesrow, 1, B))
            Fv = _chebdensity(reshape(vec(headfun(1, X)), L, B), logp)
            C = chebfit(plan, Fv)
            CI = chebintegrate(plan, C)
            if N > 0
                # B envelopes, N draws each; vec of an N × B uniform matrix groups each
                # envelope's draws contiguously, as invert_cdf_batched expects
                θdrawn[1, :] = invert_cdf_batched(CI, plan.a, plan.b, vec(rand(T, N, B)))
            end
            if logq
                Zi = chebdefinite(CI, plan.a, plan.b)
                N > 0 && (lq_drawn .+= vec(cheblogq(reshape(θdrawn[1, :], N, B), C, Zi, plan.a, plan.b)))
                F > 0 && (lq_fixed .+= vec(cheblogq(reshape(θf[1, :], F, B), C, Zi, plan.a, plan.b)))
            end
        else
            # One envelope per column, each defined by that column's prefix. Columns are
            # ordered prefix-major, nodes-minor, so that reshaping the head output to
            # L × (N*B + F*B) puts envelope n in column n with values at plan.nodes,
            # which is the layout chebfit expects. Drawn and fixed envelopes share one
            # head call.
            Xd = N > 0 ? vcat(repeat(tzs, inner = (1, N * L)), repeat(view(θdrawn, 1:(i - 1), :), inner = (1, L)), repeat(nodesrow, 1, N * B)) : nothing
            Xf = F > 0 ? vcat(repeat(tzs, inner = (1, F * L)), repeat(view(θf, 1:(i - 1), :), inner = (1, L)), repeat(nodesrow, 1, F * B)) : nothing
            X = N > 0 ? (F > 0 ? hcat(Xd, Xf) : Xd) : Xf
            Fv = _chebdensity(reshape(vec(headfun(i, X)), L, N * B + F * B), logp)
            C = chebfit(plan, Fv)
            CI = chebintegrate(plan, C)
            if N > 0
                # one uniform per drawn envelope, one draw each
                θdrawn[i, :] = invert_cdf_batched(view(CI, :, 1:(N * B)), plan.a, plan.b, rand(T, N * B))
            end
            if logq
                Zi = chebdefinite(CI, plan.a, plan.b)
                N > 0 && (lq_drawn .+= cheblogq(θdrawn[i, :], view(C, :, 1:(N * B)), view(Zi, 1:(N * B)), plan.a, plan.b))
                F > 0 && (lq_fixed .+= cheblogq(θf[i, :], view(C, :, (N * B + 1):(N * B + F * B)), view(Zi, (N * B + 1):(N * B + F * B)), plan.a, plan.b))
            end
        end
    end
 
    isnothing(lq_fixed) || (lq_fixed[.!inbox] .= T(-Inf))
    return θdrawn, lq_drawn, lq_fixed
end
 
# Unnormalised density values at the Chebyshev nodes from per-head logits, optionally
# weighted by the log marginal prior density at the nodes. The per-envelope maximum is
# subtracted before exponentiating to prevent overflow for concentrated posteriors;
# the shift cancels when the CDF is normalised inside the sampler, and likewise between
# the value and the integral in the log-density accumulation of _sequential_core.
function _chebdensity(logits::AbstractVecOrMat, logp)
    s = isnothing(logp) ? logits : logits .+ logp
    exp.(s .- maximum(s, dims = 1))
end
 
@doc raw"""
	logposterior(estimator::TelescopingRatioEstimator, θpoints, Z; lower, upper, method = :raw, kwargs...)
Log-density of the approximate posterior at each parameter configuration in `θpoints`
(a `d × M` matrix, one configuration per column), by one of two methods that are NOT
interchangeable:

- `method = :raw` (default): the summed head logits plus the log prior — one network
  pass per head over the `M` points, no Chebyshev machinery. UNNORMALISED: defined up
  to an additive per-data-set constant. The right choice for MAP search, MCMC, or any
  within-data-set comparison that only needs the density up to a constant.
- `method = :chebyshev`: the normalised log-density of the law that
  [`sampleposterior`](@ref) actually draws from — per coordinate, the same envelope is
  fitted (deterministically, given the parameter prefix) and its value and normalising
  integral are read off the same polynomial. Roughly `degree + 1` times the network
  cost of `:raw`. Required whenever draws and query points must be ranked under the
  sampled law; the coverage checks pin this method internally.

The two outputs differ by the θ-dependent product of the prefix normalising constants,
not by a constant, so never mix methods within one computation; they agree (up to a
per-data-set constant) only in the limit of perfectly self-normalised heads. Both
report `-Inf` outside the box, the support of the (truncated) sampling law.

# Keyword arguments
`lower` and `upper` (required); `logpriors = nothing` as in [`sampleposterior`](@ref),
used by both methods; `degree = 128` (`:chebyshev` only). For self-consistency with
draws, `:chebyshev` must be called with the same `degree` and `logpriors` used when
sampling.

# Returns
A length-`M` vector of log-densities (one per column of `θpoints`) for a single data
set, or a vector of such vectors when `Z` contains multiple data sets.
"""
function logposterior(
    estimator::TelescopingRatioEstimator, θpoints::AbstractMatrix, Z;
    lower::AbstractVector,
    upper::AbstractVector,
    method::Symbol = :raw,
    degree::Integer = 128,
    logpriors::Union{Nothing, AbstractVector} = nothing,
    kwargs...
)
    @assert method in (:raw, :chebyshev) "method must be :raw or :chebyshev"
    if method === :raw
        return _logposterior_raw(logratio(estimator, Z; grid = θpoints, kwargs...), θpoints, lower, upper, logpriors)
    end
    summary_stats_Z = summarystatistics(estimator, Z; kwargs...)
    headfun = (i, X) -> _head(estimator, i, X)
    _logposterior_blocks(estimator, headfun, summary_stats_Z, θpoints, lower, upper, degree, logpriors)
end
 
# Raw path: one row of logratio per data set (summed head logits at the query points),
# plus the log prior, -Inf outside the box. Unnormalised, and relative to :chebyshev
# the prefix-dependent normalisers are absorbed into the shape — see the docstring.
function _logposterior_raw(LR::AbstractMatrix, θpoints, lower, upper, logpriors)
    θ = _stripnames(_extractθ(θpoints))
    @assert size(θ, 1) == length(lower) && length(lower) == length(upper) "θpoints must have one row per parameter"
    T = eltype(LR)
    lp = isnothing(logpriors) ? zeros(T, size(θ, 2)) : sum(i -> T.(logpriors[i].(θ[i, :])), 1:size(θ, 1))
    inbox = vec(all((θ .>= lower) .& (θ .<= upper), dims = 1))
    results = map(1:size(LR, 1)) do k
        v = vec(LR[k, :]) .+ lp
        v[.!inbox] .= T(-Inf)
        v
    end
    return length(results) == 1 ? results[1] : results
end
 
# Same points evaluated for every data set; one core call per data set (N = 0, all
# columns fixed). No fusing across data sets here: the joint coverage check, where the
# workload is real, has its own fused path through _sequential_core.
function _logposterior_blocks(estimator::TelescopingRatioEstimator, headfun, summary_stats_Z, θpoints, lower, upper, degree::Integer, logpriors)
    results = map(1:size(summary_stats_Z, 2)) do k
        _, _, lq = _sequential_core(estimator, headfun, summary_stats_Z[:, k:k], lower, upper, 0, θpoints, degree, logpriors; logq = true)
        lq
    end
    return length(results) == 1 ? results[1] : results
end
 
# ---- Inference: Stateless (Lux) ----
 
function logratio(estimator::TelescopingRatioEstimator, Z, ps, st; grid, kwargs...)
    grid = f32(grid)
    summary_stats_Z = summarystatistics(estimator, Z, ps, st; kwargs...)
    _gridlogratio(estimator, summary_stats_Z, grid, ps.heads, st.heads)
end
 
function _gridlogratio(estimator::TelescopingRatioEstimator, summary_stats_Z, grid::AbstractMatrix, ps_heads, st_heads)
    K = size(summary_stats_Z, 2)
    G = size(grid, 2)
    summary_stats_Z_rep = repeat(summary_stats_Z, inner = (1, G))
    grid_rep = repeat(grid, outer = (1, K))
    logits = first(estimator.heads(_headinputs(summary_stats_Z_rep, grid_rep), ps_heads, st_heads))
    log_ratios = sum(logits, dims = 1)
    return permutedims(reshape(log_ratios, G, K))  # K × G matrix
end
 
function sampleposterior(estimator::TelescopingRatioEstimator, Z, ps, st;
    lower::AbstractVector,
    upper::AbstractVector,
    N::Integer = 1000,
    degree::Integer = 128,
    logpriors::Union{Nothing, AbstractVector} = nothing,
    batchsize::Integer = 1,
    kwargs...
)
    summary_stats_Z = summarystatistics(estimator, Z, ps, st; kwargs...)
    headfun = (i, X) -> _head(estimator, i, X, ps, st)
    _sampleposterior_blocks(estimator, headfun, summary_stats_Z, lower, upper, N, degree, logpriors, batchsize)
end
 
function logposterior(estimator::TelescopingRatioEstimator, θpoints::AbstractMatrix, Z, ps, st;
    lower::AbstractVector,
    upper::AbstractVector,
    method::Symbol = :raw,
    degree::Integer = 128,
    logpriors::Union{Nothing, AbstractVector} = nothing,
    kwargs...
)
    @assert method in (:raw, :chebyshev) "method must be :raw or :chebyshev"
    if method === :raw
        return _logposterior_raw(logratio(estimator, Z, ps, st; grid = θpoints, kwargs...), θpoints, lower, upper, logpriors)
    end
    summary_stats_Z = summarystatistics(estimator, Z, ps, st; kwargs...)
    headfun = (i, X) -> _head(estimator, i, X, ps, st)
    _logposterior_blocks(estimator, headfun, summary_stats_Z, θpoints, lower, upper, degree, logpriors)
end
