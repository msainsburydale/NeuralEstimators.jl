@doc raw"""
	TelescopingRatioEstimator <: AbstractNeuralEstimator
	TelescopingRatioEstimator(num_parameters, summary_network = identity; num_summaries, sampler, kwargs...)
A neural estimator that factorises the likelihood-to-evidence ratio sequentally accross parameter diemsnions, with
one classifier MLP head per parameter coordinate. Currently, the implementation only supports the case in which
the prior factorizes $p(\boldsymbol{\theta}) = \prod_{i=1}^d p(\theta^i)$. In this case, we have that
```math
r(\boldsymbol{Z}, \boldsymbol{\theta}) = \prod_{i=1}^{d} r_i(\boldsymbol{Z}, \theta_i \mid \theta_1, \dots, \theta_{i-1}),
```
and head $i$ discriminates the true $\theta_i$ from a fresh prior draw, holding $\theta_1, \dots, \theta_{i-1}$ fixed and conditioning on it. 
This factorization exposes each one-dimensional conditional $p(\theta_i \mid \theta_1, \dots, \theta_{i-1}, \boldsymbol{Z})$, which allows for efficient posterior sampling by inversion sampling; see [`sampleposterior`](@ref).

The data are summarised by `summary_network`; the heads are a `MultiHeadMLP` with growing
inputs, head $i$ taking the summaries together with $\theta_1, \dots, \theta_i$.
If `summary_network` is omitted, it defaults to the identity function, for use with expert summary statistics provided as a matrix. See [Expert summary statistics](@ref).

# Keyword arguments
- `num_summaries::Integer`: the number of summaries output by `summary_network`. Must match the output dimension of `summary_network`.
- `sampler::Function`: a function returning `K` independent draws from the prior as a `d × K` matrix, the same function passed as `sampler` to [`train`](@ref). During training, this is used to generate independent samples for the $d$ classes with label 0.
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
estimator = TelescopingRatioEstimator(d, summary_network; num_summaries = num_summaries, sampler = sampler)

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
    sampler         # same sampler that generates theta; used during training to generate independent samples with label = 0
end
 
@functor TelescopingRatioEstimator (summary_network, heads)
 
# Constructor: one classifier head per parameter coordinate, with inputs size growing accross coordinates.
function TelescopingRatioEstimator(
    num_parameters::Integer, summary_network = identity;
    num_summaries::Integer,
    sampler::Function,
    kwargs...
)
    backend = _backendof(summary_network)
    heads = MultiHeadMLP(
        num_summaries, 1, num_parameters; 
        backend = backend, growing = true, 
        output_activation = identity, kwargs...)
    @info "TelescopingRatioEstimator: num_summaries = $num_summaries, num_heads = $num_parameters."
    TelescopingRatioEstimator(summary_network, heads, sampler)
end
 
# Constructor: consistent argument ordering
TelescopingRatioEstimator(summary_network, num_parameters::Integer; kwargs...) = TelescopingRatioEstimator(num_parameters, summary_network; kwargs...)

# Number of heads, equivalently the number of parameters.  
_numheads(estimator::TelescopingRatioEstimator) = length(estimator.heads.layers)
 
# Evaluate a single classifier head, without the other heads
# Sequential posterior sampling repeatedly needs to evaluate one head only at many inputs values.
# Flux stores the branches in a Tuple, Lux in a NamedTuple; ps/st mirror the NamedTuple order,
# but positional indexing agrees with the corresponding parameter/state containers.
_head(estimator::TelescopingRatioEstimator, i::Integer, X) = estimator.heads.layers[i](X)
_head(estimator::TelescopingRatioEstimator, i::Integer, X, ps, st) = first(estimator.heads.layers[i](X, ps.heads[i], st.heads[i]))
 
function _inputoutput(estimator::TelescopingRatioEstimator, Z, θ)
    d, K = size(θ)
    @assert d == _numheads(estimator) "θ has $d rows but the estimator has $(_numheads(estimator)) heads"
 
    # Fresh prior draws; head i then discriminates between θ[i] and θ̃ [i] conditionally on the prefix (previous coordinates), i = 1, ..., d
    θ̃ = _stripnames(_extractθ(estimator.sampler(K)))
    @assert size(θ̃) == (d, K) "sampler(K) must return a $d × $K parameter matrix; got $(size(θ̃))"
 
    # Binary class labels: rows 1:d for the dependent pairs, rows d+1:2d for the independent pairs.
    # Positives (label=1) and negatives (label=0) are stacked along rows (not columns) so that all components of the
    # input share numobs = K, as required by the data loader; this also keeps each (Z, θ, θ̃)
    # aligned under shuffling, which is importntt bc independent pairs reuse the prefix.
    output = vcat(ones(Float32, d, K), zeros(Float32, d, K))
 
    input = (Z, θ, θ̃)
    return input, output
end
 
_loss(estimator::TelescopingRatioEstimator, loss = nothing) = logitbinarycrossentropy

# Inputs to the d classifier heads; 
# for label class = 1:  (tz, θ1, ..., θi) 
# for label class = 0:  (tz, θ1, ..., θ_{i-1},θ̃_i)
# Head indices are obtained statically and remaing compile time constants under Reactant
# mapping over the reamining tuple is reliably differentiable with Zygote.

_headindices(estimator::TelescopingRatioEstimator) =  # gets the number of parameter dimensions from the network rathar than from θ
    _headindices(estimator.heads.layers)              

_headindices(::NTuple{d, Any}) where {d} =
    ntuple(identity, Val(d))

_headindices(::NamedTuple{names}) where {names} =
    ntuple(identity, Val(length(names)))

function _headinputs(estimator::TelescopingRatioEstimator, tz, θ)
    @assert size(θ,1) == _numheads(estimator)
    map(i -> vcat(tz, θ[1:i, :]), _headindices(estimator))
end

function _headinputs(estimator::TelescopingRatioEstimator, tz, θ, θ̃)
    @assert size(θ,1) == _numheads(estimator)
    map(i -> i == 1 ?
        vcat(tz, θ̃[1:1, :]) :
        vcat(tz, θ[1:(i - 1), :], θ̃[i:i, :]),
        _headindices(estimator))
end

 
# Forward pass: Stateful (Flux)
# Returns the d × K matrix of per-head logits; the total log-ratio (as in RatioEstimator.jl) is given by the sum over rows
function (estimator::TelescopingRatioEstimator)(Z, θ)
    tz = _summarystatistics(estimator, Z)
    inputs = _headinputs(estimator,tz, θ)
    estimator.heads(inputs)
end
 
# Training forward pass: 2d × K logits, matching the class labels constructed in _inputoutput
function (estimator::TelescopingRatioEstimator)(Z, θ, θ̃)
    tz = _summarystatistics(estimator, Z)
    pos = estimator.heads(_headinputs(estimator,tz, θ))
    neg = estimator.heads(_headinputs(estimator, tz, θ, θ̃))
    vcat(pos, neg)
end
 
# Forward pass: Stateless (Lux)
function (e::TelescopingRatioEstimator)(Z, θ, ps, st)
    tz, st_s = _summarystatistics(e, Z, ps.summary_network, st.summary_network)
    tz = tz |> copy # materialise to break Enzyme's trace (see the note in RatioEstimator.jl) 
    logits, st_h = e.heads(_headinputs(e,tz, θ), ps.heads, st.heads)
    return logits, (summary_network = st_s, heads = st_h)
end
 
function (e::TelescopingRatioEstimator)(Z, θ, θ̃, ps, st)
    tz, st_s = _summarystatistics(e, Z, ps.summary_network, st.summary_network)
    tz = tz |> copy # materialise to break Enzyme's trace (see the note in RatioEstimator.jl) 
    pos, st_h = e.heads(_headinputs(e,tz, θ), ps.heads, st.heads)
    neg, st_h = e.heads(_headinputs(e, tz, θ, θ̃), ps.heads, st_h)
    return vcat(pos, neg), (summary_network = st_s, heads = st_h)
end
 
# Bridge between the generic training loop and the TRE's specific forward-pass
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
    logits = estimator.heads(_headinputs(estimator,summary_stats_Z_rep, grid_rep))
    log_ratios = sum(logits, dims = 1)  # total log-ratio: sum of the per-head conditional log-ratios
    return permutedims(reshape(log_ratios, G, K))  # K × G matrix
end
 
@doc raw"""
	sampleposterior(estimator::TelescopingRatioEstimator, Z; lower, upper, N = 1000, chebyshev_batchsize = 1, kwargs...)
Draw posterior samples sequentially in the coordinate of theta: first generate 
$\theta_1 \mid Z$ from head 1, then $\theta_2 \mid \theta_1, Z$ from head 2, and so on
up to head number d. 

At each coordinate, the corresponding one-dimensional conditional density is approximated  by a Chebyshev polynomial on `[lower[i], upper[i]]`,
and sampled by inversion. The inversion step is performed to floating-point precision for the fitted polynomial, although
it is worth keeping in mind that the polynomial itself is an approximation to the TRE learnt conditionals, so a double-approximation.
 
For the first coordinate, all `N` samples are drawn from teh same approximate conditional density; 
every other coordinate must build one approximation per sample, because each sample has a different prefix.
We processed in batches for efficiency.

# Keyword arguments
- `lower::AbstractVector`, `upper::AbstractVector`: bounds of the prior support for each of the `d` parameter coordinates of $\boldsymbol{\theta}$. These must match the prior support used during training.
- `N::Integer = 1000`: number of posterior samples (per dataset).
- `degree::Integer = 128`: degree of the Chebyshev approximation of each conditional density.
- `logpriors = nothing`: marginal log-prior densities. Supply a vector of `d` functions, where `logpriors[i]` gives the marginal prior log-density of $\theta_i$. The default `nothing` assumes the prior is uniform over the box. The logpriors must match the ones used during training.
- `chebyshev_batchsize::Integer = 1`: number of data sets $(\boldsymbol{Z}, \boldsymbol{\theta})$ processed jointly by the posterior Chebyshev sampler. Increasing chebyshev_batchsize substantially improves computational efficiency, particularly on GPU, and also requires more memory. For N = 1000 posterior samples and degree = 128, cheb_batch = 50 is safe on most GPUs.
"""
function sampleposterior(
    estimator::TelescopingRatioEstimator, Z;
    lower::AbstractVector,
    upper::AbstractVector,
    N::Integer = 1000,
    degree::Integer = 128,
    logpriors::Union{Nothing, AbstractVector} = nothing,
    chebyshev_batchsize::Integer = 1,
    kwargs...
)
    summary_stats_Z = summarystatistics(estimator, Z; kwargs...)
    headfun = (i, X) -> _head(estimator, i, X)
    _sampleposterior_blocks(estimator, headfun, summary_stats_Z, lower, upper, N, degree, logpriors, chebyshev_batchsize)
end

# Process data sets in blocks of chebyshev_batchsize
function _sampleposterior_blocks(estimator::TelescopingRatioEstimator, headfun, summary_stats_Z, lower, upper, N::Integer, degree::Integer, logpriors, chebyshev_batchsize::Integer)
    K = size(summary_stats_Z, 2)
    samples = Vector{Matrix{eltype(summary_stats_Z)}}(undef, K)
    for block in Iterators.partition(1:K, chebyshev_batchsize)
        θdrawn, _, _ = _sequential_core(estimator, headfun, summary_stats_Z[:, block], lower, upper, N, nothing, degree, logpriors)
        for (j, k) in enumerate(block)
            samples[k] = θdrawn[:, ((j - 1) * N + 1):(j * N)]
        end
    end
    return stack(samples) # TODO do samples need to be stored as a vector of matrices in the first place? Would a reshape() on θdrawn be better?
end


# Construct the inputs to head i directly to avoid large intermediate arrays.
# P is the number of parameter columns per data set.
function _headinputmatrix(tzs, θ, nodes, i::Integer, P::Integer)
    S, B = size(tzs)
    L = length(nodes)
    X = Matrix{eltype(tzs)}(undef, S + i, P * B * L)
    Xr = reshape(X, S + i, L, P, B)
    Xr[1:S, :, :, :] .= reshape(tzs, S, 1, 1, B)
    if i > 1
        θr = reshape(θ, size(θ, 1), 1, P, B)          # θ is contiguous, so this is free
        Xr[(S + 1):(S + i - 1), :, :, :] .= view(θr, 1:(i - 1), :, :, :)
    end
    Xr[S + i, :, :, :] .= nodes
    return X
end

 
# Sequential Chebyhev core shared by `sampleposterior`, `logposterior`
# and coverage checks; `headfun(i, X)` evaluates head i on the input and returns its logits.
#
# Processes B data sets jointly in one fused pass, for improved efficiency. For each data set,
# there are N posterior draws (columns), and optionally, F columns with log-density evaluations
# logposterior is the N = 0 case; plain sampling is F = 0; the joint coverage check
# should use N = M draws plus F = 1 (the true parameter).
#
# Columns are stored data-set-major as
#     [ds1 draws ... dsB draws | ds1 fixed ... dsB fixed]
# so head inputs, Chebyshev approximations. uniforms and outputs all reshape consistently.
#
# With logq = true, log-densities are accumulated from the same fitted
# polynomial used for sampling, giving the normalised density of the posterior law. 
function _sequential_core(estimator::TelescopingRatioEstimator, headfun, tzs, lower, upper, N::Integer, θfixed, degree::Integer, logpriors; logq::Bool = false)
    d = _numheads(estimator)
    @assert length(lower) == d && length(upper) == d "lower and upper must have one entry per parameter; expected length $d"
    @assert all(lower .< upper) "lower bounds must be strictly below upper bounds"
    isnothing(logpriors) || @assert length(logpriors) == d "logpriors must have one entry per parameter; expected length $d"
 
    # Match the network summary element type throughout the Chebyshev calculations 
    # to avoid promotion from Float32 to Float64; for very peaked, pathological distributions, Float64 might be worth it.
    T = eltype(tzs)
    L = degree + 1
    B = size(tzs, 2)
 
    # the Chebyshev approximation is not defined outside of [lower[i], upper[i]]; values outside this interval are clamped
    # and recorded, so that the final log-density output is set to -Inf; this agrees with 0 posterior density outside the support
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
        logp = isnothing(logpriors) ? nothing : T.(logpriors[i].(plan.nodes))
        if i == 1 #i think we can get rid of this edge-case in Julia: to discuss with Matt
            # One conditional density approximation only (per data Z), shared by all N posterior samples, since there is no prefix yet.
            X = _headinputmatrix(tzs, nothing, plan.nodes, 1, 1) 
            Fv = _chebdensity(reshape(vec(headfun(1, X)), L, B), logp)
            C = chebfit(plan, Fv)
            CI = chebintegrate(plan, C)
            if N > 0
                # B conditional densities, N draws from each; vec of an N × B uniform matrix stores each density's uniforms
                # as required for `invert_cdf_batched`.
                θdrawn[1, :] = invert_cdf_batched(CI, plan.a, plan.b, vec(rand(T, N, B)))
            end
            if logq
                Zi = chebdefinite(CI, plan.a, plan.b)
                N > 0 && (lq_drawn .+= vec(cheblogq(reshape(θdrawn[1, :], N, B), C, Zi, plan.a, plan.b)))
                F > 0 && (lq_fixed .+= vec(cheblogq(reshape(θf[1, :], F, B), C, Zi, plan.a, plan.b)))
            end
        else
            # For i > 1, each sample has its own prefix and hence it requires its own conditional-density approximation. 
            # we evaluate both drawn and fixed points together for efficienc.y
            Xd = N > 0 ? _headinputmatrix(tzs, θdrawn, plan.nodes, i, N) : nothing
            Xf = F > 0 ? _headinputmatrix(tzs, θf, plan.nodes, i, F) : nothing

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
# including the log marginal prior density at the nodes. The per-envelope maximum is
# subtracted before exponentiating to prevent overflow;
# the shift cancels when the CDF is normalised inside the sampler, and likewise between
# the value and the integral in the log-density accumulation of _sequential_core.
function _chebdensity(logits::AbstractVecOrMat, logp)
    s = isnothing(logp) ? logits : logits .+ logp
    exp.(s .- maximum(s, dims = 1))
end
 
@doc raw"""
	logposterior(estimator::TelescopingRatioEstimator, θpoints, Z; lower, upper, method = :raw, kwargs...)
Evaluate the learnt posterior log-density at the parameter values in `θpoints`,
by one of two methods that are NOT interchangeable:

- `method = :raw` (default): sums the classifier heads' log ratios and adds the log prior to
  recover the log posterior. This result is UNNORMALISED, in the sense that the TRE
  learnt posteriors do not integrate exactly to 1. Indeed, there is always some estimation error in NRE/TRE, and
  unlike in normalizing flows, posterior integrals are not equal to 1 by construction. 
  This method is suitable for MAP (or say for MCMC, although Chebyshev posterior sampling is preferable).
- `method = :chebyshev`: evaluates the (log) normalized posterior density by the same sequentail 
  Chebyshev approximations used by [`sampleposterior`](@ref). Specifically, each of the one-dimensional
  conditional densities is normalized, which ensures the posterior is also normalized. This is far more
  computationally expensive, but importantly, it gives the density of the approximate distribution from
  which `sampleposterior` actually generates.

The two outputs differ by the prefix-dependednt normalising constants.
Consequently, the two methods must not differ only by an additive constant.

Both methods report `-Inf` for parameter values outside the box, i.e., outside `[lower, upper]`.

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
 
# Raw posterior log-density: sum head log ratios plus factorised log prior, -Inf outside the box. 
function _logposterior_raw(LR::AbstractMatrix, θpoints, lower, upper, logpriors)
    θ = _stripnames(_extractθ(θpoints))
    @assert size(θ, 1) == length(lower) && length(lower) == length(upper) "θpoints must have one row per parameter"
    T = eltype(LR)
    logprior = isnothing(logpriors) ? zeros(T, size(θ, 2)) : sum(i -> T.(logpriors[i].(θ[i, :])), 1:size(θ, 1))
    inbox = vec(all((θ .>= lower) .& (θ .<= upper), dims = 1))
    results = map(1:size(LR, 1)) do k
        v = vec(LR[k, :]) .+ logprior
        v[.!inbox] .= T(-Inf)
        v
    end
    return length(results) == 1 ? results[1] : results
end
 
# For each data Z, evaluate the (normalized, Chebyshev approximate) posterior log density at every parameter vector in
# `θpoints`. No posterior samples are generated here; 
# `_sequential_core` is called with `θpoints` as columns and $N=0
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
    logits = first(estimator.heads(_headinputs(estimator, summary_stats_Z_rep, grid_rep), ps_heads, st_heads))
    log_ratios = sum(logits, dims = 1)
    return permutedims(reshape(log_ratios, G, K))  # K × G matrix
end
 
function sampleposterior(estimator::TelescopingRatioEstimator, Z, ps, st;
    lower::AbstractVector,
    upper::AbstractVector,
    N::Integer = 1000,
    degree::Integer = 128,
    logpriors::Union{Nothing, AbstractVector} = nothing,
    chebyshev_batchsize::Integer = 1,
    kwargs...
)
    summary_stats_Z = summarystatistics(estimator, Z, ps, st; kwargs...)
    headfun = (i, X) -> _head(estimator, i, X, ps, st)
    _sampleposterior_blocks(estimator, headfun, summary_stats_Z, lower, upper, N, degree, logpriors, chebyshev_batchsize)
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
