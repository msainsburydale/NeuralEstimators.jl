@doc raw"""
	TelescopingRatioEstimator <: AbstractNeuralEstimator
	TelescopingRatioEstimator(num_parameters, summary_network = identity; num_summaries, sampler, kwargs...)
A neural estimator that factorises the likelihood-to-evidence ratio sequentally accross parameter dimensions, with
one classifier MLP head per parameter coordinate. When the prior factorises, $p(\boldsymbol{\theta}) = \prod_{i=1}^d p(\theta_i)$, we have that
```math
r(\boldsymbol{Z}, \boldsymbol{\theta}) = \prod_{i=1}^{d} r_i(\boldsymbol{Z}, \theta_i \mid \theta_1, \dots, \theta_{i-1}),
```
and head $i$ discriminates the true $\theta_i$ from a fresh prior draw, holding $\theta_1, \dots, \theta_{i-1}$ fixed and conditioning on it. 
This factorization exposes each one-dimensional conditional $p(\theta_i \mid \theta_1, \dots, \theta_{i-1}, \boldsymbol{Z})$, which allows for efficient posterior sampling by inversion sampling; see [`sampleposterior`](@ref).

When the prior does not factorise (e.g., when a constraint couples the parameters), `sampler` may instead draw from
a reference distribution whose support contains that of the prior, such as the prior without the constraint. Head $i$
then learns the ratio of $p(\theta_i \mid \theta_1, \dots, \theta_{i-1}, \boldsymbol{Z})$ to the $i$th marginal density
of the reference distribution, so posterior sampling remains exact when `logpriors` are these marginal densities (see
[`sampleposterior`](@ref)); however, [`logratio`](@ref) then no longer gives the likelihood-to-evidence ratio.

The data are summarised by `summary_network`; the heads are a `MultiHeadMLP` with growing
inputs, head $i$ taking the summaries together with $\theta_1, \dots, \theta_i$.
If `summary_network` is omitted, it defaults to the identity function, for use with expert summary statistics provided as a matrix. See [Expert summary statistics](@ref).

!!! note
    To use `TelescopingRatioEstimator` with `Enzyme.jl`, set `adtype = AutoEnzyme(mode = set_runtime_activity(Enzyme.Reverse))` in [`train`](@ref).

# Keyword arguments
- `num_summaries::Integer`: the number of summaries output by `summary_network`. Must match the output dimension of `summary_network`.
- `sampler::Function`: a function returning `K` independent draws from the prior, or from a reference distribution (see above), as a `d × K` matrix. During training, this is used to generate independent samples for the $d$ classes with label 0.
- `standardise_data = true`: whether, or how, to standardise the data before they enter the summary network (see [`Standardiser`](@ref)).
- `standardise_parameters = true`: whether, or how, to standardise the parameters before they enter the heads. Neither form of standardisation changes the ratios being estimated, and the bounds, priors, and posterior draws remain on the original scale.
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
    sampler         # draws from the prior or a reference distribution; used during training to generate independent samples with label = 0
    data_standardiser      # see Standardiser
    parameter_standardiser # standardises θ before the heads
end

# The sampler and standardisers are kept out of the functor (see RatioEstimator)
@functor TelescopingRatioEstimator (summary_network, heads)

# Constructor: networks and sampler
function TelescopingRatioEstimator(summary_network, heads, sampler; standardise_data = true, standardise_parameters = true)
    TelescopingRatioEstimator(summary_network, heads, sampler, _standardiser(standardise_data), _standardiser(standardise_parameters))
end

# Constructor: one classifier head per parameter coordinate, with inputs size growing accross coordinates.
function TelescopingRatioEstimator(
    num_parameters::Integer, summary_network = identity;
    num_summaries::Integer,
    sampler::Function,
    standardise_data = true,
    standardise_parameters = true,
    kwargs...
)
    summary_network = _resolvesummarynetwork(summary_network; kwargs...)
    backend = _backendof(summary_network)
    heads = MultiHeadMLP(
        num_summaries, 1, num_parameters;
        backend = backend, growing = true,
        output_activation = identity, _dropbackend(kwargs)...)
    @info "TelescopingRatioEstimator: num_summaries = $num_summaries, num_heads = $num_parameters."
    TelescopingRatioEstimator(summary_network, heads, sampler; standardise_data, standardise_parameters)
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

    # Standardised once, on the host (see _Prestandardised)
    sθ = _unmark(estimator.parameter_standardiser)
    input = (_standardisedata(_unmark(estimator.data_standardiser), Z), _standardise(sθ, θ), _standardise(sθ, θ̃))
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
    @assert size(θ, 1) == _numheads(estimator)
    map(i -> vcat(tz, θ[1:i, :]), _headindices(estimator))
end

function _headinputs(estimator::TelescopingRatioEstimator, tz, θ, θ̃)
    @assert size(θ, 1) == _numheads(estimator)
    map(i -> i == 1 ?
             vcat(tz, θ̃[1:1, :]) :
             vcat(tz, θ[1:(i - 1), :], θ̃[i:i, :]),
        _headindices(estimator))
end

# Forward passes: the original when the standardisers apply no map (e.g., during training, when the data and parameters
# are standardised on the host); otherwise, the data and parameters are standardised first (see _NoMap)
const _TelescopingRatioEstimatorNoMap = TelescopingRatioEstimator{<:Any, <:Any, <:Any, <:_NoMap, <:_NoMap}

# Forward pass: Stateful (Flux)
# Returns the d × K matrix of per-head logits; the total log-ratio (as in RatioEstimator.jl) is given by the sum over rows
function (estimator::_TelescopingRatioEstimatorNoMap)(Z, θ)
    tz = _summarystatistics(estimator, Z)
    inputs = _headinputs(estimator, tz, θ)
    estimator.heads(inputs)
end

# Training forward pass: 2d × K logits, matching the class labels constructed in _inputoutput
function (estimator::_TelescopingRatioEstimatorNoMap)(Z, θ, θ̃)
    tz = _summarystatistics(estimator, Z)
    pos = estimator.heads(_headinputs(estimator, tz, θ))
    neg = estimator.heads(_headinputs(estimator, tz, θ, θ̃))
    vcat(pos, neg)
end

# Forward pass: Stateless (Lux)
function (e::_TelescopingRatioEstimatorNoMap)(Z, θ, ps, st)
    tz, st_s = _summarystatistics(e, Z, ps.summary_network, st.summary_network)
    tz = tz |> copy # materialise to break Enzyme's trace (see the note in RatioEstimator.jl) 
    logits, st_h = e.heads(_headinputs(e, tz, θ), ps.heads, st.heads)
    return logits, (summary_network = st_s, heads = st_h)
end

function (e::_TelescopingRatioEstimatorNoMap)(Z, θ, θ̃, ps, st)
    tz, st_s = _summarystatistics(e, Z, ps.summary_network, st.summary_network)
    tz = tz |> copy # materialise to break Enzyme's trace (see the note in RatioEstimator.jl) 
    pos, st_h = e.heads(_headinputs(e, tz, θ), ps.heads, st.heads)
    neg, st_h = e.heads(_headinputs(e, tz, θ, θ̃), ps.heads, st_h)
    return vcat(pos, neg), (summary_network = st_s, heads = st_h)
end

# Forward passes with standardisation
(e::TelescopingRatioEstimator)(Z, θ) = _withoutstandardisers(e)(_standardisedata(e.data_standardiser, Z), _standardise(e.parameter_standardiser, θ))
(e::TelescopingRatioEstimator)(Z, θ, θ̃) = _withoutstandardisers(e)(_standardisedata(e.data_standardiser, Z), _standardise(e.parameter_standardiser, θ), _standardise(e.parameter_standardiser, θ̃))
(e::TelescopingRatioEstimator)(Z, θ, ps, st) = _withoutstandardisers(e)(_standardisedata(e.data_standardiser, Z), _standardise(e.parameter_standardiser, θ), ps, st)
(e::TelescopingRatioEstimator)(Z, θ, θ̃, ps, st) = _withoutstandardisers(e)(_standardisedata(e.data_standardiser, Z), _standardise(e.parameter_standardiser, θ), _standardise(e.parameter_standardiser, θ̃), ps, st)

# Bridge between the generic training loop and the TRE's specific forward-pass
(estimator::TelescopingRatioEstimator)(input::Tuple) = estimator(input...)
(estimator::TelescopingRatioEstimator)(input::Tuple, ps, st) = estimator(input..., ps, st)
(estimator::_TelescopingRatioEstimatorNoMap)(input::Tuple, ps, st) = estimator(input..., ps, st) # resolves the ambiguity with the training forward pass

# ---- Inference: Stateful (Flux) ----

function logratio(estimator::TelescopingRatioEstimator, Z; grid, kwargs...)
    grid = f32(grid)
    summary_stats_Z = summarystatistics(estimator, Z; kwargs...)
    _gridlogratio(estimator, summary_stats_Z, grid)
end

function _gridlogratio(estimator::TelescopingRatioEstimator, summary_stats_Z, grid::AbstractMatrix)
    grid = _standardise(estimator.parameter_standardiser, grid) # the heads take standardised parameters
    K = size(summary_stats_Z, 2)    # number of data sets
    G = size(grid, 2)               # number of grid points
    # Repeat so that the summaries and the grid both have GxK columns
    summary_stats_Z_rep = repeat(summary_stats_Z, inner = (1, G))
    grid_rep = repeat(grid, outer = (1, K))
    logits = estimator.heads(_headinputs(estimator, summary_stats_Z_rep, grid_rep))
    log_ratios = sum(logits, dims = 1)  # total log-ratio: sum of the per-head conditional log-ratios
    return permutedims(reshape(log_ratios, G, K))  # K × G matrix
end

@doc raw"""
	sampleposterior(estimator::TelescopingRatioEstimator, Z; lower, upper, N = 1000, chebyshev_batchsize = 1, kwargs...)
Draw posterior samples sequentially in the coordinate of theta: first generate 
$\theta_1 \mid Z$ from head 1, then $\theta_2 \mid \theta_1, Z$ from head 2, and so on
up to head number d. 

At each coordinate, the corresponding one-dimensional conditional density is evaluated at the `degree + 1` Chebyshev nodes of `[lower[i], upper[i]]`.
If its mass lies in less than half of this interval (as when the posterior is concentrated relative to the prior support), it is evaluated
again at the nodes of the sub-interval that holds the mass, up to twice, so that the nodes resolve it. The density is then approximated by a
Chebyshev polynomial on the final interval, and sampled by inversion. The inversion step is performed to floating-point precision for the fitted polynomial, although
it is worth keeping in mind that the polynomial itself is an approximation to the TRE learnt conditionals, so a double-approximation.
 
For the first coordinate, all `N` samples are drawn from teh same approximate conditional density; 
every other coordinate must build one approximation per sample, because each sample has a different prefix.
We processed in batches for efficiency.

# Keyword arguments
- `lower::AbstractVector`, `upper::AbstractVector`: bounds for each of the `d` parameter coordinates of $\boldsymbol{\theta}$. The draws are confined to this box, which should lie within the support of the distribution sampled by the estimator's `sampler` and hold essentially all of the posterior mass: for a prior with unbounded support, choose bounds that hold essentially all of its mass.
- `N::Integer = 1000`: number of posterior samples (per dataset).
- `degree::Integer = 128`: degree of the Chebyshev approximation of each conditional density.
- `logpriors = nothing`: the marginal log densities, up to constants, of the distribution sampled by the estimator's `sampler` (the prior, or a reference distribution; see [`TelescopingRatioEstimator`](@ref)). Supply a vector of `d` functions, where `logpriors[i]` gives the log density of $\theta_i$; these must return finite values, or `-Inf` where the density is zero (e.g., at a bound), never `NaN` or `Inf` (for a density that is infinite at a bound, move the bound slightly inwards). The default `nothing` assumes that this distribution is uniform over the box.
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

# Construct the inputs to head i directly to avoid large intermediate arrays: for each column of the summaries `tzs`
# and of `θ`, whose first i - 1 rows hold the prefix θ₁, …, θᵢ₋₁ (θ is ignored for i = 1), one input per node of θᵢ.
# The nodes are a vector shared by all columns, or a matrix with one column per column. The parameters θ₁, …, θᵢ are
# standardised in place by `s`, the parameter standardiser of the estimator.
function _headinputmatrix(tzs, θ, nodes, i::Integer, s = nothing)
    S, B = size(tzs)
    L = size(nodes, 1)
    X = Matrix{eltype(tzs)}(undef, S + i, B * L)
    Xr = reshape(X, S + i, L, B)
    Xr[1:S, :, :] .= reshape(tzs, S, 1, B)
    if i > 1
        θr = reshape(θ, size(θ, 1), 1, B)          # θ is contiguous, so this is free
        Xr[(S + 1):(S + i - 1), :, :] .= view(θr, 1:(i - 1), :, :)
    end
    Xr[S + i, :, :] .= reshape(nodes, L, :)
    _standardiserows!(X, s, (S + 1):(S + i))
    return X
end

# Standardises, in place, the rows `rows` of X, which hold the first length(rows) parameters
_standardiserows!(X, s, rows) = X
function _standardiserows!(X, s::Standardiser{<:Union{Number, AbstractArray}}, rows)
    μ, σ = s.mean isa Number ? (s.mean, s.std) : (s.mean[1:length(rows), :], s.std[1:length(rows), :])
    @views X[rows, :] .= (X[rows, :] .- μ) ./ σ
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
# Each conditional density (envelope) is first evaluated at the nodes of [lower[i], upper[i]]. If its mass lies in less
# than half of that interval, it is evaluated again at the nodes of the sub-interval that holds the mass (see _zoom!),
# at most twice, so that the nodes resolve a posterior much narrower than the box. The densities are fitted, integrated
# and inverted on [-1, 1], which each envelope maps to its own interval [lo, hi].
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
    sθ = estimator.parameter_standardiser # the head inputs are standardised, while the nodes, priors, and draws remain on the original scale
    plan = ChebPlan(-1, 1; degree = degree, T = T)

    # the Chebyshev approximation is not defined outside of the interval of an envelope; fixed values outside the box are clamped
    # and recorded, as are those outside the interval of their envelope, so that the final log-density output is set to -Inf;
    # this agrees with 0 posterior density outside the support
    F = 0
    θf = nothing
    outside = nothing
    if !isnothing(θfixed)
        θf = T.(_stripnames(_extractθ(θfixed)))
        @assert size(θf, 1) == d "θfixed must have one row per parameter; expected $d rows, got $(size(θf, 1))"
        @assert size(θf, 2) % B == 0 "θfixed must hold the same number of fixed columns for each of the $B data sets"
        F = size(θf, 2) ÷ B
        lb, ub = T.(collect(lower)), T.(collect(upper))
        outside = .!vec(all((θf .>= lb) .& (θf .<= ub), dims = 1))
        θf = clamp.(θf, lb, ub)
    end
    @assert N > 0 || F > 0 "nothing to do: no drawn and no fixed columns"

    θdrawn = Matrix{T}(undef, d, N * B)
    lq_drawn = logq && N > 0 ? zeros(T, N * B) : nothing
    lq_fixed = logq && F > 0 ? zeros(T, F * B) : nothing

    for i = 1:d
        # For i = 1, one envelope per data set, shared by its N draws and F fixed columns, since there is no prefix yet
        # (i think we can get rid of this edge-case in Julia: to discuss with Matt). For i > 1, each column has its own
        # prefix, and hence its own envelope; drawn and fixed columns are evaluated together for efficiency. `drawn` and
        # `fixed` are the envelopes of the drawn and fixed columns, each shared by Md and Mf columns, and ds is the data
        # set of each envelope. (One assignment each, so that the closure below does not box them.)
        E = i == 1 ? B : (N + F) * B
        ds = i == 1 ? (1:B) : vcat(repeat(1:B, inner = N), repeat(1:B, inner = F))
        θprefix = i == 1 ? nothing : F > 0 ? hcat(θdrawn, θf) : θdrawn
        drawn, fixed = i == 1 ? (1:B, 1:B) : (1:(N * B), (N * B) .+ (1:(F * B)))
        Md, Mf = i == 1 ? (N, F) : (1, 1)

        # Densities of the envelopes ks at the nodes, a vector shared by all of them or a matrix with one column each
        function density(ks, nodes)
            X = _headinputmatrix(tzs[:, ds[ks]], isnothing(θprefix) ? nothing : θprefix[:, ks], nodes, i, sθ)
            logp = isnothing(logpriors) ? nothing : T.(logpriors[i].(nodes))
            @assert isnothing(logp) || all(x -> isfinite(x) || x == -Inf, logp) "logpriors[$i] returned NaN or Inf; it must return finite values, or -Inf where the density is zero (for a density that is infinite at a bound, move the bound slightly inwards)"
            return _chebdensity(reshape(vec(headfun(i, X)), L, length(ks)), logp)
        end

        # Densities at the nodes of the box, then at the nodes of the sub-interval that holds the mass, for the envelopes whose
        # mass lies in less than half of their interval
        lo, hi = fill(T(lower[i]), E), fill(T(upper[i]), E)
        Fv = density(1:E, (lo[1] + hi[1]) / 2 .+ (hi[1] - lo[1]) / 2 .* plan.nodes)
        ks = 1:E
        for _ = 1:2
            ks = _zoom!(lo, hi, Fv, plan.nodes, ks)
            isempty(ks) && break
            Fv[:, ks] = density(ks, (lo[ks] .+ hi[ks])' ./ 2 .+ (hi[ks] .- lo[ks])' ./ 2 .* plan.nodes)
        end
        C = chebfit(plan, Fv)
        CI = chebintegrate(plan, C)
        off, scl = (lo .+ hi) ./ 2, (hi .- lo) ./ 2  # envelope k maps z in [-1, 1] to x = off[k] + scl[k] * z

        if N > 0
            # N draws from each envelope for i = 1, one draw from each drawn envelope for i > 1; the uniforms of an
            # envelope are stored consecutively, as required by `invert_cdf_batched`
            zdrawn = invert_cdf_batched(view(CI, :, drawn), plan.a, plan.b, rand(T, N * B))
            k = repeat(drawn, inner = Md)
            θdrawn[i, :] = clamp.(off[k] .+ scl[k] .* zdrawn, lo[k], hi[k])
        end
        if logq
            # log density at z of the normalised polynomial of each envelope in cols (M points each), less log(scl), for the density of x
            Zi = chebdefinite(CI, plan.a, plan.b)
            lq(z, cols, M) = vec(cheblogq(reshape(z, M, :), view(C, :, cols), view(Zi, cols), plan.a, plan.b) .- log.(view(scl, cols))')
            N > 0 && (lq_drawn .+= lq(zdrawn, drawn, Md))
            if F > 0
                k = repeat(fixed, inner = Mf)
                x = θf[i, :]
                outside .|= (x .< lo[k]) .| (x .> hi[k])
                lq_fixed .+= lq(clamp.((x .- off[k]) ./ scl[k], plan.a, plan.b), fixed, Mf)
            end
        end
    end

    isnothing(lq_fixed) || (lq_fixed[outside] .= T(-Inf))
    return θdrawn, lq_drawn, lq_fixed
end

# Narrows, in place, the interval [lo[k], hi[k]] of each envelope k in ks to the sub-interval that holds its mass, if that
# is less than half of the interval, and returns the envelopes narrowed. Fv[:, k] holds the density of envelope k at the
# nodes z of [-1, 1] mapped to [lo[k], hi[k]], scaled to a maximum of one (see _chebdensity). The sub-interval spans the
# nodes at which the density exceeds ε, padded by one node on each side; every node outside it has density at most ε.
function _zoom!(lo, hi, Fv, z, ks; ε = 1e-8)
    narrowed = Int[]
    for k in ks
        f = view(Fv, :, k)
        j₁, j₂ = findfirst(>(ε), f), findlast(>(ε), f)  # the nodes decrease, from 1 to -1
        isnothing(j₁) && continue
        zhi, zlo = z[max(j₁ - 1, 1)], z[min(j₂ + 1, length(z))]
        off, scl = (lo[k] + hi[k]) / 2, (hi[k] - lo[k]) / 2
        # An end reached by the padding is kept exactly (recomputing it can round it out of, or into, the box)
        a = j₂ + 1 >= length(z) ? lo[k] : off + scl * zlo
        b = j₁ <= 2 ? hi[k] : off + scl * zhi
        if zhi - zlo < 1 && a < b  # less than half of [-1, 1], and wider than zero in floating point
            lo[k], hi[k] = a, b
            push!(narrowed, k)
        end
    end
    return narrowed
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

Both methods report `-Inf` for parameter values outside the box, i.e., outside `[lower, upper]`. The method `:chebyshev`
also reports `-Inf` outside the sub-interval on which a concentrated conditional density is approximated (see
[`sampleposterior`](@ref)), since the sampler assigns no mass there.

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
    grid = _standardise(estimator.parameter_standardiser, grid) # the heads take standardised parameters
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
