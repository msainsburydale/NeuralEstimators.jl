module NeuralEstimatorsPlottingMakieExt

using NeuralEstimators
using Makie
import Makie: plot
export plot

# ===========================================================================
#  plotrisk()
# ===========================================================================
import NeuralEstimators: _plotrisk
function _plotrisk(savepath::String)
    history = loadrisk(savepath)
    epochs = 0:(size(history, 1) - 1)

    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel = "Epoch",
        ylabel = "Empirical risk (average loss)"
    )

    lines!(ax, epochs, history[:, 1];
        label = "Training",
        linewidth = 2
    )

    lines!(ax, epochs, history[:, 2];
        label = "Validation",
        linewidth = 2
    )

    axislegend(ax)

    return fig
end

# ===========================================================================
#  plot(assessment)
# ===========================================================================

using DataFrames
using Random: Xoshiro
using Statistics: mean, std, var, quantile
using StatsFuns: binominvcdf, poisinvcdf, poispdf

"""
    plot(assessment::Assessment; plots = nothing, ...)

Visualise the performance of a neural estimator, given the [`Assessment`](@ref)
object returned by [`assess`](@ref). Returns a Makie `Figure` with one panel
for each parameter.

!!! note "Extension"
    This function is defined in the `NeuralEstimatorsPlottingMakieExt` extension and
    requires `CairoMakie` (or another Makie backend) to be loaded.

The plots that are available depend on the type of estimator that was assessed.
By default all of them are drawn; use the keyword argument `plots` to select a subset.

**Point and interval estimates** ([`PointEstimator`](@ref), [`IntervalEstimator`](@ref)):

- `:recovery`: estimates against the true values, with intervals (when the
  assessment contains them) drawn as vertical line segments. Accurate estimates
  lie close to the dashed identity line.

**Quantile estimates** ([`QuantileEstimator`](@ref)):

- `:calibration`: the proportion of estimated quantiles that are greater than the
  true value, against the nominal probability level τ. Specifically, for
  k = 1,…,K, sample pairs (θᵏ, Zᵏ) with θᵏ ∼ p(θ) and Zᵏ ∼ p(Z ∣ θᵏ), so that θᵏ
  is a draw from the posterior p(θ ∣ Zᵏ); then, for each τ, plot the proportion of
  the estimated quantiles Q(Zᵏ, τ) that are greater than θᵏ. A well-calibrated
  estimator follows the dashed identity line.

**Posterior samples** ([`PosteriorEstimator`](@ref), [`RatioEstimator`](@ref), [`TelescopingRatioEstimator`](@ref)):

- `:recovery`: point estimates (the `pointsummary` given to [`assess`](@ref), by
  default the posterior mean) against the true values, with central 95% credible
  intervals drawn as vertical line segments.
- `:ecdf`: simulation-based calibration. For each parameter, the empirical
  distribution function of the fractional rank of the true value among the
  posterior draws, together with a simultaneous `prob`-level confidence band
  ([Säilynoja et al., 2022](https://doi.org/10.1007/s11222-022-10090-6)). A
  well-calibrated posterior gives a curve that stays within the band.
- `:zscore`: the posterior z-score, (posterior mean − true value) / posterior
  standard deviation, against the posterior contraction, 1 − posterior variance /
  prior variance. Ideally the z-scores are centred on zero and the contractions
  are close to one.

# Keyword arguments
- `plots = nothing`: the plots to draw, given as a `Symbol` or a collection of
  `Symbol`s from those listed above, in the order in which they should appear
  (e.g., `plots = (:recovery, :ecdf)`). By default, all available plots are drawn.
- `prob = 0.99`: simultaneous coverage of the confidence band in the `:ecdf` plot.
- `difference = true`: if `true`, the `:ecdf` plot shows the difference between the
  empirical distribution function and that of the uniform distribution, which
  makes departures from calibration easier to see; if `false`, it shows the
  empirical distribution function itself.
- `grid = false`: when the assessment contains several estimators (see `merge()`),
  they are by default drawn in the same panels in different colours. If
  `grid = true`, each estimator is instead given its own row of panels, which is
  easier to read with more than three estimators.
- `ncols = nothing`: the number of panels in each row, after which the parameters
  wrap onto a new row. By default, at most four, balanced across rows.
- `transpose = nothing`: by default, the figure has one row of panels for each plot
  and one column for each parameter. If `transpose = true`, it instead has one row
  for each parameter and one column for each plot (e.g., 2 × 3 for two parameters
  and three plots), and `ncols` has no effect. With a single parameter, the figure
  is transposed unless `transpose = false`.
- `figure = (;)`, `axis = (;)`: attributes passed to the `Figure` and to every
  `Axis`, respectively. By default the panels are of a fixed size and the figure
  is sized to fit them; give `figure = (; size = (w, h))` to fix the size of the
  figure instead. Colours and fonts are taken from the current Makie theme.

# Examples
```julia
using NeuralEstimators, CairoMakie

# Given an estimator and test parameters and data (see `assess`)
assessment = assess(estimator, θ_test, Z_test)

plot(assessment)                               # all available plots
plot(assessment; plots = :recovery)            # a single plot
plot(assessment; plots = (:recovery, :ecdf))   # a subset (posterior samples)
plot(assessment; transpose = true)             # one row per parameter, one column per plot
```
"""
function plot(assessment::Assessment;
    plots = nothing,
    prob::Real = 0.99,
    difference::Bool = true,
    grid::Bool = false,
    ncols::Union{Integer, Nothing} = nothing,
    transpose::Union{Bool, Nothing} = nothing,
    figure = (;),
    axis = (;)
)
    0 < prob < 1 || throw(ArgumentError("`prob` must lie strictly between 0 and 1"))
    kinds = _resolveplots(assessment, plots)
    data = _plotdata(assessment, kinds, prob, difference)
    parameters = unique(assessment.estimates.parameter)
    estimators = "estimator" ∈ names(assessment.estimates) ? unique(assessment.estimates.estimator) : [""]
    colors, ink = _themecolors(estimators)

    # The figure is made of blocks of panels, one block for each plot and, if `grid = true`, for each estimator.
    # A block has one panel per parameter, wrapped over `nc` columns, with its axis labels in a column to its
    # left and in a row beneath it. The blocks are stacked, so that each row of panels belongs to one plot.
    # If the figure is transposed (the default with a single parameter), a block is instead a single column of
    # panels and the plots are placed side by side, with one row of blocks for each group of estimators.
    groups = grid && length(estimators) > 1 ? [[estimator] for estimator in estimators] : [estimators]
    d = length(parameters)
    transposed = something(transpose, d == 1)
    nc = transposed ? 1 : isnothing(ncols) ? cld(d, cld(d, 4)) : clamp(ncols, 1, d)
    nr = cld(d, nc)
    blockcols = transposed ? length(kinds) : 1

    # Panels have a fixed size, and the figure is resized to fit them, unless the size of the figure is given
    fixed = !haskey(figure, :size)
    side = (300, 260, 230, 200)[min(nc * blockcols, 4)]
    fig = Figure(; figure...)
    layout = GridLayout(fig[1, 1])

    for (ki, kind) in enumerate(kinds)
        xlabel, ylabel = _axislabels(kind, assessment, difference)
        limits = _limits(kind, data[kind], difference)
        panels = groupby(data[kind], [:estimator, :parameter])
        shared = kind !== :recovery   # the panels have a common scale, so only the outer ones need tick labels

        for (gi, group) in enumerate(groups)
            # position of the block in the grid of blocks, and the rows above it and columns to its left
            R, C = transposed ? (gi, ki) : ((ki - 1) * length(groups) + gi, 1)
            top, left = (R - 1) * (nr + 1), (C - 1) * (nc + 1)

            for (i, parameter) in enumerate(parameters)
                row, col = cld(i, nc), mod1(i, nc)
                ax = Axis(layout[top + row, left + 1 + col];
                    title = _typeset(parameter),
                    limits = limits(parameter),
                    xticklabelsvisible = !shared || i + nc > d,
                    yticklabelsvisible = !shared || col == 1,
                    (fixed ? (; width = side, height = shared && !transposed ? 0.75 * side : side) : (;))...,
                    axis...
                )
                _draw!(ax, kind, panels, parameter, group, colors, ink)
            end

            Label(layout[top .+ (1:nr), left + 1], ylabel; rotation = π / 2, tellheight = false)
            Label(layout[top + nr + 1, left .+ (2:(nc + 1))], xlabel; tellwidth = false)
            rowgap!(layout, top + nr, 8)
            colgap!(layout, left + 1, 8)
            if length(groups) > 1 && C == blockcols   # name the estimator at the end of its row of panels
                Label(layout[top .+ (1:nr), blockcols * (nc + 1) + 1], only(group); rotation = -π / 2, font = :bold, tellheight = false)
            end
        end
    end

    if length(groups) == 1 && length(estimators) > 1
        markers = [MarkerElement(; color = colors[estimator], marker = :circle, markersize = 12) for estimator in estimators]
        Legend(fig[2, 1], markers, estimators;
            orientation = :horizontal, nbanks = cld(length(estimators), nc * blockcols + 1),
            framevisible = false, padding = 0, tellwidth = fixed
        )
        rowgap!(fig.layout, 1, 12)
    end
    fixed && resize_to_layout!(fig)

    return fig
end

# ---- Which plots to draw, and the data behind them ----

function _availableplots(assessment::Assessment)
    columns = names(assessment.estimates)
    isnothing(assessment.samples) || return (:recovery, :ecdf, :zscore)
    "prob" ∈ columns && return (:calibration,)
    ("estimate" ∈ columns || ["lower", "upper"] ⊆ columns) && return (:recovery,)
    throw(ArgumentError("unrecognised assessment format: expected the columns `estimate`, `lower` and `upper`, or `prob`"))
end

function _resolveplots(assessment::Assessment, plots)
    available = _availableplots(assessment)
    isnothing(plots) && return collect(available)
    kinds = plots isa Union{Symbol, AbstractString} ? [Symbol(plots)] : unique(Symbol.(collect(plots)))
    if isempty(kinds) || !(kinds ⊆ available)
        throw(ArgumentError("`plots` should contain one or more of $(join(repr.(available), ", ")) for this assessment; received $(repr(plots))"))
    end
    return kinds
end

# Long-form data behind each plot, with one row per mark and the columns `estimator` and `parameter` in every case
function _plotdata(assessment::Assessment, kinds, prob, difference)
    estimates = assessment.estimates
    posteriors = isnothing(assessment.samples) ? nothing : _posteriorsummaries(assessment.samples)
    data = Dict{Symbol, DataFrame}()
    for kind in kinds
        df = if kind === :calibration
            empiricalprob(assessment)
        elseif kind === :ecdf
            _ecdf(posteriors, prob, difference)
        elseif kind === :zscore
            filter([:zscore, :contraction] => (z, c) -> isfinite(z) && isfinite(c), posteriors)
        elseif isnothing(posteriors)
            estimates
        else
            # the point estimates honour the `pointsummary` given to assess(); the intervals come from the samples
            by = intersect(["estimator", "parameter", "k", "j"], names(estimates))
            innerjoin(select(estimates, by, :estimate, :truth), select(posteriors, by, :lower, :upper); on = by)
        end
        data[kind] = "estimator" ∈ names(df) ? df : insertcols(df, :estimator => "")
    end
    return data
end

# One row for each posterior distribution: its summaries, and those of the true value relative to it
function _posteriorsummaries(samples::DataFrame)
    rng = Xoshiro(1)
    by = intersect(["estimator", "parameter", "k", "j"], names(samples))
    df = combine(groupby(samples, by),
        :truth => first => :truth,
        :value => mean => :mean,
        :value => std => :sd,
        :value => (v -> quantile(v, 0.025)) => :lower,
        :value => (v -> quantile(v, 0.975)) => :upper,
        [:value, :truth] => ((v, t) -> _rank(rng, v, first(t))) => :rank,
        nrow => :draws;
        threads = false   # the groups share `rng`
    )
    "estimator" ∈ by || insertcols!(df, 1, :estimator => "")

    # the prior variance is estimated from the true values, which are draws from the prior
    transform!(groupby(df, [:estimator, :parameter]), :truth => var => :prior_variance)
    df.zscore = (df.mean .- df.truth) ./ df.sd
    df.contraction = 1 .- df.sd .^ 2 ./ df.prior_variance

    return df
end

# Number of posterior draws below the true value. Ties are broken at random, so that the rank is uniformly
# distributed under a calibrated posterior even if it has point masses (e.g., SpikeAndSlab)
function _rank(rng, draws, truth)
    ties = count(==(truth), draws)
    return count(<(truth), draws) + (ties > 0 ? rand(rng, 0:ties) : 0)
end

# Empirical distribution function of the ranks and its simultaneous confidence band, for each estimator and parameter
function _ecdf(posteriors::DataFrame, prob, difference)
    posteriors = filter(:j => ==(1), posteriors)   # the band assumes that the (θ, Z) pairs are independent
    bands = Dict{NTuple{2, Int}, NTuple{2, Vector{Int}}}()   # the band depends only on the numbers of ranks and draws
    combine(groupby(posteriors, [:estimator, :parameter]); threads = false) do df
        n, L = nrow(df), first(df.draws)
        # Evaluate at G thresholds on the rank scale; under calibration the rank is uniform on 0:L, so that
        # P(rank < threshold) = threshold / (L + 1) exactly
        G = min(L + 1, n, 1000)
        thresholds = [fld(i * (L + 1), G) for i in 0:G]
        x = thresholds ./ (L + 1)
        ranks = sort(df.rank)
        y = [searchsortedfirst(ranks, threshold) - 1 for threshold in thresholds] ./ n
        lower, upper = get!(() -> _ecdfband(n, x[2:end], prob), bands, (n, L))
        shift = difference ? x : zero(x)
        (; x, y = y .- shift, lower = [0; lower] ./ n .- shift, upper = [0; upper] ./ n .- shift, reference = x .- shift)
    end
end

"""
    _ecdfband(n, p, prob)

Simultaneous confidence band for the empirical distribution function of `n` independent
uniform variates evaluated at the increasing probabilities `p`, the last of which is 1
(Säilynoja et al., 2022). The band consists of pointwise binomial intervals whose level γ is
chosen as large as possible subject to the empirical distribution function lying within all
of them with probability at least `prob`. Returns the lower and upper limits as counts.
"""
function _ecdfband(n::Integer, p::AbstractVector, prob::Real)
    limits(γ) = (Int.(binominvcdf.(n, p, γ / 2)), Int.(binominvcdf.(n, p, 1 - γ / 2)))
    lo, hi = 0.0, 1 - prob
    _coverage(n, p, limits(hi)...) >= prob && return limits(hi)
    for _ in 1:25   # the coverage is non-increasing in γ, and the limits are integers
        γ = (lo + hi) / 2
        _coverage(n, p, limits(γ)...) >= prob ? (lo = γ) : (hi = γ)
    end
    return limits(lo)
end

# Probability that the number of the n variates below p[i] lies in lower[i]:upper[i] for every i. These counts are
# distributed as a Poisson process of rate n conditioned on its total being n, and the probability that the
# process is within the limits so far with a given current count follows from one truncated convolution per step.
function _coverage(n::Integer, p::AbstractVector, lower::Vector{Int}, upper::Vector{Int})
    inside, from, to = [1.0], 0, 0   # inside[x - from + 1] = P(within the limits so far, current count = x)
    previous = 0.0
    for i in eachindex(p)
        λ = n * (p[i] - previous)
        previous = p[i]
        increment = poispdf.(λ, 0:Int(poisinvcdf(λ, 1 - 1e-12)))
        updated = zeros(upper[i] - lower[i] + 1)
        for x in lower[i]:upper[i], x₀ in max(from, x - length(increment) + 1):min(to, x)
            updated[x - lower[i] + 1] += inside[x₀ - from + 1] * increment[x - x₀ + 1]
        end
        inside, from, to = updated, lower[i], upper[i]
    end
    return sum(inside) / poispdf(n, n)   # both limits equal n at the final probability of 1
end

# ---- Drawing ----

# Colours from the current theme: a palette colour for each estimator, and the text colour for reference marks
function _themecolors(estimators)
    palette = Makie.to_color.(Makie.to_value(Makie.theme(:palette).color))
    colors = Dict(estimator => palette[mod1(i, length(palette))] for (i, estimator) in enumerate(estimators))
    ink = Makie.to_color(Makie.to_value(Makie.theme(:textcolor)))
    return colors, ink
end

# Marker size and opacity for a scatter of n points, so that dense panels remain legible
_markersize(n) = n <= 100 ? 8 : n <= 500 ? 6 : 5
_opacity(n) = n <= 100 ? 0.9 : n <= 500 ? 0.7 : 0.5

function _axislabels(kind, assessment, difference)
    kind === :ecdf && return ("Fractional rank statistic", difference ? "ECDF difference" : "ECDF")
    kind === :zscore && return ("Posterior contraction", "Posterior z-score")
    kind === :calibration && return ("Probability level, τ", "Pr(Q(Z, τ) ≥ θ)")
    estimates = assessment.estimates
    interval = if !isnothing(assessment.samples)
        "95% credible interval"
    elseif "α" ∈ names(estimates)
        "$(round(Int, 100 * (1 - first(estimates.α))))% interval"
    elseif "lower" ∈ names(estimates)
        "interval"
    end
    ylabel = isnothing(interval) ? "Estimate" : "estimate" ∈ names(estimates) ? "Estimate ($interval)" : uppercasefirst(interval)
    return ("True value", ylabel)
end

# Axis limits (xmin, xmax, ymin, ymax) as a function of the parameter. Recovery panels each have their own scale,
# with common x and y limits so that the identity line is the diagonal; the panels of the other plots share a scale.
function _limits(kind, df, difference)
    if kind === :recovery
        columns = intersect(["truth", "estimate", "lower", "upper"], names(df))
        spans = Dict(key.parameter => _span(reduce(vcat, eachcol(sub[!, columns]))) for (key, sub) in pairs(groupby(df, :parameter)))
        return parameter -> (spans[parameter]..., spans[parameter]...)
    end
    unit = _span([0, 1], 0.03)
    limits = if kind === :zscore
        (_span([0; 1; df.contraction])..., (-1, 1) .* 1.05 .* max(3, maximum(abs, df.zscore; init = 0))...)
    elseif kind === :ecdf && difference
        extent = maximum(abs, [df.y; df.lower; df.upper])
        (unit..., _span([-extent, extent], 0.05)...)
    else
        (unit..., unit...)
    end
    return parameter -> limits
end

# Range of the finite values, padded on both sides
function _span(values, padding = 0.04)
    values = filter(isfinite, values)
    isempty(values) && return (0.0, 1.0)
    lo, hi = Float64.(extrema(values))
    lo == hi && return (lo - 1, hi + 1)
    return (lo - padding * (hi - lo), hi + padding * (hi - lo))
end

function _draw!(ax, kind, panels, parameter, estimators, colors, ink)
    subsets = [(estimator, get(panels, (; estimator, parameter), nothing)) for estimator in estimators]
    filter!(subset -> !isnothing(last(subset)), subsets)
    isempty(subsets) && return ax
    reference = (; color = (ink, 0.7), linewidth = 1, linestyle = :dash)

    if kind === :recovery
        # intervals beneath the points, and the identity line on top so that it is visible in dense panels
        for (estimator, df) in subsets
            hasproperty(df, :lower) || continue
            opacity = (hasproperty(df, :estimate) ? 0.5 : 1) * _opacity(nrow(df))
            rangebars!(ax, df.truth, df.lower, df.upper; color = (colors[estimator], opacity), linewidth = 1)
        end
        for (estimator, df) in subsets
            hasproperty(df, :estimate) || continue
            scatter!(ax, df.truth, df.estimate; color = (colors[estimator], _opacity(nrow(df))), markersize = _markersize(nrow(df)))
        end
        ablines!(ax, 0, 1; reference...)
    elseif kind === :ecdf
        bounds = last(first(subsets))
        if !all(df -> df.lower == bounds.lower && df.upper == bounds.upper, last.(subsets))
            throw(ArgumentError("the estimators were assessed with different numbers of data sets or posterior draws, so that their confidence bands differ; use `grid = true` to draw them separately"))
        end
        band!(ax, bounds.x, bounds.lower, bounds.upper; color = (ink, 0.12))
        lines!(ax, bounds.x, bounds.reference; reference...)
        for (estimator, df) in subsets
            lines!(ax, df.x, df.y; color = colors[estimator], linewidth = 2)
        end
    elseif kind === :zscore
        hlines!(ax, 0; reference...)
        for (estimator, df) in subsets
            scatter!(ax, df.contraction, df.zscore; color = (colors[estimator], _opacity(nrow(df))), markersize = _markersize(nrow(df)))
        end
    elseif kind === :calibration
        ablines!(ax, 0, 1; reference...)
        for (estimator, df) in subsets
            df = sort(df, :prob)
            scatterlines!(ax, df.prob, df.empirical_prob; color = colors[estimator], linewidth = 2, markersize = 9)
        end
    end

    return ax
end

# Parameter names are often written with Unicode subscripts (e.g., θ₁), which many fonts lack; typeset them instead
function _typeset(name)
    name = string(name)
    i = findfirst(in('₀':'₉'), name)
    (isnothing(i) || i == firstindex(name) || !all(in('₀':'₉'), name[i:end])) && return name
    return rich(name[1:prevind(name, i)], subscript(map(c -> '0' + (c - '₀'), name[i:end])))
end

end  # module
