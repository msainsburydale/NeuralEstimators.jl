module NeuralEstimatorsPlotExt

using NeuralEstimators
using AlgebraOfGraphics, CairoMakie
import CairoMakie: plot; export plot

using DataFrames
using Statistics: mean, var, std, quantile

# Thin wrappers around StatsFuns
using StatsFuns: binompdf, binominvcdf, hyperinvcdf
_binom_quantile(q::Float64, n::Int, p::Float64)::Int = Int(binominvcdf(n, p, q))
_binom_pdf(k::Int, n::Int, p::Float64)::Float64      = binompdf(n, p, k)
_hyper_quantile(q::Float64, ns::Int, nf::Int, n::Int)::Int = Int(hyperinvcdf(ns, nf, n, q))


# ===========================================================================
#  plot(assessment)
# ===========================================================================

"""
    plot(assessment::Assessment; grid::Bool = false, prob = 0.99)

Visualise the performance of a neural estimator. Accepts the `Assessment`
object returned by [`assess`](@ref).

!!! note "Extension"
    This function is defined in the `NeuralEstimatorsPlotExt` extension and
    requires `AlgebraOfGraphics` and `CairoMakie` to be loaded.

The plot produced depends on the type of estimator being assessed:

[`PointEstimator`](@ref): produces a scatter plot of estimates vs. true values,
faceted by parameter.

[`IntervalEstimator`](@ref): produces a plot of estimated credible intervals
vs. true values, faceted by parameter. Each interval is drawn as a vertical
line segment from lower to upper bound, with tick marks at the endpoints.

[`QuantileEstimator`](@ref): produces a calibration plot of the empirical
coverage probability vs. the nominal probability level τ, faceted by
parameter. A well-calibrated estimator will follow the red diagonal line.
Specifically, the diagnostic is constructed as follows:

1. For k = 1,…,K, sample pairs (θᵏ, Zᵏ) with θᵏ ∼ p(θ), Zᵏ ∼ p(Z ∣ θᵏ).
   This gives K "posterior draws", θᵏ ∼ p(θ ∣ Zᵏ).
2. For each k and each τ ∈ {τⱼ : j = 1,…,J}, estimate the posterior quantile
   Q(Zᵏ, τ).
3. For each τ, compute the proportion of quantiles Q(Zᵏ, τ) exceeding the
   corresponding θᵏ, and plot this proportion against τ.

[`PosteriorEstimator`](@ref): produces a three-row figure:

1. **Recovery plot**: posterior mean vs. true value (scatter), with vertical
   line segments showing the 95% posterior credible interval, faceted by
   parameter.
2. **ECDF plot**: for each parameter, the empirical CDF of the fractional rank
   of the true value within the posterior samples, together with a simultaneous
   `prob`-level confidence band. A well-calibrated posterior yields an ECDF
   that stays within the band.
3. **Z-score / contraction plot**: posterior z-score
   (posterior mean − truth) / posterior SD vs. posterior contraction
   1 − Var(posterior) / Var(prior), faceted by parameter. Ideally z-scores
   are centred near zero and contractions are near one.

# Keyword arguments

- `grid::Bool = false`: when comparing multiple estimators, set `grid = true`
  to facet by both estimator (columns) and parameter (rows), making
  between-estimator differences easier to read. When `false` (the default),
  estimators are overlaid on the same panel using distinct colours.
  (Currently only implemented for `PointEstimator` and `IntervalEstimator`.)
- `prob = 0.99`: nominal simultaneous coverage level for the SBC
  confidence band. Only used when `assessment` contains posterior samples.
"""
function plot(assessment::Assessment; grid::Bool = false, prob = 0.99)
    df = assessment.df

    # ---- PosteriorEstimator path ----
    if hasproperty(assessment, :samples) && !isnothing(assessment.samples)
        sbc = _sbc_data(assessment, prob)
        d   = length(sbc.params)
        fig = Figure(size = (200 * d, 750))
        _plot_recovery_row!(fig, 1, assessment, d)
        _plot_ecdf_row!(fig, 3, sbc, d)
        _plot_zscore_row!(fig, 5, sbc, d)
        return fig
    end

    # ---- QuantileEstimator path ----
    num_estimators = "estimator" ∉ names(df) ? 1 : length(unique(df.estimator))
    figure = mapping([0], [1]) * visual(ABLines, color = :red, linestyle = :dash)

    if "prob" ∈ names(df)
        df     = empiricalprob(assessment)
        figure = mapping([0], [1]) * visual(ABLines, color = :red, linestyle = :dash)
        figure += data(df) * mapping(:prob, :empirical_prob, layout = :parameter) * visual(Lines, color = :black)
        figure  = draw(figure, facet = (; linkxaxes = :none, linkyaxes = :none),
                       axis = (; xlabel = "Probability level, τ", ylabel = "Pr(Q(Z, τ) ≥ θ)"))
        return figure
    end

    # ---- IntervalEstimator path ----
    if all(["lower", "upper"] .∈ Ref(names(df)))
        df_stacked = stack(df, [:lower, :upper], variable_name = :bound, value_name = :interval)
        figure += data(df_stacked) * mapping(:truth, :interval, group = :k => nonnumeric, layout = :parameter) * visual(Lines, color = :black)
        figure += data(df_stacked) * mapping(:truth, :interval, layout = :parameter) * visual(Scatter, color = :black, marker = '⎯')
    end

    # ---- PointEstimator path ----
    linkyaxes = :none
    if "estimate" ∈ names(df)
        if num_estimators > 1
            if grid
                figure += data(df) * mapping(:truth, :estimate, color = :estimator, col = :estimator, row = :parameter) * visual(alpha = 0.75)
                linkyaxes = :minimal
            else
                figure += data(df) * mapping(:truth, :estimate, color = :estimator, layout = :parameter) * visual(alpha = 0.75)
            end
        else
            figure += data(df) * mapping(:truth, :estimate, layout = :parameter) * visual(color = :black, alpha = 0.75)
        end
    end

    figure += mapping([0], [1]) * visual(ABLines, color = :red, linestyle = :dash)
    figure  = draw(figure, facet = (; linkxaxes = :none, linkyaxes = linkyaxes))
    return figure
end


# ===========================================================================
#  Row-drawing helpers
# ===========================================================================

"""
Draw a recovery row into `fig` starting at grid row `row_start`.
Plots posterior mean ± 95% CI vs. true value, one panel per parameter.
Occupies grid rows `row_start` (axes) and `row_start+1` (x-axis label).
"""
function _plot_recovery_row!(fig::Figure, row_start::Int,
                             assessment::Assessment, d::Int)
    samples_df = assessment.samples
    params = unique(samples_df.parameter)
    K      = maximum(samples_df.k)

    for (pi, param) in enumerate(params)
        sub = filter(r -> r.parameter == param, samples_df)

        truths = Float64[]
        means  = Float64[]
        lowers = Float64[]
        uppers = Float64[]
        for k in 1:K
            sub_k = filter(r -> r.k == k, sub)
            isempty(sub_k) && continue
            vals = sub_k.value
            push!(truths, sub_k[1, :truth])
            push!(means,  mean(vals))
            push!(lowers, quantile(vals, 0.025))
            push!(uppers, quantile(vals, 0.975))
        end

        all_vals = vcat(truths, lowers, uppers)
        vmin, vmax = minimum(all_vals), maximum(all_vals)

        ax = Axis(fig[row_start, pi],
            title  = param,
            ylabel = pi == 1 ? "Posterior mean" : "",
            limits = (vmin, vmax, vmin, vmax),
        )

        for i in eachindex(truths)
            lines!(ax, [truths[i], truths[i]], [lowers[i], uppers[i]];
                color = (:black, 0.3), linewidth = 0.8)
        end

        scatter!(ax, truths, means;
            color = (:black, 0.6), markersize = 4)

        lines!(ax, [vmin, vmax], [vmin, vmax];
            color = :red, linestyle = :dash, linewidth = 1.2)
    end

    Label(fig[row_start + 1, 1:d], "True value"; tellwidth = false)
end

"""
Draw an ECDF row into `fig` starting at grid row `row_start`.
Occupies grid rows `row_start` (axes) and `row_start+1` (x-axis label).
"""
function _plot_ecdf_row!(fig::Figure, row_start::Int, sbc::NamedTuple, d::Int)
    for (pi, param) in enumerate(sbc.params)
        ax = Axis(fig[row_start, pi],
            ylabel = pi == 1 ? "ECDF" : "",
            xticks = [0.0, 0.5, 1.0],
            yticks = [0.0, 0.5, 1.0],
            limits = (0.0, 1.0, 0.0, 1.0),
        )

        band!(ax, sbc.band_df.x, sbc.band_df.lower, sbc.band_df.upper;
            color = (:skyblue, 0.4))
        lines!(ax, [0.0, 1.0], [0.0, 1.0];
            color = :skyblue, linewidth = 1.0)

        sub = filter(r -> r.parameter == param, sbc.ecdf_plot_df)
        sort!(sub, :z)
        stairs!(ax, sub.z, sub.ecdf;
            color = :black, linewidth = 1.2, step = :post)
    end

    Label(fig[row_start + 1, 1:d], "Fractional rank statistic"; tellwidth = false)
end

"""
Draw a z-score/contraction row into `fig` starting at grid row `row_start`.
Occupies grid rows `row_start` (axes) and `row_start+1` (x-axis label).
"""
function _plot_zscore_row!(fig::Figure, row_start::Int, sbc::NamedTuple, d::Int)
    contraction_xmin = min(0.0, minimum(sbc.zscore_df.contraction))

    for (pi, param) in enumerate(sbc.params)
        ax = Axis(fig[row_start, pi],
            ylabel = pi == 1 ? "Posterior z-score" : "",
            limits = (contraction_xmin, 1.0, nothing, nothing),
        )

        sub = filter(r -> r.parameter == param, sbc.zscore_df)
        scatter!(ax, sub.contraction, sub.z_score;
            color = (:black, 0.6), markersize = 5)
    end

    Label(fig[row_start + 1, 1:d], "Posterior contraction"; tellwidth = false)
end


# ===========================================================================
#  SBC data preparation
# ===========================================================================

"""
Compute all data needed for SBC plots from an `Assessment` with posterior
samples. Returns a `NamedTuple` with fields:
- `params`:       vector of parameter names
- `band_df`:      confidence-band DataFrame (columns `x`, `lower`, `upper`)
- `ecdf_plot_df`: long-form ECDF DataFrame (columns `z`, `parameter`, `ecdf`)
- `zscore_df`:    long-form DataFrame (columns `parameter`, `k`, `z_score`, `contraction`)
"""
function _sbc_data(assessment::Assessment, prob::Float64)
    samples_df = assessment.samples
    params = unique(samples_df.parameter)
    d  = length(params)
    K  = maximum(samples_df.k)

    # ---- ranks ----
    rank_rows = NamedTuple{(:parameter, :k, :rank), Tuple{String, Int, Int}}[]
    for param in params
        sub = filter(r -> r.parameter == param, samples_df)
        for k in 1:K
            sub_k = filter(r -> r.k == k, sub)
            isempty(sub_k) && continue
            truth_val = sub_k[1, :truth]
            push!(rank_rows, (parameter = param, k = k, rank = sum(sub_k.value .< truth_val)))
        end
    end
    ranks_df = DataFrame(rank_rows)

    ranks_matrix = Matrix{Int}(undef, K, d)
    for (pi, param) in enumerate(params)
        sub = filter(r -> r.parameter == param, ranks_df)
        for k in 1:K
            row = filter(r -> r.k == k, sub)
            ranks_matrix[k, pi] = isempty(row) ? 0 : row[1, :rank]
        end
    end

    max_rank  = maximum(ranks_matrix)
    N         = K
    grid_size = min(max_rank + 1, N)

    # ---- confidence band ----
    gamma   = _adjust_gamma_optimize(N, grid_size, prob)
    z_grid  = range(0, 1; length = grid_size + 1)
    z_twice = vcat(0.0, repeat(collect(z_grid[2:end]), inner = 2))

    lower_counts, upper_counts = _ecdf_intervals(N, 1, grid_size, gamma)
    band_df = DataFrame(
        x     = z_twice,
        lower = lower_counts ./ N,
        upper = upper_counts ./ N,
    )

    # ---- ECDF values ----
    base_vals = floor.(Int, (0:grid_size) .* ((max_rank + 1) / grid_size))
    ecdf_vals = Matrix{Float64}(undef, grid_size + 1, d)
    for i in 1:(grid_size + 1)
        for (pi, _) in enumerate(params)
            ecdf_vals[i, pi] = mean(ranks_matrix[:, pi] .< base_vals[i])
        end
    end

    ecdf_rows = NamedTuple{(:z, :parameter, :ecdf), Tuple{Float64, String, Float64}}[]
    for (pi, param) in enumerate(params)
        for i in 1:(grid_size + 1)
            push!(ecdf_rows, (z = z_grid[i], parameter = param, ecdf = ecdf_vals[i, pi]))
        end
    end
    ecdf_plot_df = DataFrame(ecdf_rows)

    # ---- z-scores and contraction ----
    prior_var = Dict{String, Float64}()
    for param in params
        truths = unique(filter(r -> r.parameter == param, samples_df)[!, [:k, :truth]])
        prior_var[param] = var(truths.truth)
    end

    zscore_rows = NamedTuple{(:parameter, :k, :z_score, :contraction), Tuple{String, Int, Float64, Float64}}[]
    for param in params
        sub = filter(r -> r.parameter == param, samples_df)
        for k in 1:K
            sub_k = filter(r -> r.k == k, sub)
            isempty(sub_k) && continue
            truth_val  = sub_k[1, :truth]
            mu_post    = mean(sub_k.value)
            sd_post    = std(sub_k.value)
            push!(zscore_rows, (
                parameter   = param,
                k           = k,
                z_score     = (mu_post - truth_val) / sd_post,
                contraction = 1.0 - sd_post^2 / prior_var[param],
            ))
        end
    end
    zscore_df = DataFrame(zscore_rows)

    return (; params, band_df, ecdf_plot_df, zscore_df)
end


# ===========================================================================
#  Internal distribution / band helpers
# ===========================================================================

function _ecdf_intervals(N::Int, L::Int, K::Int, gamma::Float64)
    z = range(0, 1; length = K + 1)

    if L == 1
        lower = [_binom_quantile(gamma / 2,     N, Float64(zi)) for zi in z]
        upper = [_binom_quantile(1 - gamma / 2, N, Float64(zi)) for zi in z]
    else
        n_fail = N * (L - 1)
        n_draw = N * L
        k_vec  = floor.(Int, collect(z) .* n_draw)
        lower  = [_hyper_quantile(gamma / 2,     N, n_fail, kk) for kk in k_vec]
        upper  = [_hyper_quantile(1 - gamma / 2, N, n_fail, kk) for kk in k_vec]
    end

    lower_step = vcat(repeat(lower[1:K], inner = 2), lower[K + 1])
    upper_step = vcat(repeat(upper[1:K], inner = 2), upper[K + 1])
    return lower_step, upper_step
end

function _adjust_gamma_optimize(N::Int, K::Int, conf_level::Float64)
    function target(gamma)
        z  = collect(range(0, 1; length = K))
        z1 = vcat(0.0, z[1:end-1])
        z2 = z

        x2_lower  = [_binom_quantile(gamma / 2, N, Float64(zi)) for zi in z2]
        rev_lower = reverse(x2_lower)
        x2_upper  = vcat(N .- rev_lower[2:K], N)

        x1_vec = [0]
        p_int  = [1.0]
        for i in eachindex(z1)
            x1_vec, p_int = _p_interior(p_int, x1_vec, x2_lower[i]:x2_upper[i], z1[i], z2[i], N)
        end
        abs(conf_level - sum(p_int))
    end

    φ = (√5 - 1) / 2
    a, b = 0.0, 1.0 - conf_level
    c, d_pt = b - φ * (b - a), a + φ * (b - a)
    fc, fd  = target(c), target(d_pt)
    while (b - a) > 1e-8
        if fc < fd
            b, d_pt, fd = d_pt, c, fc
            c  = b - φ * (b - a); fc = target(c)
        else
            a, c, fc = c, d_pt, fd
            d_pt = a + φ * (b - a); fd = target(d_pt)
        end
    end
    return (a + b) / 2
end

function _p_interior(p_int::Vector{Float64}, x1::Vector{Int},
                     x2_range::AbstractRange{Int},
                     z1::Float64, z2::Float64, N::Int)
    z_tilde  = (z2 - z1) / (1 - z1)
    x2_vec   = collect(x2_range)
    p_x2_int = zeros(length(x2_vec), length(x1))
    for (j, x1j) in enumerate(x1)
        N_tilde = N - x1j
        for (i, x2i) in enumerate(x2_vec)
            diff = x2i - x1j
            (diff < 0 || diff > N_tilde) && continue
            p_x2_int[i, j] = p_int[j] * _binom_pdf(diff, N_tilde, z_tilde)
        end
    end
    return x2_vec, vec(sum(p_x2_int; dims = 2))
end

end  # module NeuralEstimatorsPlotExt
