# Lux-specific DeepSet methods. Included from NeuralEstimatorsLuxExt so that
# Lux/LuxCore stay weak dependencies. Shared packing and aggregation live in DeepSet.jl.

using NeuralEstimators: _aggregatereplicates, _stacksummaries, _rowofsummaries, _first_N_minus_1_dims_identical, @ignore_derivatives
import NeuralEstimators: _deepsetsummaries

# ---- Forward pass: (x, ps, st) -> (y, st) ----

function (d::DeepSet)(Z, ps, st)
    t, st_ψ = _deepsetsummaries(d, Z, ps.ψ, st.ψ)
    y, st_ϕ = d.ϕ(_stacksummaries(t), ps.ϕ, st.ϕ)
    return y, (ψ = st_ψ, ϕ = st_ϕ)
end

# Single data set
function _deepsetsummaries(d::DeepSet, Z, ps_ψ, st_ψ)
    ψZ, st_new = d.ψ(Z, ps_ψ, st_ψ)
    t = d.a(ψZ)
    if !isnothing(d.S)
        s = @ignore_derivatives d.S(Z)
        t = vcat(t, s)
    end
    return t, st_new
end

# Multiple data sets: general fallback, applying ψ to each data set independently
function _deepsetsummaries(d::DeepSet, Z::V, ps_ψ, st_ψ) where {V <: AbstractVector{A}} where {A}
    _deepsetsummaries_each(d, Z, ps_ψ, st_ψ)
end

# Multiple data sets: optimised version for array data
function _deepsetsummaries(d::DeepSet, Z::V, ps_ψ, st_ψ) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
    if _first_N_minus_1_dims_identical(Z)
        P = @ignore_derivatives PackedReplicates(Z)
        return _deepsetsummaries(d, P, ps_ψ, st_ψ)
    else
        return _deepsetsummaries_each(d, Z, ps_ψ, st_ψ)
    end
end

function _deepsetsummaries(d::DeepSet, P::PackedReplicates, ps_ψ, st_ψ)
    ψa, st_new = d.ψ(P.data, ps_ψ, st_ψ)
    t = _aggregatereplicates(d.a, ψa, P)
    if !isnothing(d.S)
        s = if isnothing(P.mask)
            @ignore_derivatives _rowofsummaries(d.S, P, t)
        else
            _rowofsummaries(d.S, P, t)
        end
        t = vcat(t, s)
    end
    return t, st_new
end

function _deepsetsummaries_each(d::DeepSet, Z, ps_ψ, st_ψ)
    t1, st_new = _deepsetsummaries(d, first(Z), ps_ψ, st_ψ)
    t = Vector{typeof(t1)}(undef, length(Z))
    t[1] = t1
    for i = 2:length(Z)
        t[i], st_new = _deepsetsummaries(d, Z[i], ps_ψ, st_ψ)
    end
    return t, st_new
end

# ---- LuxCore setup ----

function LuxCore.initialparameters(rng::AbstractRNG, d::DeepSet)
    (ψ = LuxCore.initialparameters(rng, d.ψ),
        ϕ = LuxCore.initialparameters(rng, d.ϕ))
end
function LuxCore.initialstates(rng::AbstractRNG, d::DeepSet)
    (ψ = LuxCore.initialstates(rng, d.ψ),
        ϕ = LuxCore.initialstates(rng, d.ϕ))
end
function LuxCore.parameterlength(d::DeepSet)
    LuxCore.parameterlength(d.ψ) + LuxCore.parameterlength(d.ϕ)
end
function LuxCore.statelength(d::DeepSet)
    LuxCore.statelength(d.ψ) + LuxCore.statelength(d.ϕ)
end
