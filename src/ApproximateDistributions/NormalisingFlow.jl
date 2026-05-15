@doc raw"""
    NormalisingFlow <: ApproximateDistribution
    NormalisingFlow(d::Integer, num_summaries::Integer; num_coupling_layer = 6, kwargs...)
A normalising flow for amortised inference with a [`PosteriorEstimator`](@ref), where `d` is the dimension of the parameter vector and `num_summaries` is the dimension of the summary statistics for the data.

Normalising flows are diffeomorphisms (i.e., invertible, differentiable transformations with differentiable inverses) that map a simple base distribution (e.g., standard Gaussian) to a more complex target distribution (e.g., the posterior). They achieve this by applying a sequence of learned transformations, the forms of which are chosen to be invertible and allow for tractable density computation via the change of variables formula. This allows for efficient density evaluation during the training stage, and efficient sampling during the inference stage. For further details, see the reviews by [Kobyzev et al. (2020)](https://ieeexplore.ieee.org/document/9089305) and [Papamakarios (2021)](https://dl.acm.org/doi/abs/10.5555/3546258.3546315).

`NormalisingFlow` uses affine coupling blocks (see [`AffineCouplingBlock`](@ref)), with optional activation normalisation ([`ActNorm`](@ref); [Kingma and Dhariwal, 2018](https://dl.acm.org/doi/10.5555/3327546.3327685)) and permutations applied between each block via [`CouplingLayer`](@ref). The base distribution is taken to be a standard multivariate Gaussian distribution.

When using a `NormalisingFlow` as the approximate distribution of a [`PosteriorEstimator`](@ref), the (learned) summary statistics are used to condition the affine coupling blocks at each layer.

# Keyword arguments
- `num_coupling_layers::Integer = 6`: number of coupling layers.
- `kwargs`: additional keyword arguments passed to [`CouplingLayer`](@ref) and [`AffineCouplingBlock`](@ref).
"""
@concrete struct NormalisingFlow <: ApproximateDistribution
    d
    layers
end

function NormalisingFlow(
    d::Integer,
    num_summaries::Integer;
    num_coupling_layers::Integer = 6,
    use_act_norm::Bool = true,
    backend::Union{Nothing, Module} = nothing,
    kwargs...
)
    @assert num_coupling_layers > 0
    backend = _resolvebackend(backend)
    layers = ntuple(_ -> CouplingLayer(d, num_summaries; backend = backend, use_act_norm = use_act_norm, kwargs...), num_coupling_layers)
    NormalisingFlow(d, layers)
end

numdistributionalparams(q::NormalisingFlow) = sum(numdistributionalparams.(q.layers))

# ---- Flux (stateful) -------------------------------------------------------

function forward(flow::NormalisingFlow, θ::AbstractMatrix, tz::AbstractMatrix)
    U = θ
    log_det_J = zero(similar(θ, 1, size(θ, 2)))
    for layer in flow.layers
        U, log_det = forward(layer, U, tz)
        log_det_J += log_det
    end
    return U, log_det_J
end

function inverse(flow::NormalisingFlow, U::AbstractMatrix, tz::AbstractMatrix)
    X = U
    for layer in reverse(flow.layers)
        X = inverse(layer, X, tz)
    end
    return X
end

function _logdensity(flow::NormalisingFlow, θ::AbstractMatrix, tz::AbstractMatrix)
    d, K = size(θ)
    @assert d == flow.d
    @assert K == size(tz, 2)
    U, log_det_J = forward(flow, θ, tz)
    log_densities = -0.5f0 * d * Float32(log(2π)) .- 0.5f0 * sum(U .* U; dims = 1) .+ log_det_J
    return log_densities
end

function sampleposterior(flow::NormalisingFlow, tz::AbstractMatrix, N::Integer; device::AbstractDevice)
    K = size(tz, 2)
    U = randn(Float32, flow.d, N * K)
    tz = repeat(tz, inner = (1, N))
    U = device(U)
    tz = device(tz)
    flow = device(flow)
    θ = inverse(flow, U, tz) |> cpu_device()
    return [θ[:, ((i - 1) * N + 1):(i * N)] for i = 1:K]
end

# ---- Lux (stateless) -------------------------------------------------------

function forward(
    flow::NormalisingFlow,
    θ::AbstractMatrix,
    tz::AbstractMatrix,
    ps,
    st::NamedTuple
)
    U = θ
    log_det_J = zero(similar(θ, 1, size(θ, 2)))
    new_st_layers = st.layers
    for (i, layer) in enumerate(flow.layers)
        U, log_det, layer_st_new = forward(layer, U, tz, ps.layers[i], new_st_layers[i])
        log_det_J = log_det_J .+ log_det
        new_st_layers = Base.setindex(new_st_layers, layer_st_new, i)
    end
    return U, log_det_J, merge(st, (layers = new_st_layers,))
end

function inverse(
    flow::NormalisingFlow,
    U::AbstractMatrix,
    tz::AbstractMatrix,
    ps,
    st::NamedTuple
)
    X = U
    new_st_layers = st.layers
    for i in reverse(eachindex(flow.layers))
        X, layer_st_new = inverse(flow.layers[i], X, tz, ps.layers[i], new_st_layers[i])
        new_st_layers = Base.setindex(new_st_layers, layer_st_new, i)
    end
    return X, merge(st, (layers = new_st_layers,))
end

function _logdensity(
    flow::NormalisingFlow,
    θ::AbstractMatrix,
    tz::AbstractMatrix,
    ps,
    st::NamedTuple
)
    d, K = size(θ)
    @assert d == flow.d
    @assert K == size(tz, 2)
    U, log_det_J, new_st = forward(flow, θ, tz, ps, st)
    log_densities = -0.5f0 * d * Float32(log(2π)) .- 0.5f0 * sum(U .* U; dims = 1) .+ log_det_J
    return log_densities, new_st
end

function sampleposterior(
    flow::NormalisingFlow,
    tz::AbstractMatrix,
    N::Integer,
    ps,
    st::NamedTuple;
    device = cpu_device()
)
    K = size(tz, 2)
    U = randn(Float32, flow.d, N * K)
    tz = repeat(tz, inner = (1, N))
    ps = device(ps)
    st = device(st)
    U = device(U)
    tz = device(tz)
    θ, _ = inverse(flow, U, tz, ps, st)
    θ = cpu_device()(θ)
    samples = [θ[:, ((i - 1) * N + 1):(i * N)] for i = 1:K]
    return samples
end

# --------------------------------------------------------------------------
# CouplingLayer
# --------------------------------------------------------------------------

"""
    CouplingLayer(d, num_summaries; use_act_norm = true, use_permutation = true, kwargs...)

A coupling layer used in a [`NormalisingFlow`](@ref), combining two
[`AffineCouplingBlock`](@ref)s with optional activation normalisation and permutation.

The layer splits its `d`-dimensional input into a lower half of dimension
`d₁ = div(d, 2)` and an upper half of dimension `d₂ = d - d₁`. The two halves are then passed
through a pair of affine coupling blocks in sequence: the first block transforms the lower
half conditioned on the upper, and the second block transforms the upper half conditioned on
the already-transformed lower half. This ensures every component is updated in a single forward pass, unlike a standard coupling layer where one half is left unchanged. When `d` = 1, the layer reduces to a single affine transformation of the one component conditioned on the summary statistics.

Optionally, activation normalisation ([`ActNorm`](@ref)) is applied before the coupling
blocks, and a random [`Permutation`](@ref) is applied after.

The argument `num_summaries` is the dimension of the conditioning summary statistics
(see [`PosteriorEstimator`](@ref)) and `kwargs` are passed to [`AffineCouplingBlock`](@ref).
"""
@concrete struct CouplingLayer
    d
    d₁
    d₂
    block1
    block2
    actnorm     # ActNorm or nothing
    permutation # Permutation or nothing
end

function CouplingLayer(
    d::Integer,
    num_summaries::Integer;
    use_act_norm::Bool = true,
    use_permutation::Bool = true,
    kwargs...
)
    d₁ = div(d, 2)
    d₂ = d - d₁
    block1 = d > 1 ? AffineCouplingBlock(d₂, num_summaries, d₁; kwargs...) : nothing
    block2 = AffineCouplingBlock(d₁, num_summaries, d₂; kwargs...)
    actnorm = use_act_norm ? ActNorm(d) : nothing
    permutation = (use_permutation && d > 1) ? Permutation(d) : nothing
    CouplingLayer(d, d₁, d₂, block1, block2, actnorm, permutation)
end

numdistributionalparams(::Nothing) = 0
numdistributionalparams(layer::CouplingLayer) = numdistributionalparams(layer.block1) + numdistributionalparams(layer.block2)

# Flux (stateful)
forward(::Nothing, θ1, θ2, tz) = θ1, zero(similar(θ2, 1, size(θ2, 2)))
inverse(::Nothing, θ2, U1, tz) = U1

function forward(layer::CouplingLayer, θ::AbstractMatrix, tz::AbstractMatrix)
    log_det_J = zero(similar(θ, 1, size(θ, 2)))
    if !isnothing(layer.actnorm)
        θ, log_det_act = forward(layer.actnorm, θ)
        log_det_J = log_det_J .+ log_det_act
    end
    if !isnothing(layer.permutation)
        θ = forward(layer.permutation, θ)
    end
    θ1 = layer.d₁ > 0 ? θ[1:layer.d₁, :] : similar(θ, 0, size(θ, 2))
    θ2 = θ[(layer.d₁ + 1):end, :]
    U1, ldj1 = forward(layer.block1, θ1, θ2, tz)
    U2, ldj2 = forward(layer.block2, θ2, U1, tz)
    return vcat(U1, U2), log_det_J .+ ldj1 .+ ldj2
end

function inverse(layer::CouplingLayer, U::AbstractMatrix, tz::AbstractMatrix)
    U1 = layer.d₁ > 0 ? U[1:layer.d₁, :] : similar(U, 0, size(U, 2))
    U2 = U[(layer.d₁ + 1):end, :]
    θ2 = inverse(layer.block2, U1, U2, tz)
    θ1 = inverse(layer.block1, θ2, U1, tz)
    θ = vcat(θ1, θ2)
    if !isnothing(layer.permutation)
        ;
        θ = inverse(layer.permutation, θ);
    end
    if !isnothing(layer.actnorm)
        ;
        θ = inverse(layer.actnorm, θ);
    end
    return θ
end

# Lux (stateless)
forward(::Nothing, θ1, θ2, tz, ps, st) = θ1, zero(similar(θ2, 1, size(θ2, 2))), st
inverse(::Nothing, θ2, U1, tz, ps, st) = U1, st

function forward(
    layer::CouplingLayer,
    θ::AbstractMatrix,
    tz::AbstractMatrix,
    ps,
    st::NamedTuple
)
    log_det_J = zero(similar(θ, 1, size(θ, 2)))
    if !isnothing(layer.actnorm)
        θ, log_det_act, st_an = forward(layer.actnorm, θ, ps.actnorm, st.actnorm)
        log_det_J = log_det_J .+ log_det_act
    else
        st_an = NamedTuple()
    end
    if !isnothing(layer.permutation)
        θ, st_perm = forward(layer.permutation, θ, ps.permutation, st.permutation)
    else
        st_perm = NamedTuple()
    end
    θ1 = layer.d₁ > 0 ? θ[1:layer.d₁, :] : similar(θ, 0, size(θ, 2))
    θ2 = θ[(layer.d₁ + 1):end, :]
    U1, ldj1, st_b1 = forward(layer.block1, θ1, θ2, tz, ps.block1, st.block1)
    U2, ldj2, st_b2 = forward(layer.block2, θ2, U1, tz, ps.block2, st.block2)
    new_st = (block1 = st_b1, block2 = st_b2, actnorm = st_an, permutation = st_perm)
    return vcat(U1, U2), log_det_J .+ ldj1 .+ ldj2, new_st
end

function inverse(
    layer::CouplingLayer,
    U::AbstractMatrix,
    tz::AbstractMatrix,
    ps,
    st::NamedTuple
)
    U1 = layer.d₁ > 0 ? U[1:layer.d₁, :] : similar(U, 0, size(U, 2))
    U2 = U[(layer.d₁ + 1):end, :]
    θ2, st_b2 = inverse(layer.block2, U1, U2, tz, ps.block2, st.block2)
    θ1, st_b1 = inverse(layer.block1, θ2, U1, tz, ps.block1, st.block1)
    θ = vcat(θ1, θ2)
    if !isnothing(layer.permutation)
        θ, st_perm = inverse(layer.permutation, θ, ps.permutation, st.permutation)
    else
        st_perm = NamedTuple()
    end
    if !isnothing(layer.actnorm)
        θ, st_an = inverse(layer.actnorm, θ, ps.actnorm, st.actnorm)
    else
        st_an = NamedTuple()
    end
    return θ, (block1 = st_b1, block2 = st_b2, actnorm = st_an, permutation = st_perm)
end

# --------------------------------------------------------------------------
# AffineCouplingBlock
# --------------------------------------------------------------------------

@doc raw"""
    AffineCouplingBlock(κ₁::MLP, κ₂::MLP)
    AffineCouplingBlock(d₁::Integer, num_summaries::Integer, d₂; kwargs...)
An affine coupling block used in a [`NormalisingFlow`](@ref). 

An affine coupling block splits its input $\boldsymbol{\theta}$ into two disjoint components, $\boldsymbol{\theta}_1$ and $\boldsymbol{\theta}_2$, with dimensions $d_1$ and $d_2$, respectively. The block then applies the following transformation: 
```math
\begin{aligned}
    \tilde{\boldsymbol{\theta}}_1 &= \boldsymbol{\theta}_1,\\
    \tilde{\boldsymbol{\theta}}_2 &= \boldsymbol{\theta}_2 \odot \exp\{\boldsymbol{\kappa}_{\boldsymbol{\gamma},1}(\tilde{\boldsymbol{\theta}}_1, \boldsymbol{T}(\boldsymbol{Z}))\} + \boldsymbol{\kappa}_{\boldsymbol{\gamma},2}(\tilde{\boldsymbol{\theta}}_1, \boldsymbol{T}(\boldsymbol{Z})),
\end{aligned}
```
where $\boldsymbol{\kappa}_{\boldsymbol{\gamma},1}(\cdot)$ and $\boldsymbol{\kappa}_{\boldsymbol{\gamma},2}(\cdot)$ are generic, non-invertible multilayer perceptrons (MLPs) that are functions of both the (transformed) first input component $\tilde{\boldsymbol{\theta}}_1$ and the learned $d^*$-dimensional summary statistics $\boldsymbol{T}(\boldsymbol{Z})$ (see [`PosteriorEstimator`](@ref)). 

To prevent numerical overflows and stabilise the training of the model, the scaling factors $\boldsymbol{\kappa}_{\boldsymbol{\gamma},1}(\cdot)$ are clamped using the function 
```math
f(\boldsymbol{s}) = \frac{2c}{\pi}\tan^{-1}(\frac{\boldsymbol{s}}{c}),
```
where $c = 1.9$ is a fixed clamping threshold. This transformation ensures that the scaling factors do not grow excessively large.

Additional keyword arguments `kwargs` are passed to the [`MLP`](@ref) constructor when creating `κ₁` and `κ₂`. 
"""
struct AffineCouplingBlock{M, D}
    scale::M
    translate::M
    d₁::D
    d₂::D
end

function AffineCouplingBlock(
    d₁::Integer,
    num_summaries::Integer,
    d₂::Integer;
    backend::Union{Nothing, Module} = nothing,
    kwargs...
)
    backend = _resolvebackend(backend)
    scale = MLP(d₁ + num_summaries, d₂; backend = backend, kwargs...)
    translate = MLP(d₁ + num_summaries, d₂; backend = backend, kwargs...)
    AffineCouplingBlock(scale, translate, d₁, d₂)
end

numdistributionalparams(block::AffineCouplingBlock) = 2 * block.d₂

const clamp_value = 1.9f0
const softclamp_scale = 2.0f0 * clamp_value / Float32(π)
softclamp(s) = softclamp_scale .* atan.(s ./ clamp_value)

# Flux (stateful)
function forward(net::AffineCouplingBlock, X, Y, tz)
    S = softclamp(net.scale(vcat(Y, tz)))
    T = net.translate(vcat(Y, tz))
    return X .* exp.(S) .+ T, sum(S; dims = 1)
end

function inverse(net::AffineCouplingBlock, U, V, tz)
    S = softclamp(net.scale(vcat(U, tz)))
    T = net.translate(vcat(U, tz))
    return (V .- T) .* exp.(-S)
end

# Lux (stateless)
function forward(net::AffineCouplingBlock, X, Y, tz, ps, st::NamedTuple)
    inp = vcat(Y, tz)
    s_raw, st_scale = net.scale(inp, ps.scale, st.scale)
    t_val, st_trans = net.translate(inp, ps.translate, st.translate)
    S = softclamp(s_raw)
    return X .* exp.(S) .+ t_val, sum(S; dims = 1), (scale = st_scale, translate = st_trans)
end

function inverse(net::AffineCouplingBlock, U, V, tz, ps, st::NamedTuple)
    inp = vcat(U, tz)
    s_raw, st_scale = net.scale(inp, ps.scale, st.scale)
    t_val, st_trans = net.translate(inp, ps.translate, st.translate)
    S = softclamp(s_raw)
    return (V .- t_val) .* exp.(-S), (scale = st_scale, translate = st_trans)
end

# --------------------------------------------------------------------------
# ActNorm
# --------------------------------------------------------------------------

#TODO data-dependent initialisation. That is, initialise scale/bias based on a first batch of data (could introduce another field `initialised` that get set to `true` when we've run through data; something similar is done in Lux.TrainState, see their source code)

""" 
    ActNorm(d::Integer)
Activation normalisation layer [Kingma and Dhariwal, 2018](https://dl.acm.org/doi/10.5555/3327546.3327685) for an input of dimension `d`.
"""
@concrete struct ActNorm
    scale
    bias
end
# NB `scale` and `bias` stored in the struct serve double duty:
# - Flux: they are the trainable parameters (collected by Flux.params).
# - Lux:  they seed `initialparameters`; the live values come from `ps` at runtime.

ActNorm(d::Integer) = ActNorm(ones(Float32, d, 1), zeros(Float32, d, 1))

# Flux (stateful)
function forward(an::ActNorm, θ::AbstractMatrix)
    U = an.scale .* θ .+ an.bias
    return U, sum(log.(abs.(an.scale)))
end
inverse(an::ActNorm, U::AbstractMatrix) = (U .- an.bias) ./ an.scale

# Lux (stateless) — ps carries scale/bias; struct fields are ignored at runtime
function forward(::ActNorm, θ::AbstractMatrix, ps, st::NamedTuple)
    U = ps.scale .* θ .+ ps.bias
    return U, sum(log.(abs.(ps.scale))), st
end
inverse(::ActNorm, U::AbstractMatrix, ps, st::NamedTuple) = ((U .- ps.bias) ./ ps.scale, st)

# --------------------------------------------------------------------------
# Permutation
# --------------------------------------------------------------------------

"""
    Permutation(in::Integer)
A layer that permutes the inputs (of dimension `in`) entering a coupling block. 

Note that a permutation layer is invertible with Jacobian determinant |J| = 1. 
"""
@concrete struct Permutation
    permutation
    inv_permutation
end

function Permutation(d::Integer)
    perm = randperm(d)
    inv_perm = sortperm(perm)
    Permutation(perm, inv_perm)
end

# Flux (stateful)
forward(l::Permutation, θ::AbstractMatrix) = θ[l.permutation, :]
inverse(l::Permutation, U::AbstractMatrix) = U[l.inv_permutation, :]

# Lux (stateless) — ps/st are empty; just thread them through
forward(l::Permutation, θ::AbstractMatrix, ps, st::NamedTuple) = θ[l.permutation, :], st
inverse(l::Permutation, U::AbstractMatrix, ps, st::NamedTuple) = U[l.inv_permutation, :], st
