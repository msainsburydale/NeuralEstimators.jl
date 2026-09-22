"""
    MLP(in::Integer, out::Integer; kwargs...)

A traditional fully-connected multilayer perceptron (MLP) with input dimension `in` and output dimension `out`.

# Keyword arguments
- `depth::Integer = 2`: the number of hidden layers.
- `width::Integer = 128`: the width of each hidden layer.
- `activation = relu`: the activation function used in each hidden layer.
- `output_activation = identity`: the activation function used in the output layer.
- `backend::Union{Nothing, Module} = nothing`: the backend to use for constructing the network (e.g., `Lux` or `Flux`). If `nothing`, the backend is resolved automatically.
"""
function MLP(in::Integer, out::Integer; depth::Integer = 2, width::Integer = 128, activation = relu, output_activation = identity, backend::Union{Nothing, Module} = nothing, kwargs...)
    @assert depth >= 0
    B = _resolvebackend(backend)
    if depth == 0
        layers = Any[B.Dense(in => out, output_activation; kwargs...)]
    else
        layers = []
        push!(layers, B.Dense(in => width, activation; kwargs...))
        append!(layers, [B.Dense(width => width, activation; kwargs...) for _ ∈ 2:depth])
        push!(layers, B.Dense(width => out, output_activation; kwargs...))
    end

    return B.Chain(layers...)
end

"""
    MultiHeadMLP(in::Integer, out::Integer, num_heads::Integer; growing::Bool = false, kwargs...)

A multi-head MLP consisting of `num_heads` independent MLP heads, each with input dimension
`in` and output dimension `out`. The outputs of all heads are concatenated to give a final
output of dimension `num_heads * out`.

# Keyword arguments
- `growing::Bool = false`: if `true`, the input dimension of the `i`-th head is `in + i`, allowing each head to receive an incrementally larger input.
- `kwargs`: keyword arguments passed to [`MLP`](@ref).

# Examples
```julia
using NeuralEstimators, Lux, Random
rng = Random.default_rng()
Random.seed!(rng, 0)

# Dummy data
batchsize = 16
num_summaries = 5
num_parameters = 3
s = rand(Float32, num_summaries, batchsize)
θ = rand(Float32, num_parameters, batchsize)

m = MultiHeadMLP(num_summaries, 1, num_parameters)
ps, st = Lux.setup(rng, m)
m(s, ps, st)[1]

m = MultiHeadMLP(num_summaries + num_parameters, 1, num_parameters)
ps, st = Lux.setup(rng, m)
m(vcat(s, θ), ps, st)[1]

m = MultiHeadMLP(num_summaries, 1, num_parameters; growing = true)
sθ_split = Tuple(vcat(s, θ[1:i, :]) for i in 1:num_parameters)
ps, st = Lux.setup(rng, m)
m(sθ_split, ps, st)[1]
```
"""
function MultiHeadMLP(in::Integer, out::Integer, num_heads::Integer; growing::Bool = false, backend::Union{Nothing, Module} = nothing, kwargs...)
    # NB: heads are executed sequentially by `Parallel`. A more efficient implementation could use
    #     3D tensor weights with padded inputs stacked into a 3D tensor, using
    #     `NNlib.batched_mul` for parallel execution. But for typical `num_heads` values the
    #     sequential overhead is probably negligible.
    @assert num_heads > 0
    backend = _resolvebackend(backend)
    mlps = Tuple(MLP(in + (growing ? i : 0), out; backend = backend, kwargs...) for i = 1:num_heads)
    return backend.Parallel(vcat, mlps...)
end

"""
    ResidualBlock(filter, in => out; stride = 1, backend = nothing)

Basic residual block (see [here](https://en.wikipedia.org/wiki/Residual_neural_network#Basic_block)),
consisting of two sequential convolutional layers and a skip (shortcut) connection
that connects the input of the block directly to the output,
facilitating the training of deep networks.

# Examples
```julia
using NeuralEstimators, Flux
z = rand(16, 16, 1, 1)
b = ResidualBlock((3, 3), 1 => 32)
b(z)
```
"""
function ResidualBlock(filter, channels; stride = 1, backend::Union{Nothing, Module} = nothing)
    B = _resolvebackend(backend)
    lux = get(Base.loaded_modules, _LUX_UUID, nothing)
    is_lux = !isnothing(lux) && B === lux

    bias_kwarg = is_lux ? :use_bias : :bias
    id = is_lux ? lux.WrappedFunction(identity) : identity

    layer = B.Chain(
        B.Conv(filter, channels; stride = stride, pad = 1, bias_kwarg => false),
        B.BatchNorm(channels[2], relu),
        B.Conv(filter, channels[2] => channels[2]; pad = 1, bias_kwarg => false),
        B.BatchNorm(channels[2])
    )

    connection = if stride == 1 && channels[1] == channels[2]
        +
    else
        projection = B.Chain(
            B.Conv(ntuple(_ -> 1, length(filter)), channels; stride = stride, bias_kwarg => false),
            B.BatchNorm(channels[2])
        )
        B.Parallel(+, id, projection)
    end

    return B.Chain(B.SkipConnection(layer, connection), B.relu)
end

# ---- Structs for GNNs: Only compatible with Flux ----

"""
	PowerDifference(a, b)
Function ``f(x, y) = |\\tilde{a}x - (1-\\tilde{a})y|^{\\tilde{b}}``, where
``\\tilde{a} = \\text{sigmoid}(a) \\in (0, 1)`` and ``\\tilde{b} = \\text{softplus}(b) > 0``
are constrained transformations of the trainable parameters `a` and `b`.

# Examples
```julia
using NeuralEstimators

X = rand(5, 100)
Y = rand(5, 100)
f = PowerDifference(0, 1.55)
f(X, Y)   # two arg method
f((X, Y)) # tuple method
```
"""
struct PowerDifference{A, B}
    a::A
    b::B
end
PowerDifference() = PowerDifference([0.0f0], [1.55f0]) # default initial values chosen such that ã = 0.5 and b̃ ≈ 2
PowerDifference(a::Number, b::AbstractArray) = PowerDifference([a], b)
PowerDifference(a::AbstractArray, b::Number) = PowerDifference(a, [b])
# Kept as a single fused broadcast: with a non-dotted `-`, the two products and their
# difference each materialise, which costs three extra arrays the size of the (possibly
# very large) edge-feature tensor, plus their Zygote Dual copies.
(f::PowerDifference)(x, y) = abs.(sigmoid.(f.a) .* x .- (1 .- sigmoid.(f.a)) .* y) .^ softplus.(f.b)
(f::PowerDifference)(tup::Tuple) = f(tup[1], tup[2])

@doc raw"""
	IndicatorWeights(h_max, n_bins::Integer)
	(w::IndicatorWeights)(h::Matrix) 
For spatial locations $\boldsymbol{s}$ and  $\boldsymbol{u}$, creates a spatial weight function defined as

```math 
\boldsymbol{w}(\boldsymbol{s}, \boldsymbol{u}) \equiv (\mathbb{I}(h \in B_k) : k = 1, \dots, K)',
```

where $\mathbb{I}(\cdot)$ denotes the indicator function, 
$h \equiv \|\boldsymbol{s} - \boldsymbol{u} \|$ is the spatial distance between $\boldsymbol{s}$ and 
$\boldsymbol{u}$, and $\{B_k : k = 1, \dots, K\}$ is a set of $K =$`n_bins` equally-sized distance bins covering the spatial distances between 0 and `h_max`. 

# Examples 
```julia
using NeuralEstimators, GraphNeuralNetworks

h_max = 1
n_bins = 10
w = IndicatorWeights(h_max, n_bins)
h = rand(1, 30) # distances between 30 pairs of spatial locations 
w(h)
```
"""
struct IndicatorWeights{T}
    h_cutoffs::T
end
function IndicatorWeights(h_max, n_bins::Integer)
    h_cutoffs = range(0, stop = h_max, length = n_bins+1)
    h_cutoffs = collect(h_cutoffs)
    IndicatorWeights(h_cutoffs)
end
function (l::IndicatorWeights)(h::M) where {M <: AbstractMatrix{T}} where {T}
    h_cutoffs = l.h_cutoffs
    bins_upper = h_cutoffs[2:end]   # upper bounds of the distance bins
    bins_lower = h_cutoffs[1:(end - 1)] # lower bounds of the distance bins 
    N = [bins_lower[i:i] .< h .<= bins_upper[i:i] for i in eachindex(bins_upper)] # NB avoid scalar indexing by i:i
    N = reduce(vcat, N)
    # NB Float32.() rather than f32(): the comparisons above give a Bool array, and f32()
    # leaves a non-floating-point array untouched, so the result would be a BitMatrix that
    # cannot hold the normalised weights computed by SpatialGraphConv
    Float32.(N)
end
Optimisers.trainable(l::IndicatorWeights) = NamedTuple()

@doc raw"""
	KernelWeights(h_max, n_bins::Integer)
	(w::KernelWeights)(h::Matrix) 
For spatial locations $\boldsymbol{s}$ and  $\boldsymbol{u}$, creates a spatial weight function defined as

```math 
\boldsymbol{w}(\boldsymbol{s}, \boldsymbol{u}) \equiv (\exp(-(h - \mu_k)^2 / (2\sigma_k^2)) : k = 1, \dots, K)',
```

where $h \equiv \|\boldsymbol{s} - \boldsymbol{u}\|$ is the spatial distance between $\boldsymbol{s}$ and $\boldsymbol{u}$, and ${\mu_k : k = 1, \dots, K}$ and ${\sigma_k : k = 1, \dots, K}$ are the means and standard deviations of the Gaussian kernels for each bin, covering the spatial distances between 0 and h_max.

# Examples 
```julia
using NeuralEstimators, GraphNeuralNetworks

h_max = 1
n_bins = 10
w = KernelWeights(h_max, n_bins)
h = rand(1, 30) # distances between 30 pairs of spatial locations 
w(h)
```
"""
struct KernelWeights{T1, T2}
    mu::T1
    sigma::T2
end
function KernelWeights(h_max, n_bins::Integer)
    h_cutoffs = range(0, stop = h_max, length = n_bins+1)
    h_cutoffs = collect(h_cutoffs)
    mu = [(h_cutoffs[i] + h_cutoffs[i + 1]) / 2 for i = 1:n_bins] # midpoints of the intervals 
    sigma = [(h_cutoffs[i + 1] - h_cutoffs[i]) / 4 for i = 1:n_bins] # std dev so that 95% of mass is within the bin 
    mu = f32(mu)
    sigma = f32(sigma)
    KernelWeights(mu, sigma)
end
function (l::KernelWeights)(h::M) where {M <: AbstractMatrix{T}} where {T}
    mu = l.mu
    sigma = l.sigma
    N = [exp.(-(h .- mu[i:i]) .^ 2 ./ (2 * sigma[i:i] .^ 2)) for i in eachindex(mu)] # Gaussian kernel for each bin (NB avoid scalar indexing by i:i)
    N = reduce(vcat, N)
    f32(N)
end
Optimisers.trainable(l::KernelWeights) = NamedTuple()
