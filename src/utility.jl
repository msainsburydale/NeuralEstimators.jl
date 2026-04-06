# ---- Flux helpers ----

# Same as Flux but just defined here so that we can use it without loading the package

#TODO just use cpu_device() where relevant (no need to define this helper function)
cpu(x) = cpu_device()(x)

struct FluxEltypeAdaptor{T} end

Adapt.adapt_storage(::FluxEltypeAdaptor{T}, x::AbstractArray{<:AbstractFloat}) where {T<:AbstractFloat} = convert(AbstractArray{T}, x)
Adapt.adapt_storage(::FluxEltypeAdaptor{T}, x::AbstractArray{<:Complex{<:AbstractFloat}}) where {T<:AbstractFloat} = convert(AbstractArray{Complex{T}}, x)

_paramtype(::Type{T}, m) where T = fmap(adapt(FluxEltypeAdaptor{T}()), m)

# fastpath for arrays
_paramtype(::Type{T}, x::AbstractArray{<:AbstractFloat}) where {T<:AbstractFloat} = convert(AbstractArray{T}, x)
_paramtype(::Type{T}, x::AbstractArray{<:Complex{<:AbstractFloat}}) where {T<:AbstractFloat} = convert(AbstractArray{Complex{T}}, x)

f32(m) = _paramtype(Float32, m)

# ---- Device and memory management ----

"""
    _applywithdevice(network, z, ps = nothing, st = nothing; batchsize, kwargs...)

Internal helper that applies a `network` to data `z`, handling device placement, 
Float32 conversion, minibatching, and DeepSet input checking.

Always runs in testmode: this function is never called during gradient computation (i.e. never
inside `_risk`), so testmode is always appropriate. This includes the case of frozen summary
networks during training.

Keyword arguments are passed onto [`_resolvedevice`](@ref).
"""
function _applywithdevice end # methods defined in extensions

"""
    _resolvedevice(; device = nothing, use_gpu::Bool = true, verbose::Bool = true)

Returns the device to use for moving data and parameters during training or inference.

If `device` is explicitly provided, it is returned as-is. Accepted values are
`cpu_device()`, `gpu_device()`, or (when using Lux) `reactant_device()`, all from
[MLDataDevices.jl](https://github.com/LuxDL/MLDataDevices.jl).

If `device = nothing` (the default), the device is inferred from `use_gpu`: a GPU
is used if `use_gpu = true` and one is available, otherwise the CPU is used.

The `device` argument takes priority over `use_gpu`.
"""
function _resolvedevice(; use_gpu::Bool = true, device = nothing, verbose::Bool = true)
    isnothing(device) ? _getdevice(use_gpu; verbose = verbose) : device
end

function _getdevice(use_gpu::Bool; verbose::Bool = true)
    device = use_gpu ? gpu_device() : cpu_device()
    verbose && @info "Running on $(nameof(typeof(device)))"
    return device
end

# Here, we define _manualgc() for the case that CUDA has not been loaded (so, we will be using the CPU)
# For the case that CUDA is loaded, the function is overloaded in ext/NeuralEstimatorsCUDAExt.jl
# NB Julia complains if we overload functions in package extensions... to get around this, here we
# use a slightly different function signature (omitting ::Bool)
function _forcegc(verbose)
    if verbose
        @info "Forcing garbage collection..."
    end
    GC.gc(true)
    return nothing
end

# Wraps a bare array in a single-element vector when using a DeepSet-based network,
# allowing users to pass a single dataset without manually wrapping it in a vector
function _check_deepset_input(z)
    bare_array = typeof(z) <: AbstractArray && !(typeof(z) <: AbstractVector)
    bare_array_in_tuple = typeof(z) <: Tuple && !(typeof(z[1]) <: AbstractVector)
    if bare_array
        z = [z]
    elseif bare_array_in_tuple
        z = ([z[1]], z[2])
    end
    return z
end

@inline _uses_deepset(T::DataType) = T <: DeepSet || any(p -> _uses_deepset(p), T.parameters)
@inline _uses_deepset(T::Type) = false
@inline _uses_deepset(x) = _uses_deepset(typeof(x))


# ---- Backend helpers ----

const _LUX_UUID  = Base.PkgId(Base.UUID("b2108857-7c20-44ae-9111-449ecde12c47"), "Lux")
const _FLUX_UUID = Base.PkgId(Base.UUID("587475ba-b771-5e3f-ad9e-33799f191a9c"), "Flux")

function _resolvebackend(B::Union{Nothing, Module})
    lux_loaded  = haskey(Base.loaded_modules, _LUX_UUID)
    flux_loaded = haskey(Base.loaded_modules, _FLUX_UUID)
    if !isnothing(B)
        lux  = get(Base.loaded_modules, _LUX_UUID, nothing)
        flux = get(Base.loaded_modules, _FLUX_UUID, nothing)
        B === lux || B === flux || error("Backend must be either Lux or Flux.")
        return B
    end

    if lux_loaded && flux_loaded
        @warn "Both Lux and Flux are loaded; defaulting to Lux. Pass the backend explicitly to suppress this warning."
        return Base.loaded_modules[_LUX_UUID]
    elseif lux_loaded
        return Base.loaded_modules[_LUX_UUID]
    elseif flux_loaded
        return Base.loaded_modules[_FLUX_UUID]
    else
        error("Neither Lux nor Flux is loaded. Please load one of them.")
    end
end

function _backendof(network)
    lux  = get(Base.loaded_modules, _LUX_UUID, nothing)
    flux = get(Base.loaded_modules, _FLUX_UUID, nothing)
    if isnothing(lux) && isnothing(flux)
        error("One of Flux or Lux must be loaded; run `using Flux` or `using Lux`")
    elseif !isnothing(lux) && !isnothing(flux) # both loaded, so try to determine from the network
        if _is_lux_network(network, lux)
            return lux
        elseif _is_flux_network(network, flux)
            return flux
        else 
            error("Could not determine backend from network of type $(typeof(network)).")
        end
    elseif !isnothing(lux)
        return lux
    else
        return flux
    end
end

function _is_lux_network(network, lux, visited=Base.IdSet())
    network in visited && return false
    push!(visited, network)
    network isa lux.AbstractLuxLayer && return true
    for field in fieldnames(typeof(network))
        val = getfield(network, field)
        isstructtype(typeof(val)) && _is_lux_network(val, lux, visited) && return true
    end
    return false
end

function _is_flux_network(network, flux, visited=Base.IdSet())
    network in visited && return false
    push!(visited, network)
    network isa flux.Chain && return true
    for field in fieldnames(typeof(network))
        val = getfield(network, field)
        isstructtype(typeof(val)) && _is_flux_network(val, flux, visited) && return true
    end
    return false
end


# function _maybelux(estimator::NeuralEstimator, backend::Module)
#     if backend === get(Base.loaded_modules, _LUX_UUID, nothing)
#         @info "Wrapping estimator in a LuxEstimator to handle Lux parameters and states."
#         LuxEstimator(estimator)
#     else
#         estimator
#     end
# end

# ---- Misc. ----

"""
	rowwisenorm(A)
Computes the row-wise norm of a matrix `A`.
"""
rowwisenorm(A) = sqrt.(sum(abs2, A; dims = 2))

# Original discussion: https://groups.google.com/g/julia-users/c/UARlZBCNlng
vectotri_docs = """
	vectotril(v; strict = false)
	vectotriu(v; strict = false)
Converts a vector `v` of length ``d(d+1)÷2`` (a triangular number) into a
``d × d`` lower or upper triangular matrix.

If `strict = true`, the matrix will be *strictly* lower or upper triangular,
that is, a ``(d+1) × (d+1)`` triangular matrix with zero diagonal.

Note that the triangular matrix is constructed on the CPU, but the returned
matrix will be a GPU array if `v` is a GPU array. Note also that the
return type is not of type `Triangular` matrix (i.e., the zeros are
materialised) since `Triangular` matrices are not always compatible with other
GPU operations.

# Examples
```julia
using NeuralEstimators

d = 4
n = d*(d+1)÷2
v = collect(range(1, n))
vectotril(v)
vectotriu(v)
vectotril(v; strict = true)
vectotriu(v; strict = true)
```
"""

"$vectotri_docs"
function vectotril(v; strict::Bool = false)
    if strict
        vectotrilstrict(v)
    else
        ArrayType = containertype(v)
        T = eltype(v)
        v = cpu(v)
        n = length(v)
        d = (-1 + isqrt(1 + 8n)) ÷ 2
        d*(d+1)÷2 == n || error("vectotril: length of vector is not triangular")
        k = 0
        L = [i >= j ? (k+=1; v[k]) : zero(T) for i = 1:d, j = 1:d]
        convert(ArrayType, L)
    end
end

"$vectotri_docs"
function vectotriu(v; strict::Bool = false)
    if strict
        vectotriustrict(v)
    else
        ArrayType = containertype(v)
        T = eltype(v)
        v = cpu(v)
        n = length(v)
        d = (-1 + isqrt(1 + 8n)) ÷ 2
        d*(d+1)÷2 == n || error("vectotriu: length of vector is not triangular")
        k = 0
        U = [i <= j ? (k+=1; v[k]) : zero(T) for i = 1:d, j = 1:d]
        convert(ArrayType, U)
    end
end

function vectotrilstrict(v)
    ArrayType = containertype(v)
    T = eltype(v)
    v = cpu(v)
    n = length(v)
    d = (-1 + isqrt(1 + 8n)) ÷ 2 + 1
    d*(d-1)÷2 == n || error("vectotrilstrict: length of vector is not triangular")
    k = 0
    L = [i > j ? (k+=1; v[k]) : zero(T) for i = 1:d, j = 1:d]
    convert(ArrayType, L)
end

function vectotriustrict(v)
    ArrayType = containertype(v)
    T = eltype(v)
    v = cpu(v)
    n = length(v)
    d = (-1 + isqrt(1 + 8n)) ÷ 2 + 1
    d*(d-1)÷2 == n || error("vectotriustrict: length of vector is not triangular")
    k = 0
    U = [i < j ? (k+=1; v[k]) : zero(T) for i = 1:d, j = 1:d]
    convert(ArrayType, U)
end

# Get the non-parametrized type name: https://stackoverflow.com/a/55977768/16776594
"""
	containertype(A::Type)
	containertype(::Type{A}) where A <: SubArray
	containertype(a::A) where A
Returns the container type of its argument.

If given a `SubArray`, returns the container type of the parent array.

# Examples
```julia
a = rand(3, 4)
containertype(a)
containertype(typeof(a))
[containertype(x) for x ∈ eachcol(a)]
```
"""
containertype(A::Type) = Base.typename(A).wrapper
containertype(a::A) where {A} = containertype(A)
containertype(::Type{A}) where {A <: SubArray} = containertype(A.types[1])

"""
	numberreplicates(Z)

Generic function that returns the number of replicates in a given object.
Default implementations are provided for commonly used data formats, namely,
data stored as an `Array` or as a `GNNGraph`.
"""
function numberreplicates end

# fallback broadcasting method
function numberreplicates(Z::V) where {V <: AbstractVector{A}} where {A}
    numberreplicates.(Z)
end

# specific methods
function numberreplicates(Z::A) where {A <: AbstractArray{T, N}} where {T <: Union{Number, Missing}, N}
    size(Z, N)
end
function numberreplicates(Z::V) where {V <: AbstractVector{T}} where {T <: Union{Number, Missing}}
    numberreplicates(reshape(Z, :, 1))
end
function numberreplicates(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B}
    Z = tup[1]
    X = tup[2]
    @assert length(Z) == length(X)
    numberreplicates(Z)
end
function numberreplicates(tup::Tup) where {Tup <: Tuple{V₁, M}} where {V₁ <: AbstractVector{A}, M <: AbstractMatrix{T}} where {A, T}
    Z = tup[1]
    X = tup[2]
    @assert length(Z) == size(X, 2)
    numberreplicates(Z)
end

"""
	subsetreplicates(Z::V, i) where {V <: AbstractArray{A}} where {A <: Any}
	subsetreplicates(Z::A, i) where {A <: AbstractArray{T, N}} where {T, N}
	subsetreplicates(Z::G, i) where {G <: GNNGraph}
Return replicate(s) `i` from each data set in `Z`.

If working with data that are not covered by the default methods, overload the function with the appropriate type for `Z`.

For graphical data, calls
[`getgraph()`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/api/gnngraph/#GraphNeuralNetworks.GNNGraphs.getgraph-Tuple{GNNGraph,%20Int64}),
where the replicates are assumed be to stored as batched graphs. Since this can
be slow, one should consider using a method of [`train()`](@ref) that does not require
the data to be subsetted when working
with graphical data (use [`numberreplicates()`](@ref) to check that the training
and validation data sets are equally replicated, which prevents subsetting).

# Examples
```julia
using NeuralEstimators
using GraphNeuralNetworks
using Flux: batch

d = 1  # dimension of the response variable
n = 4  # number of observations in each realisation
m = 6  # number of replicates in each data set
K = 2  # number of data sets

# Array data
Z = [rand(n, d, m) for k ∈ 1:K]
subsetreplicates(Z, 2)   # extract second replicate from each data set
subsetreplicates(Z, 1:3) # extract first 3 replicates from each data set

# Graphical data
e = 8 # number of edges
Z = [batch([rand_graph(n, e, ndata = rand(d, n)) for _ ∈ 1:m]) for k ∈ 1:K]
subsetreplicates(Z, 2)   # extract second replicate from each data set
subsetreplicates(Z, 1:3) # extract first 3 replicates from each data set
```
"""
function subsetreplicates end

function subsetreplicates(Z::V, i) where {V <: AbstractVector{A}} where {A}
    subsetreplicates.(Z, Ref(i))
end

function subsetreplicates(tup::Tup, i) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B}
    Z = tup[1]
    X = tup[2]
    @assert length(Z) == length(X)
    (subsetreplicates(Z, i), X) # X is not subsetted because it is set-level information
end

function subsetreplicates(Z::A, i) where {A <: AbstractArray{T, N}} where {T, N}
    getobs(Z, i)
end


"""
    expandgrid(xs, ys)
Generates a grid of all possible combinations of the elements from two input vectors, `xs` and `ys`. 

Same as `expand.grid()` in `R`, but currently caters for two dimensions only.
"""
function expandgrid(xs, ys)
    lx, ly = length(xs), length(ys)
    lxly = lx*ly
    res = Array{Base.promote_eltype(xs, ys), 2}(undef, lxly, 2)
    ind = 1
    for y in ys, x in xs
        res[ind, 1] = x
        res[ind, 2] = y
        ind += 1
    end
    return res
end
expandgrid(N::Integer) = expandgrid(1:N, 1:N)

"""
    stackarrays(v::Vector{<:AbstractArray}; merge::Bool = true)
Stack a vector of arrays `v` into a single higher-dimensional array.

If all arrays have the same size along the last dimension, stacks along a new final dimension. Then, if `merge = true`, merges the last two dimensions into one.

Alternatively, if sizes differ along the last dimension, concatenates along the last dimension.

# Examples
```julia
# Vector containing arrays of the same size:
Z = [rand(2, 3, m) for m ∈ (1, 1)];
stackarrays(Z)
stackarrays(Z, merge = false)

# Vector containing arrays with differing final dimension size:
Z = [rand(2, 3, m) for m ∈ (1, 2)];
stackarrays(Z)
```
"""
function stackarrays(v::AbstractVector{A}; merge::Bool = true) where {A <: AbstractArray}
    N = ndims(v[1])  # number of dimensions of the arrays
    lastdims = size.(v, N)  # get size along last dimension for each array

    if length(unique(lastdims)) == 1
        a = cat(v...; dims = N+1)  # make a new (N+1)-dimensional array
        if merge
            sz = size(a)
            a = reshape(a, ntuple(i -> sz[i], N-1)..., sz[N]*sz[N + 1])  # merge last two dims
        end
    else
        if merge
            # Direct cat along last dimension
            a = cat(v...; dims = N)
        else
            error("Cannot stack arrays with differing sizes along dimension $N without merging.")
        end
    end

    return a
end
