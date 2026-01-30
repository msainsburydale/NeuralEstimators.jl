nparams(model) = length(Flux.trainables(model)) > 0 ? sum(length, Flux.trainables(model)) : 0

# Drop fields from NamedTuple: https://discourse.julialang.org/t/filtering-keys-out-of-named-tuples/73564/8
drop(nt::NamedTuple, key::Symbol) = Base.structdiff(nt, NamedTuple{(key,)})
drop(nt::NamedTuple, keys::NTuple{N, Symbol}) where {N} = Base.structdiff(nt, NamedTuple{keys})

# Check element type of arbitrarily nested array: https://stackoverflow.com/a/41847530
nested_eltype(x) = nested_eltype(typeof(x))
nested_eltype(::Type{T}) where {T <: AbstractArray} = nested_eltype(eltype(T))
nested_eltype(::Type{T}) where {T} = T

# Subsetting Assessment
Base.getindex(A::Assessment, i::Integer) = getfield(A, i)
Base.getindex(A::Assessment, s::Symbol) = getfield(A, s)

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
```
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
```
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
Default implementations are provided for commonly used data formats.
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
	subsetdata(Z::V, i) where {V <: AbstractArray{A}} where {A <: Any}
	subsetdata(Z::A, i) where {A <: AbstractArray{T, N}} where {T, N}
	subsetdata(Z::G, i) where {G <: AbstractGraph}
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
```
using NeuralEstimators
using GraphNeuralNetworks
using Flux: batch

d = 1  # dimension of the response variable
n = 4  # number of observations in each realisation
m = 6  # number of replicates in each data set
K = 2  # number of data sets

# Array data
Z = [rand(n, d, m) for k ∈ 1:K]
subsetdata(Z, 2)   # extract second replicate from each data set
subsetdata(Z, 1:3) # extract first 3 replicates from each data set

# Graphical data
e = 8 # number of edges
Z = [batch([rand_graph(n, e, ndata = rand(d, n)) for _ ∈ 1:m]) for k ∈ 1:K]
subsetdata(Z, 2)   # extract second replicate from each data set
subsetdata(Z, 1:3) # extract first 3 replicates from each data set
```
"""
function subsetdata end

function subsetdata(Z::V, i) where {V <: AbstractVector{A}} where {A}
    subsetdata.(Z, Ref(i))
end

function subsetdata(tup::Tup, i) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B}
    Z = tup[1]
    X = tup[2]
    @assert length(Z) == length(X)
    (subsetdata(Z, i), X) # X is not subsetted because it is set-level information
end

function subsetdata(Z::A, i) where {A <: AbstractArray{T, N}} where {T, N}
    getobs(Z, i)
end



# ---- End test code ----

function _DataLoader(data, batchsize::Integer; shuffle = true, partial = false)
    oldstd = stdout
    redirect_stderr(devnull)
    data_loader = DataLoader(data, batchsize = batchsize, shuffle = shuffle, partial = partial)
    redirect_stderr(oldstd)
    return data_loader
end

# Here, we define _checkgpu() for the case that CUDA has not been loaded (so, we will be using the CPU)
# For the case that CUDA is loaded, _checkgpu() is overloaded in ext/NeuralEstimatorsCUDAExt.jl
# NB Julia complains if we overload functions in package extensions... to get around this, here we
# use a slightly different function signature (omitting ::Bool)
function _checkgpu(use_gpu; verbose::Bool = true)
    if verbose
        @info "Running on CPU"
    end
    device = cpu
    return (device)
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
```
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
