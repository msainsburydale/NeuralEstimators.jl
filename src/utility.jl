nparams(model) = length(Flux.trainables(model)) > 0 ? sum(length, Flux.trainables(model)) : 0

# Drop fields from NamedTuple: https://discourse.julialang.org/t/filtering-keys-out-of-named-tuples/73564/8
drop(nt::NamedTuple, key::Symbol) =  Base.structdiff(nt, NamedTuple{(key,)})
drop(nt::NamedTuple, keys::NTuple{N,Symbol}) where {N} = Base.structdiff(nt, NamedTuple{keys})

# Check element type of arbitrarily nested array: https://stackoverflow.com/a/41847530
nested_eltype(x) = nested_eltype(typeof(x))
nested_eltype(::Type{T}) where T <:AbstractArray = nested_eltype(eltype(T))
nested_eltype(::Type{T}) where T = T

# Subsetting Assessment
Base.getindex(A::Assessment, i::Integer) = getfield(A, i)
Base.getindex(A::Assessment, s::Symbol) = getfield(A, s)

"""
	rowwisenorm(A)
Computes the row-wise norm of a matrix `A`.
"""
rowwisenorm(A) = sqrt.(sum(abs2,A; dims = 2))

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
		L = [ i >= j ? (k+=1; v[k]) : zero(T) for i=1:d, j=1:d ]
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
		U = [ i <= j ? (k+=1; v[k]) : zero(T) for i=1:d, j=1:d ]
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
	L = [ i > j ? (k+=1; v[k]) : zero(T) for i=1:d, j=1:d ]
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
	U = [ i < j ? (k+=1; v[k]) : zero(T) for i=1:d, j=1:d ]
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
containertype(a::A) where A = containertype(A)
containertype(::Type{A}) where A <: SubArray = containertype(A.types[1])

"""
	numberreplicates(Z)

Generic function that returns the number of replicates in a given object.
Default implementations are provided for commonly used data formats, namely,
data stored as an `Array` or as a `GNNGraph`.
"""
function numberreplicates end

# fallback broadcasting method
function numberreplicates(Z::V) where {V <: AbstractVector{A}} where A
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
function numberreplicates(Z::G) where {G <: GNNGraph}
	x = :Z ∈ keys(Z.ndata) ? Z.ndata.Z : first(values(Z.ndata))
	if ndims(x) == 3
		size(x, 2)
	else
		Z.num_graphs
	end
end


#TODO Recall that I set the code up to have ndata as a 3D array; with this format,
#     non-parametric bootstrap would be exceedingly fast (since we can subset the array data, I think).
"""
	subsetdata(Z::V, i) where {V <: AbstractArray{A}} where {A <: Any}
	subsetdata(Z::A, i) where {A <: AbstractArray{T, N}} where {T, N}
	subsetdata(Z::G, i) where {G <: AbstractGraph}
Return replicate(s) `i` from each data set in `Z`.

If the user is working with data that are not covered by the default methods,
simply overload the function with the appropriate type for `Z`.

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


function subsetdata(Z::G, i) where {G <: AbstractGraph}
	if typeof(i) <: Integer i = i:i end
	sym = collect(keys(Z.ndata))[1]
	if ndims(Z.ndata[sym]) == 3
		GNNGraph(Z; ndata = Z.ndata[sym][:, i, :])
	else
		# @warn "`subsetdata()` is slow for graphical data."
		# TODO getgraph() doesn't currently work with the GPU: see https://github.com/CarloLucibello/GraphNeuralNetworks.jl/issues/161
		# TODO getgraph() doesn’t return duplicates. So subsetdata(Z, [1, 1]) returns just a single graph
		# TODO can't check for CuArray (and return to GPU) because CuArray won't always be defined (no longer depend on CUDA) and we can't overload exact signatures in package extensions... it's low priority, but will be good to fix when time permits. Hopefully, the above issue with GraphNeuralNetworks.jl will get fixed, and we can then just remove the call to cpu() below
		#flag = Z.ndata[sym] isa CuArray
		Z = cpu(Z)
		Z = getgraph(Z, i)
		#if flag Z = gpu(Z) end
		Z
	end
end

# ---- Test code for GNN ----

# n = 250  # number of observations in each realisation
# m = 100  # number of replicates in each data set
# d = 1    # dimension of the response variable
# K = 1000  # number of data sets
#
# # Array data
# Z = [rand(n, d, m) for k ∈ 1:K]
# @elapsed subsetdata(Z_array, 1:3) # ≈ 0.03 seconds
#
# # Graphical data
# e = 100 # number of edges
# Z = [batch([rand_graph(n, e, ndata = rand(d, n)) for _ ∈ 1:m]) for k ∈ 1:K]
# @elapsed subsetdata(Z, 1:3) # ≈ 2.5 seconds
#
# # Graphical data: efficient storage
# Z2 = [rand_graph(n, e, ndata = rand(d, m, n)) for k ∈ 1:K]
# @elapsed subsetdata(Z2, 1:3) # ≈ 0.13 seconds

# ---- End test code ----

# Wrapper to ensure that the number of dimensions in the subsetted Z is preserved
# This causes dispatch ambiguity; instead, convert i to a range with each method
# subsetdata(Z, i::Int) = subsetdata(Z, i:i)

function subsetdata(Z::V, i) where {V <: AbstractVector{A}} where A
	subsetdata.(Z, Ref(i))
end

function subsetdata(tup::Tup, i) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B}
	Z = tup[1]
	X = tup[2]
	@assert length(Z) == length(X)
	(subsetdata(Z, i), X) # X is not subsetted because it is set-level information
end

function subsetdata(Z::A, i) where {A <: AbstractArray{T, N}} where {T, N}
	if typeof(i) <: Integer i = i:i end
	colons  = ntuple(_ -> (:), N - 1)
	Z[colons..., i]
end

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
	if verbose @info "Running on CPU" end
	device = cpu
	return(device)
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


# ---- Helper functions ----

"""
	_getindices(v::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}

Suppose that a vector of N-dimensional arrays, v = [A₁, A₂, ...], where the size
of the last dimension of each Aᵢ may vary, are concatenated along the dimension
N to form one large N-dimensional array, A. Then, this function returns the
indices of A (along dimension N) associated with each Aᵢ.

# Examples
```
v = [rand(16, 16, 1, m) for m ∈ (3, 4, 6)]
_getindices(v)
```
"""
function _getindices(v::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
	mᵢ  = size.(v, N) # number of independent replicates for every element in v
	cs  = cumsum(mᵢ)
	indices = [(cs[i] - mᵢ[i] + 1):cs[i] for i ∈ eachindex(v)]
	return indices
end


function _mergelastdims(X::A) where {A <: AbstractArray{T, N}} where {T, N}
	reshape(X, size(X)[1:(end - 2)]..., :)
end

"""
	stackarrays(v::V; merge = true) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
Stacks a vector of arrays `v` along the last dimension of each array, optionally merging the final dimension of the stacked array.

The arrays must be of the same size for the first `N-1` dimensions. However, if
`merge = true`, the size of the final dimension can vary.

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
function stackarrays(v::V; merge::Bool = true) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}

	m = size.(v, N) # last-dimension size for each array in v

	if length(unique(m)) == 1

		# Lazy-loading via the package RecursiveArrayTools. This is much faster
		# than cat(v...) when length(v) is large. However, this requires mᵢ = mⱼ ∀ i, j,
		# where mᵢ denotes the size of the last dimension of the array vᵢ.
		v = VectorOfArray(v)
		a = convert(containertype(A), v)            # (N + 1)-dimensional array
		if merge a = _mergelastdims(a) end  # N-dimensional array

	else

		if merge
			#FIXME Really bad to splat here
			a = cat(v..., dims = N) # N-dimensional array
		else
			error("Since the sizes of the arrays do not match along dimension N (the final dimension), we cannot stack the arrays along the (N + 1)th dimension; please set merge = true")
		end
	end

	return a
end
