#TODO document these functions (and add to online docs) and add unit testing

nparams(model) = length(Flux.params(model)) > 0 ? sum(length, Flux.params(model)) : 0

# Drop fields from NamedTuple: https://discourse.julialang.org/t/filtering-keys-out-of-named-tuples/73564/8
drop(nt::NamedTuple, key::Symbol) =  Base.structdiff(nt, NamedTuple{(key,)})
drop(nt::NamedTuple, keys::NTuple{N,Symbol}) where {N} = Base.structdiff(nt, NamedTuple{keys})

# Check element type of arbitrarily nested array: https://stackoverflow.com/a/41847530
nested_eltype(x) = nested_eltype(typeof(x))
nested_eltype(::Type{T}) where T <:AbstractArray = nested_eltype(eltype(T))
nested_eltype(::Type{T}) where T = T


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
materialised) since `Traingular` matrices are not always compatible with other
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
	numberofreplicates(Z)

Generic function that returns the number of replicates in a given object.
Default implementations are provided for commonly used data formats, namely,
data stored as an `Array` or as a `GNNGraph`.
"""
function numberreplicates end

function numberreplicates(Z::A) where {A <: AbstractArray{T, N}} where {T, N}
	size(Z, N)
end

function numberreplicates(Z::G) where {G <: GNNGraph}
	x = Z.ndata.x
	if ndims(x) == 3
		size(x, 2)
	else
		Z.num_graphs
	end
end

function numberreplicates(Z::V) where {V <: AbstractVector{A}} where A
	numberreplicates.(Z)
end

function numberreplicates(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B}
	Z = tup[1]
	X = tup[2]
	@assert length(Z) == length(X)

	numberreplicates.(Z)
end


"""
Generic function for subsetting replicates from a data set. Default methods are:

	subsetdata(Z::A, m) where {A <: AbstractArray{T, N}} where {T, N}
	subsetdata(Z::G, m) where {G <: AbstractGraph}

Note that `subsetdata` is slow for graphical data, and one should consider using
a method of `train` that does not require the data to be subsetted when working
with graphical data: use `numberreplicates` to check that the training and
validation data sets are equally replicated, which prevents the invocation of
`subsetdata`. Note also that `subsetdata` only applies to vectors of batched graphs.

If the user is working with data that is not covered by the default methods,
simply overload `subsetdata` with the appropriate type for `Z`.

# Examples
```
using NeuralEstimators
using GraphNeuralNetworks
using Flux: batch

n = 5  # number of observations in each realisation
m = 6  # number of replicates for each parameter vector
d = 1  # dimension of the response variable
K = 2  # number of parameter vectors

# Array data
Z = [rand(n, d, m) for k ∈ 1:K]
subsetdata(Z, 1:3) # extract first 3 replicates for each parameter vector

# Graphical data
e = 8 # number of edges
Z = [batch([rand_graph(n, e, ndata = rand(d, n)) for _ ∈ 1:m]) for k ∈ 1:K]
subsetdata(Z, 1:3) # extract first 3 replicates for each parameter vector
```
"""
function subsetdata end

function subsetdata(Z::V, m) where {V <: AbstractVector{A}} where A
	subsetdata.(Z, Ref(m))
end

function subsetdata(tup::Tup, m) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B}
	Z = tup[1]
	X = tup[2]
	@assert length(Z) == length(X)

	(subsetdata(Z, m), X) # NB X is not subsetted because it is set-level information
end

function subsetdata(Z::A, m) where {A <: AbstractArray{T, N}} where {T, N}
	colons  = ntuple(_ -> (:), N - 1)
	Z[colons..., m]
end

function subsetdata(Z::G, m) where {G <: AbstractGraph}
	 # @warn "`subsetdata()` is slow for graphical data: consider using a method of `train` that does not require the data data to be subsetted. Use `numberreplicates()` to check that the training and validation data sets are equally replicated, which prevents the invocation of `subsetdata()`."
	 getgraph(Z, m)
end



# Note that these arguments differ to DataLoader(), so this function should never be exported.
function _quietDataLoader(data, batchsize::Integer; shuffle = true, partial = false)
	oldstd = stdout
	redirect_stderr(open("/dev/null", "w"))
	data_loader = DataLoader(data, batchsize = batchsize, shuffle = shuffle, partial = partial)
	redirect_stderr(oldstd)
	return data_loader
end

function _checkgpu(use_gpu::Bool; verbose::Bool = true)

	if use_gpu && CUDA.functional()
		if verbose @info "Running on CUDA GPU" end
		CUDA.allowscalar(false)
		device = gpu
	else
		if verbose @info "Running on CPU" end
		device = cpu
	end

	return(device)
end

# ---- Functions for finding, saving, and loading the best neural network ----

common_docstring = """
Given a `path` to a training run containing neural networks saved with names
`"network_epochx.bson"` and an object saved as `"loss_per_epoch.bson"`,
"""

"""
	_findbestweights(path::String)

$common_docstring finds the epoch of the best network (measured by validation loss).
"""
function _findbestweights(path::String)
	loss_per_epoch = load(joinpath(path, "loss_per_epoch.bson"), @__MODULE__)[:loss_per_epoch]

	# The first row is the risk evaluated for the initial neural network, that
	# is, the network at epoch 0. Since Julia starts indexing from 1, we hence
	# subtract 1 from argmin().
	best_epoch = argmin(loss_per_epoch[:, 2]) -1

	return best_epoch
end


"""
	_savebestweights(path::String)

$common_docstring saves the weights of the best network (measured by validation loss) as 'best_network.bson'.
"""
function _savebestweights(path::String)
	best_epoch = _findbestweights(path)
	load_path   = joinpath(path, "network_epoch$(best_epoch).bson")
	save_path   = joinpath(path, "best_network.bson")
	cp(load_path, save_path, force = true)
	return nothing
end

"""
	loadbestweights(path::String)

$common_docstring returns the weights of the best network (measured by validation loss).
"""
function loadbestweights(path::String)
	load_path     = joinpath(path, "best_network.bson")
	best_weights = load(load_path, @__MODULE__)[:weights]
	return best_weights
end


"""
	estimateinbatches(θ̂, z; batchsize::Integer = 32, use_gpu::Bool = true)

Apply the estimator `θ̂` on minibatches of `z` of size `batchsize`, to avoid
memory issues that can occur when `z` is very large.

Minibatching will only be done if there are multiple data sets in `z`; this
will be inferred by `z` being a vector, or a tuple whose first element is a vector.
"""
function estimateinbatches(θ̂, z; batchsize::Integer = 32, use_gpu::Bool = true)

	# Only do minibatching if we have multiple data sets
	if typeof(z) <: AbstractVector
		minibatching = true
		batchsize = min(length(z), batchsize)
	elseif typeof(z) <: Tuple && typeof(z[1]) <: AbstractVector
		minibatching = true
		batchsize = min(length(z[1]), batchsize)
	else # we dont have replicates: just apply the estimator without minibatching
		minibatching = false
	end

	device  = _checkgpu(use_gpu, verbose = false)
	θ̂ = θ̂ |> device

	if !minibatching
		z = z |> device
		ŷ = θ̂(z)
		ŷ = ŷ |> cpu
	else
		data_loader = _quietDataLoader(z, batchsize, shuffle=false, partial=true)
		ŷ = map(data_loader) do zᵢ
			zᵢ = zᵢ |> device
			ŷ = θ̂(zᵢ)
			ŷ = ŷ |> cpu
			ŷ
		end
		ŷ = stackarrays(ŷ)
	end

	return ŷ
end
# Backwards compatability:
_runondevice(θ̂, z, use_gpu::Bool; batchsize::Integer = 32) = estimateinbatches(θ̂, z; batchsize = batchsize, use_gpu = use_gpu)


"""
    expandgrid(xs, ys)

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

Stack a vector of arrays `v` along the last dimension of each array, optionally merging the final dimension of the stacked array.

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
