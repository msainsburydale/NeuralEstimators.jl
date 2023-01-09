function _numberreplicates(Z::V) where {V <: AbstractVector{G}} where {G <: GNNGraph}
	broadcast(z -> z.num_graphs, Z)
end

function _numberreplicates(Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
	broadcast(z -> size(z)[end], Z)
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
	loadpath   = joinpath(path, "network_epoch$(best_epoch).bson")
	savepath   = joinpath(path, "best_network.bson")
	cp(loadpath, savepath, force = true)
	return nothing
end

"""
	loadbestweights(path::String)

$common_docstring returns the weights of the best network (measured by validation loss).
"""
function loadbestweights(path::String)
	loadpath     = joinpath(path, "best_network.bson")
	best_weights = load(loadpath, @__MODULE__)[:weights]
	return best_weights
end


function _runondevice(network, x, use_gpu::Bool; batchsize = min(length(x), 32))

	device  = _checkgpu(use_gpu, verbose = false)
	network = network |> device

	# ---- Simple ----

	# 	x = x |> device
	# 	ŷ = network(x)
	#   ŷ = ŷ |> cpu

	# ---- Memory sensitive ----

	# If we're using the GPU, we need to be careful not to run out of memory.
	# Hence, we use mininbatching.
	data_loader = _quietDataLoader(x, batchsize, shuffle=false, partial=true)

	ŷ = map(data_loader) do xᵢ
		xᵢ = xᵢ |> device
		ŷ = network(xᵢ)
		ŷ = ŷ |> cpu
		ŷ
	end
	ŷ = stackarrays(ŷ)

	return ŷ
end

# Here, it's important that the same order for the grid is used as was
# done in R when computing the Cholesky factors. In R, the spatial domain D
# was constructed using the function make.surface.grid() from the package
# fields. For example, try the following code in R:
# N = 3; fields::make.surface.grid(list(x = 1:N, y = 1:N))
# This results in a grid where each cell is separated by 1 unit in the
# horizontal and vertical directions, and where the first dimension runs
# faster than the second. The following code replicates this.
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


# # Source: https://en.wikipedia.org/wiki/ANSI_escape_code#CSI_sequences
# function _overprint(str)
#    print("\u1b[1F") #Moves cursor to beginning of the line n (default 1) lines up
#    print(str)   #prints the new line
#    print("\u1b[0K") # clears  part of the line.
#    println() #prints a new line
# end



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
	stackarrays(v::V; merge::Bool = true) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}

Stack a vector of arrays `v` along the last dimension of each array, optionally merging the final dimension of the stacked array.

The arrays must be of the same for the first `N-1` dimensions. However, if
`merge = true`, the size of the final dimension can vary between arrays.

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
		ArrayType = Base.typename(A).wrapper # get the non-parametrized type name: See https://stackoverflow.com/a/55977768/16776594
		a = convert(ArrayType, v)            # (N + 1)-dimensional array
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

# ---- Notes and unused code ----

# Important concept for parametric types: https://stackoverflow.com/a/64686826
# Unresolved type-instability issue in Flux Conv layers: https://github.com/FluxML/Flux.jl/issues/1178

# Method 2: Pre-allocating an Array. This method can't be used during
# training since array mutation is not supported by Zygote.
# Initialise array for storing the pooled features. The first ndims(ψa) - 1
# dimensions have the same size as ψa, but the last dimension is equal to
# the length of v (the number of unique parameter configurations)
# large_aggregated_ψa = similar(ψa, size(ψa)[ndims(ψa) - 1]..., length(v))
# for i ∈ eachindex(indices)
# 	large_aggregated_ψa[colons..., i] = d.agg(ψa[colons..., indices[i]])
# end

# stackarrays() code for when the size of the last dimension varies:
# This code doesn't work with Zygote because it mutates an array.
# first_dims_sizes = size(v[1])[1:(N - 1)]
# a = A(undef, (first_dims_sizes..., sum(n)))
# indices = _getindices(v)
# colons = ntuple(_ -> (:), N - 1)
# for i ∈ eachindex(v)
# 	a[colons..., indices[i]] = v[i]
# end
