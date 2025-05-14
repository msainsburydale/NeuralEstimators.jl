"""
	samplesize(Z)
Computes the number of independent replicates in the data set `Z`. 

Note that this function is a wrapper around [`numberreplicates`](@ref) with return type equal to the eltype of `Z`.
"""
samplesize(Z) = eltype(Z)(numberreplicates(Z))

"""
	logsamplesize(Z)
Computes the log of the number of independent replicates in the data set `Z`. 
"""
logsamplesize(Z) = log.(samplesize(Z))

"""
	invsqrtsamplesize(Z)
Computes the inverse of the square root of the number of independent replicates in the data set `Z`. 
"""
invsqrtsamplesize(Z) = 1 ./ (sqrt.(samplesize(Z)))

"""
	samplecovariance(Z::AbstractArray)

Computes the [sample covariance matrix](https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Definition_of_sample_covariance),
Σ̂, and returns the vectorised lower triangle of Σ̂.

# Examples
```
# 5 independent replicates of a 3-dimensional vector
z = rand(3, 5)
samplecovariance(z)
```
"""
function samplecovariance(z::A) where {A <: AbstractArray{T, N}} where {T, N}
    @assert size(z, N) > 1 "The number of replicates, which are stored in the final dimension of the input array, should be greater than 1"
    z = Flux.flatten(z) # convert to matrix (allows for arbitrary sized data inputs)
    d = size(z, 1)
    Σ̂ = cov(z, dims = 2, corrected = false)
    tril_idx = tril(trues(d, d))
    return Σ̂[tril_idx]
end
samplecovariance(z::AbstractVector) = samplecovariance(reshape(z, :, 1))

"""
	samplecorrelation(Z::AbstractArray)

Computes the sample correlation matrix,
R̂, and returns the vectorised strict lower triangle of R̂.

# Examples
```
# 5 independent replicates of a 3-dimensional vector
z = rand(3, 5)
samplecorrelation(z)
```
"""
function samplecorrelation(z::A) where {A <: AbstractArray{T, N}} where {T, N}
    @assert size(z, N) > 1 "The number of replicates, which are stored in the final dimension of the input array, should be greater than 1"
    z = Flux.flatten(z) # convert to matrix (allows for arbitrary sized data inputs)
    d = size(z, 1)
    Σ̂ = cor(z, dims = 2)
    tril_idx = tril(trues(d, d), -1)
    return Σ̂[tril_idx]
end
samplecorrelation(z::AbstractVector) = samplecorrelation(reshape(z, :, 1))

# NB I thought the following functions might be better on the GPU, but after
# some benchmarking it turns out the base implementation is better (at least
# when considering only a single data set at a time). Still, I will leave these
# functions here in case I want to implement something similar later.

# function samplecov(z::A) where {A <: AbstractArray{T, N}} where {T, N}
# 	@assert size(z, N) > 1 "The number of replicates, which are stored in the final dimension of the input array, should be greater than 1"
# 	z = Flux.flatten(z) # convert to matrix (allows for arbitrary sized data inputs)
# 	d, n = size(z)
# 	z̄ = mean(z, dims = 2)
# 	e = z .- z̄
# 	e = reshape(e, (size(e, 1), 1, n)) # 3D array for batched mul and transpose
# 	Σ̂ = sum(e ⊠ batched_transpose(e), dims = 3) / T(n)
# 	Σ̂ = reshape(Σ̂, d, d) # convert matrix (drop final singelton)
# 	tril_idx = tril(trues(d, d))
# 	return Σ̂[tril_idx]
# end
#
# function samplecor(z::A) where {A <: AbstractArray{T, N}} where {T, N}
# 	@assert size(z, N) > 1 "The number of replicates, which are stored in the final dimension of the input array, should be greater than 1"
# 	z = Flux.flatten(z) # convert to matrix (allows for arbitrary sized data inputs)
# 	d, n = size(z)
# 	z̄ = mean(z, dims = 2)
# 	e = z .- z̄
# 	e = reshape(e, (size(e, 1), 1, n)) # 3D array for batched mul and transpose
# 	Σ̂ = sum(e ⊠ batched_transpose(e), dims = 3) / T(n)
# 	Σ̂ = reshape(Σ̂, d, d) # convert matrix (drop final singelton)
# 	σ̂ = Σ̂[diagind(Σ̂)]
# 	D = Diagonal(1 ./ sqrt.(σ̂))
# 	Σ̂ = D * Σ̂ * D
# 	tril_idx = tril(trues(d, d), -1)
# 	return Σ̂[tril_idx]
# end
#
# using NeuralEstimators
# using Flux
# using BenchmarkTools
# using Statistics
# using LinearAlgebra
# z = rand(3, 4000) |> gpu
# @time samplecovariance(z)
# @time samplecov(z)
# @time samplecorrelation(z)
# @time samplecor(z)
#
# @btime samplecovariance(z);
# @btime samplecov(z);
# @btime samplecorrelation(z);
# @btime samplecor(z);

#TODO clean up this documentation (e.g., don't bother with the bin notation)
#TODO there is a more general structure that we could define, that has message(xi, xj, e) as a slot
@doc raw"""
	NeighbourhoodVariogram(h_max, n_bins) 
	(l::NeighbourhoodVariogram)(g::GNNGraph)

Computes the empirical variogram, 

```math
\hat{\gamma}(h \pm \delta) = \frac{1}{2|N(h \pm \delta)|} \sum_{(i,j) \in N(h \pm \delta)} (Z_i - Z_j)^2
```

where $N(h \pm \delta) \equiv \left\{(i,j) : \|\boldsymbol{s}_i - \boldsymbol{s}_j\| \in (h-\delta, h+\delta)\right\}$ 
is the set of pairs of locations separated by a distance within $(h-\delta, h+\delta)$, and $|\cdot|$ denotes set cardinality. 

The distance bins are constructed to have constant width $2\delta$, chosen based on the maximum distance 
`h_max` to be considered, and the specified number of bins `n_bins`. 

The input type is a `GNNGraph`, and the empirical variogram is computed based on the corresponding graph structure. 
Specifically, only locations that are considered neighbours will be used when computing the empirical variogram. 

# Examples 
```
using NeuralEstimators, Distances, LinearAlgebra
  
# Simulate Gaussian spatial data with exponential covariance function 
θ = 0.1                                 # true range parameter 
n = 250                                 # number of spatial locations 
S = rand(n, 2)                          # spatial locations 
D = pairwise(Euclidean(), S, dims = 1)  # distance matrix 
Σ = exp.(-D ./ θ)                       # covariance matrix 
L = cholesky(Symmetric(Σ)).L            # Cholesky factor 
m = 5                                   # number of independent replicates 
Z = L * randn(n, m)                     # simulated data 

# Construct the spatial graph 
r = 0.15                                # radius of neighbourhood set
g = spatialgraph(S, Z, r = r)

# Construct the variogram object with 10 bins
nv = NeighbourhoodVariogram(r, 10) 

# Compute the empirical variogram 
nv(g)
```
"""
struct NeighbourhoodVariogram{T} <: GNNLayer
    h_cutoffs::T
    # TODO inner constructor, add 0 into h_cutoffs if it is not already in there 
end
function NeighbourhoodVariogram(h_max, n_bins::Integer)
    h_cutoffs = range(0, stop = h_max, length = n_bins+1)
    h_cutoffs = collect(h_cutoffs)
    NeighbourhoodVariogram(h_cutoffs)
end
function (l::NeighbourhoodVariogram)(g::GNNGraph)

    # NB in the case of a batched graph, see the comments in the method summarystatistics(d::DeepSet, Z::V) where {V <: AbstractVector{G}} where {G <: GNNGraph}
    Z = g.ndata.Z
    h = g.graph[3]

    message(xi, xj, e) = (xi - xj) .^ 2
    z = apply_edges(message, g, Z, Z, h) # (Zⱼ - Zᵢ)², possibly replicated 
    z = mean(z, dims = 2) # average over the replicates 
    z = vec(z)

    # Bin the distances
    h_cutoffs = l.h_cutoffs
    bins_upper = h_cutoffs[2:end]   # upper bounds of the distance bins
    bins_lower = h_cutoffs[1:(end - 1)] # lower bounds of the distance bins 
    N = [bins_lower[i:i] .< h .<= bins_upper[i:i] for i in eachindex(bins_upper)] # NB avoid scalar indexing by i:i
    N = reduce(hcat, N)

    # Compute the average over each bin
    N_card = sum(N, dims = 1)        # number of occurences in each distance bin 
    N_card = N_card + (N_card .== 0) # prevent division by zero 
    Σ = sum(z .* N, dims = 1)        # ∑(Zⱼ - Zᵢ)² in each bin
    vec(Σ ./ 2N_card)
end
Flux.trainable(l::NeighbourhoodVariogram) = NamedTuple()
