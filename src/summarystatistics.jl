#TODO samplemean, samplequantile (this will have to be marginal quantiles), measures of multivariate skewness and kurtosis (https://www.jstor.org/stable/2334770). See what Gerber did.

"""
	samplesize(Z::AbstractArray)

Computes the sample size of a set of independent realisations `Z`.

Note that this function is a wrapper around [`numberreplicates`](@ref), but this
function returns the number of replicates as the eltype of `Z`, rather than as an integer.
"""
samplesize(Z) = eltype(Z)(numberreplicates(Z))

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



## Summary statistics for spatial point processes 
#TODO unit testing, document and export
#TODO Don't use S for the expert summary statistics... clashes with the spatial locations 
"""
	DistanceQuantiles(probs) <: GNNLayer

# Examples
```
using NeuralEstimators, Flux, GraphNeuralNetworks
using Statistics: mean

# Generate some toy spatial data
m = 5            # number of replicates
d = 2            # spatial dimension
n = 100          # number of spatial locations
S = rand(n, d)   # spatial locations
Z = rand(n, m)   # toy data
g = spatialgraph(S, Z)

# Propagation and readout modules forming the neural-summary network
propagation = GNNChain(
	SpatialGraphConv(1 => 16), 
	SpatialGraphConv(16 => 32)
	)
readout = GlobalPool(mean)
ψ = GNNSummary(propagation, readout)

# Expert summary statistics 
probs = collect(0.1:0.2:0.9)
S = DistanceQuantiles(probs)
S(g) # can apply directly to spatial graph

# Inference network and DeepSeet object
p = 3 # number of parameters in the statistical model
ϕ = Chain(Dense(32 + length(probs), 64, relu), Dense(64, p))
θ̂ = DeepSet(ψ, ϕ; S = S)

# Apply full estimator to spatial graph 
θ̂(g)

# Batched graph 
G = Flux.batch([g, g])
@test S(G) == S(g)
```
"""
struct DistanceQuantiles{T} <: GNNLayer
    probs::T
	#TODO assert that all probs are greater than 0 and less than 1, or figure out how to accomodate the zero case 
	#TODO assert that probs are ascending
	#TODO convert to Float32 and ensure probs is a vector
end
@layer DistanceQuantiles
function (l::DistanceQuantiles)(g::GNNGraph)
	
	# NB in the case of a batched graph, see the comments in the method summarystatistics(d::DeepSet, Z::V) where {V <: AbstractVector{G}} where {G <: GNNGraph}

	h = g.graph[3] # spatial distances 

	# Don't wish to include zero distances... there will be exactly num_nodes 
	# zero values corresponding to the diagonal of the adjacency matrix. 
	# Since we need to sort the distances anyway to compute the quantiles, 
	# we can exploit this as follows:
	h_sorted = sort(h)[Not(1:g.num_nodes)]

	# quantile() is not implemented yet on the GPU (see https://github.com/JuliaGPU/CUDA.jl/issues/265)
	# instead, since sort() does work well on the GPU, we will use that (the functionality that I need is 
	# simpler, and easier to implement, than that provided in the general function quantile()).
	# quantile(h, l.probs) 
	n = length(h_sorted)
	idx = ceil.(Int64, probs .* n)
	h_sorted[idx]
end

#TODO clean up this documentation (e.g., don't bother with the bin notation)
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

# Construct the variogram object wth 10 bins
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
	h_cutoffs = range(0, stop= h_max, length = n_bins+1)
	h_cutoffs = collect(h_cutoffs)
	NeighbourhoodVariogram(h_cutoffs)
end
function (l::NeighbourhoodVariogram)(g::GNNGraph)
	
	# NB in the case of a batched graph, see the comments in the method summarystatistics(d::DeepSet, Z::V) where {V <: AbstractVector{G}} where {G <: GNNGraph}

	# Note that we do not need to remove self-loops, since we define the bins to be greater than 0
	Z = g.ndata.Z
	h = g.graph[3]

	message(xi, xj, e) = (xi - xj).^2
	z = apply_edges(message, g, Z, Z, h) # (Zⱼ - Zᵢ)², possibly replicated 
	z = mean(z, dims = 2) # average over the replicates 
	z = vec(z)

	# Bin the distances
	h_cutoffs = l.h_cutoffs
	bins_upper = h_cutoffs[2:end]   # upper bounds of the distance bins
	bins_lower = h_cutoffs[1:end-1] # lower bounds of the distance bins 
	N = [bins_lower[i:i] .< h .<= bins_upper[i:i] for i in eachindex(bins_upper)] # NB avoid scalar indexing by i:i
	N = reduce(hcat, N)

	# Compute the average over each bin
	N_card = sum(N, dims = 1)        # number of occurences in each distance bin 
	N_card = N_card + (N_card .== 0) # prevent division by zero 
	Σ = sum(z .* N, dims = 1)        # ∑(Zⱼ - Zᵢ)² in each bin
	vec(Σ ./ 2N_card)
end
@layer NeighbourhoodVariogram
Flux.trainable(l::NeighbourhoodVariogram) =  ()

#TODO there is a more general structure that we could define, that has message(xi, xj, e) as a slot



"""
	ripleyk(points, radius_range)	
Calculates Ripley's K function for each radius in `radius_range`. 

# Examples 
```
# Generate some random points
d = 2 # two-dimensional points 
points = [rand(d) for _ in 1:250]

# Define the range of radii
radius_range = 0.1:0.1:0.5

# Calculate Ripley's K function
K_values = ripleyk(points, radius_range)

# Illustrate with matern cluster process
radius_range = 0.01:0.01:0.2
n = 1000
λ = [10, 25, 50, 90]
μ = n ./ λ
S = [maternclusterprocess(λ = λ[i], μ = μ[i]) for i in eachindex(λ)]
ripleyk.(S, Ref(radius_range))
```
"""
function ripleyk(points, radius_range)
    n = length(points)
    K_values = zeros(length(radius_range))

    for (i, r) in enumerate(radius_range)
        count = 0
        for j in 1:n
            for k in 1:n
                if j != k && norm(points[j] - points[k]) <= r
                    count += 1
                end
            end
        end
        K_values[i] = count / (n * (n - 1))
    end

    return K_values
end
ripleyk(points::AbstractMatrix, radius_range) = ripleyk([points[i,:] for i in 1:size(S, 1)], radius_range) # TODO make efficient if we do this in the end 

