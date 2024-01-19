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
