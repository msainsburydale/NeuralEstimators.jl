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