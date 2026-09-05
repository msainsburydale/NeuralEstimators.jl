@doc raw"""
    variogram(Z::AbstractVector, D::AbstractMatrix; n_bins = 20, maxlag = 0.5)
    variogram(Z::AbstractMatrix; n_bins = 20, maxlag = 0.5)
    variogram(Z::AbstractArray{<:Real,3}; n_bins = 20, maxlag = 0.5)

Empirical isotropic variogram,

```math
\hat{\gamma}(h) = \frac{1}{2|N(h)|} \sum_{(i,j) \in N(h)} (Z_i - Z_j)^2
```

where ``N(h)`` is the set of pairs whose distance falls in the bin containing ``h``.
Distances are partitioned into `n_bins` equal-width bins. Empty bins are `NaN`.

`maxlag` is a relative cutoff in ``(0, 1]``: the fraction of the maximum
distance that is included (default `0.5`). Bins span ``[0, \mathrm{maxlag} \cdot d_{\max}]``,
and pairs farther than that are dropped. For the pairwise method, ``d_{\max}``
is ``\max D``; for the FFT methods, ``d_{\max}`` is the grid diagonal in grid steps.

The pairwise method takes observations `Z` at `n` locations and an `n×n`
distance matrix `D`. The diagonal of `D` is excluded.

The matrix and 3-dimensional-array methods compute the same estimator on a
complete regular grid via FFT ([Marcotte, 1996](https://www.sciencedirect.com/science/article/pii/S009830049600026X)),
and require FFTW.jl (`using FFTW`). Distances are in grid steps, so no coordinate or distance
matrix is required. A matrix is one field and returns a vector. A
3-dimensional array is a stack of fields and returns a matrix with
`size(Z, 3)` columns.

See the [GeoStats.jl variogram documentation](https://juliaearth.github.io/GeoStatsDocs/stable/variograms/#Variograms).

# Examples
```julia
using NeuralEstimators

# Pairwise method (irregular locations)
using Distances, LinearAlgebra
n = 300
S = rand(n, 2)
D = pairwise(Euclidean(), S, dims = 1)
Σ = exp.(-D ./ 0.1)       
L = cholesky(Symmetric(Σ)).L
Z = L * randn(n)
variogram(Z, D)

# FFT method (complete regular grid; requires `using FFTW`)
using FFTW
using GaussianRandomFields
grid_dim = 64
pts = range(0, 1, grid_dim)
cov = CovarianceFunction(2, Exponential(0.1))
grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts; minpadding = grid_dim)
# one field
Z = sample(grf)                
variogram(Z)
# stack of fields
Z = [sample(grf) for _ in 1:5] |> stack
variogram(Z)
```
"""
function variogram(z::AbstractVector, D::AbstractMatrix; n_bins = 20, maxlag = 0.5)
    (0 < maxlag <= 1) || throw(ArgumentError("maxlag must be in (0, 1]"))
    n = length(z)
    hmax = maxlag * maximum(D)
    edges  = range(0, hmax; length = n_bins + 1)
    sums   = zeros(n_bins)
    counts = zeros(Int, n_bins)
    @inbounds for j in 1:n, i in (j+1):n
        d = D[i, j]
        d > hmax && continue
        bin = min(searchsortedlast(edges, d), n_bins)
        sums[bin]   += (z[i] - z[j])^2
        counts[bin] += 1
    end
    sums ./ (2 .* counts)
end
# NB FFT methods are defined in ext/NeuralEstimatorsFFTWExt.jl

"""
	samplesize(Z)
Computes the number of replicates in the data set `Z`. 

Note that this function is a wrapper around [`numberreplicates`](@ref) with return type equal to the eltype of `Z`.
"""
samplesize(Z) = eltype(Z)(numberreplicates(Z))

"""
	logsamplesize(Z)
Computes the log of the number of replicates in the data set `Z`. 
"""
logsamplesize(Z) = log.(samplesize(Z))

"""
	invsqrtsamplesize(Z)
Computes the inverse of the square root of the number of replicates in the data set `Z`. 
"""
invsqrtsamplesize(Z) = 1 ./ (sqrt.(samplesize(Z)))

"""
	samplecovariance(Z::AbstractArray)

Computes the [sample covariance matrix](https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Definition_of_sample_covariance),
Σ̂, and returns the vectorised lower triangle of Σ̂.

# Examples
```julia
# 5 independent replicates of a 3-dimensional vector
z = rand(3, 5)
samplecovariance(z)
```
"""
function samplecovariance(z::A) where {A <: AbstractArray{T, N}} where {T, N}
    @assert size(z, N) > 1 "The number of replicates, which are stored in the final dimension of the input array, should be greater than 1"
    z = flatten(z) # convert to matrix (allows for arbitrary sized data inputs)
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
```julia
# 5 independent replicates of a 3-dimensional vector
z = rand(3, 5)
samplecorrelation(z)
```
"""
function samplecorrelation(z::A) where {A <: AbstractArray{T, N}} where {T, N}
    @assert size(z, N) > 1 "The number of replicates, which are stored in the final dimension of the input array, should be greater than 1"
    z = flatten(z) # convert to matrix (allows for arbitrary sized data inputs)
    d = size(z, 1)
    Σ̂ = cor(z, dims = 2)
    tril_idx = tril(trues(d, d), -1)
    return Σ̂[tril_idx]
end
samplecorrelation(z::AbstractVector) = samplecorrelation(reshape(z, :, 1))