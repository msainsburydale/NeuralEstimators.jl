"""
	DataSet(Z, S)
	DataSet(Z)
A container that couples raw data `Z` with precomputed expert summary statistics
`S` (a matrix with one column per data set). Passing a `DataSet` to any neural
estimator causes the summary network to be applied to `Z`, with the resulting
learned summary statistics concatenated with `S` before being passed to the
inference network:
```math
\\boldsymbol{t}(\\mathbf{Z}) = (\\text{summary\\_network}(\\mathbf{Z})', \\mathbf{S})',
```
Since `S` is precomputed and stored as a plain matrix, no special treatment is
needed during training: gradients do not flow through `S`.

If `S` is not provided, `DataSet(Z)` is equivalent to passing `Z` directly.

See also [`summarystatistics`](@ref).

# Examples
```julia
using NeuralEstimators
using Statistics: mean, var

# Simulate data: Z|μ,σ ~ N(μ, σ²)
n, m, K = 1, 50, 500
θ = rand(2, K)
Z = [θ[1, k] .+ θ[2, k] .* randn(n, m) for k in 1:K]

# Precompute expert summary statistics (e.g., sample mean and variance)
S = hcat([vcat(mean(z), var(z)) for z in Z]...)

# Package into a DataSet object
datasets = DataSet(Z, S)
```
"""
struct DataSet{Z, S}
    Z::Z
    S::S
    function DataSet(Z, S)
        @assert numobs(Z) == size(S, 2) "The number of data sets in Z ($(numobs(Z))) must match the number of columns in S ($(size(S, 2)))"
        new{typeof(Z), typeof(S)}(Z, S)
    end
    DataSet(Z) = new{typeof(Z), Nothing}(Z, nothing)
end

# Methods
numobs(d::DataSet) = numobs(d.Z)
Base.getindex(d::DataSet, i::Integer) = DataSet(getobs(d.Z, i:i), d.S[:, i:i])
Base.getindex(d::DataSet, i) = DataSet(getobs(d.Z, i), d.S[:, i])

numberreplicates(d::DataSet) = numberreplicates(d.Z)
subsetreplicates(d::DataSet, idx) = DataSet(subsetreplicates(d.Z, idx), d.S)
