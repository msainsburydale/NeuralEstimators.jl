


# Parameters {#Parameters}

Sampled parameters (e.g., from the prior distribution) are typically stored as a $d \times K$ matrix (possibly with named rows; see [`NamedMatrix`](/API/parametersdata#NamedArrays.NamedMatrix)), where $d$ is the dimension of the parameter vector of interest and $K$ is the number of sampled parameter vectors. However, any batchable object (compatible with [`numobs`](https://juliaml.github.io/MLUtils.jl/dev/api/#MLCore.numobs)/[`getobs`](https://juliaml.github.io/MLUtils.jl/dev/api/#MLCore.getobs)) is supported.

It can sometimes be helpful to wrap the parameters in a user-defined type that also stores expensive intermediate objects needed for simulating data (e.g., Cholesky factors). The user-defined type should be a subtype of [`AbstractParameterSet`](/API/parametersdata#NeuralEstimators.AbstractParameterSet), whose only requirement is a field `θ` that stores parameters.
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.AbstractParameterSet' href='#NeuralEstimators.AbstractParameterSet'><span class="jlbinding">NeuralEstimators.AbstractParameterSet</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractParameterSet
```


An abstract supertype for user-defined types that store parameters and any auxiliary objects needed for data simulation.

The user-defined type must have a field `θ` that stores the parameters. Typically,  `θ` is a $d$ × $K$ matrix, where $d$ is the dimension of the parameter vector and $K$ is the number of sampled parameter vectors, though any batchable object compatible with `numobs`/`getobs` is supported. There are no other requirements.

The number of parameter instances can be retrieved with `numobs`, and the size of `θ` can be inspected with `size`. 

Subtypes of `AbstractParameterSet` support indexing via `Base.getindex`,  with any batchable fields subsetted accordingly and all other fields left unchanged. To modify this default behaviour, provide a specific `Base.getindex` method for your concrete subtype.

**Examples**

```julia
struct Parameters <: AbstractParameterSet
	θ
	# auxiliary objects needed for data simulation
end

θ = randn(2, 100)
parameters = Parameters(θ)
numobs(parameters)   # 100
size(parameters)     # (2, 100)
parameters[1:10]     # subset of 10 parameter vectors
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Parameters.jl#L1-L30" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NamedArrays.NamedMatrix' href='#NamedArrays.NamedMatrix'><span class="jlbinding">NamedArrays.NamedMatrix</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
NamedMatrix(; kwargs...)
```


Construct a named matrix where each keyword argument defines a named row.

**Examples**

```julia
NamedMatrix(μ = randn(3), σ = rand(3))
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Parameters.jl#L63-L72" target="_blank" rel="noreferrer">source</a></Badge>

</details>


# Data {#Data}

Simulated data sets are stored as mini-batches in a format amenable to the chosen neural-network architecture; the only requirement is compatibility with [`numobs`](https://juliaml.github.io/MLUtils.jl/dev/api/#MLCore.numobs)/[`getobs`](https://juliaml.github.io/MLUtils.jl/dev/api/#MLCore.getobs). For example, when constructing an estimator from data collected over a two-dimensional grid, one may use a CNN, with each data set stored in the final dimension of a four-dimensional array.

Precomputed (expert) summary statistics can be incorporated by wrapping the simulated data in a [`DataSet`](/API/parametersdata#NeuralEstimators.DataSet) object, which couples the raw data with a matrix of summary statistics.
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.DataSet' href='#NeuralEstimators.DataSet'><span class="jlbinding">NeuralEstimators.DataSet</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DataSet(Z, S)
DataSet(Z)
```


A container that couples raw data `Z` with precomputed expert summary statistics `S` (a matrix with one column per data set). Passing a `DataSet` to any neural estimator causes the summary network to be applied to `Z`, with the resulting learned summary statistics concatenated with `S` before being passed to the inference network:

$$\boldsymbol{t}(\mathbf{Z}) = (\text{summary\_network}(\mathbf{Z})', \mathbf{S})',$$

Since `S` is precomputed and stored as a plain matrix, no special treatment is needed during training: gradients do not flow through `S`.

If `S` is not provided, `DataSet(Z)` is equivalent to passing `Z` directly.

See also [`summarystatistics`](/API/estimators#NeuralEstimators.summarystatistics).

**Examples**

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



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/DataSet.jl#L1-L35" target="_blank" rel="noreferrer">source</a></Badge>

</details>

