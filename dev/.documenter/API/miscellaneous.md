


# Miscellaneous {#Miscellaneous}

## Core {#Core}

These functions can appear during the core workflow, and may need to be overloaded in some applications.
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.numberreplicates' href='#NeuralEstimators.numberreplicates'><span class="jlbinding">NeuralEstimators.numberreplicates</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
numberreplicates(Z)
```


Generic function that returns the number of replicates in a given object. Default implementations are provided for commonly used data formats, namely, data stored as an `Array` or as a `GNNGraph`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/utility.jl#L225-L231" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.subsetreplicates' href='#NeuralEstimators.subsetreplicates'><span class="jlbinding">NeuralEstimators.subsetreplicates</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
subsetreplicates(Z::V, i) where {V <: AbstractArray{A}} where {A <: Any}
subsetreplicates(Z::A, i) where {A <: AbstractArray{T, N}} where {T, N}
subsetreplicates(Z::G, i) where {G <: GNNGraph}
```


Return replicate(s) `i` from each data set in `Z`.

If working with data that are not covered by the default methods, overload the function with the appropriate type for `Z`.

For graphical data, calls [`getgraph()`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/api/gnngraph/#GraphNeuralNetworks.GNNGraphs.getgraph-Tuple{GNNGraph,%20Int64}), where the replicates are assumed be to stored as batched graphs. Since this can be slow, one should consider using a method of [`train()`](/API/training#NeuralEstimators.train) that does not require the data to be subsetted when working with graphical data (use [`numberreplicates()`](/API/miscellaneous#NeuralEstimators.numberreplicates) to check that the training and validation data sets are equally replicated, which prevents subsetting).

**Examples**

```julia
using NeuralEstimators
using GraphNeuralNetworks
using Flux: batch

d = 1  # dimension of the response variable
n = 4  # number of observations in each realisation
m = 6  # number of replicates in each data set
K = 2  # number of data sets

# Array data
Z = [rand(n, d, m) for k ∈ 1:K]
subsetreplicates(Z, 2)   # extract second replicate from each data set
subsetreplicates(Z, 1:3) # extract first 3 replicates from each data set

# Graphical data
e = 8 # number of edges
Z = [batch([rand_graph(n, e, ndata = rand(d, n)) for _ ∈ 1:m]) for k ∈ 1:K]
subsetreplicates(Z, 2)   # extract second replicate from each data set
subsetreplicates(Z, 1:3) # extract first 3 replicates from each data set
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/utility.jl#L259-L297" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Downstream-inference algorithms {#Downstream-inference-algorithms}
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.EM' href='#NeuralEstimators.EM'><span class="jlbinding">NeuralEstimators.EM</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
EM(simulateconditional::Function, MAP::Union{Function, NeuralEstimator}, θ₀ = nothing)
```


Implements a (Bayesian) Monte Carlo expectation-maximization (EM) algorithm for  parameter estimation with missing data. The algorithm iteratively simulates missing  data conditional on current parameter estimates, then updates parameters using a  (neural) maximum a posteriori (MAP) estimator.

The $l$th iteration is given by:

$$\boldsymbol{\theta}^{(l)} =
\underset{\boldsymbol{\theta}}{\mathrm{arg\,max}}
\sum_{h = 1}^H \ell(\boldsymbol{\theta};  \boldsymbol{Z}_1,  \boldsymbol{Z}_2^{(l, h)}) + H\log \pi(\boldsymbol{\theta})$$

where $\ell((\boldsymbol{\theta};  \boldsymbol{Z}_1,  \boldsymbol{Z}_2)$ denotes the complete-data log-likelihood function,  $\boldsymbol{Z} \equiv (\boldsymbol{Z}_1', \boldsymbol{Z}_2')'$ denotes the complete  data with $\boldsymbol{Z}_1$ and $\boldsymbol{Z}_2$ the observed and missing  components respectively, $\boldsymbol{Z}_2^{(l, h)}$, $h = 1, \dots, H$, are simulated  from the distribution of $\boldsymbol{Z}_2 \mid \boldsymbol{Z}_1, \boldsymbol{\theta}^{(l-1)}$, and  $\pi(\boldsymbol{\theta})$ is the prior density (which can be viewed as a penalty function).

The algorithm monitors convergence by computing:

$$\max_i \left| \frac{\bar{\theta}_i^{(l+1)} - \bar{\theta}_i^{(l)}}{|\bar{\theta}_i^{(l)}| + \epsilon} \right|$$

where $\bar{\theta}^{(l)}$ is the average of parameter estimates from iteration  `burnin+1` to iteration $l$, and $\epsilon$ is machine precision. Convergence is  declared when this quantity is less than `tol` for `nconsecutive` consecutive iterations (see keyword arguments below).

**Fields**
- `simulateconditional::Function`: Function for simulating missing data conditional on  observed data and current parameter estimates. Must have signature:
  
  ```julia
  simulateconditional(Z::AbstractArray{Union{Missing, T}}, θ; nsims = 1, kwargs...)
  ```
  
  Returns completed data in the format expected by `MAP` (e.g., 4D array for CNNs).
  
- `MAP::NeuralEstimator`: MAP estimator applied to completed data. 
  
- `θ₀`: Optional initial parameter values (vector). Can also be provided when calling  the `EM` object.
  

**Methods**

Once constructed, objects of type `EM` can be applied to data via the following methods (corresponding to single or multiple data sets, respectively):

```
(em::EM)(Z::A, θ₀::Union{Nothing, Vector} = nothing; ...) where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}
(em::EM)(Z::V, θ₀::Union{Nothing, Vector, Matrix} = nothing; ...) where {V <: AbstractVector{A}} where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}
```


where `Z` is the complete data containing the observed data and `Missing` values.

For multiple datasets, `θ₀` can be a vector (same initial values for all) or a matrix  with one column per dataset.

**Keyword Arguments**
- `niterations::Integer = 50`: Maximum number of EM iterations.
  
- `nsims::Union{Integer, Vector{<:Integer}} = 1`: Number of conditional simulations per  iteration. Can be fixed (scalar) or varying (vector of length `niterations`).
  
- `burnin::Integer = 1`: Number of initial iterations to discard before averaging  parameter estimates for convergence assessment.
  
- `nconsecutive::Integer = 3`: Number of consecutive iterations meeting the convergence  criterion required to halt.
  
- `tol = 0.01`: Convergence tolerance. Algorithm stops if the relative change in  post-burnin averaged parameters is less than `tol` for `nconsecutive` iterations.
  
- `use_gpu::Bool = true`: Whether to use a GPU (if available) for MAP estimation.
  
- `verbose::Bool = false`: Whether to print iteration details.
  
- `kwargs...`: Additional arguments passed to `simulateconditional`.
  

**Returns**

For a single data set, returns a named tuple containing:
- `estimate`: Final parameter estimate (post-burnin average).
  
- `iterates`: Matrix of all parameter estimates across iterations (each column is one iteration).
  
- `burnin`: The burnin value used.
  

For multiple data set, returns a matrix with one column per dataset.

**Notes**
- If `Z` contains no missing values, the MAP estimator is applied directly (after  passing through `simulateconditional` to ensure correct format).
  
- When using a GPU, data are moved to the GPU for MAP estimation and back to the CPU  for conditional simulation.
  

**Examples**

```julia
# Below we give a pseudocode example; see the "Missing data" section in "Advanced usage" for a concrete example.

# Define conditional simulation function
function sim_conditional(Z, θ; nsims = 1)
    # Your implementation here
    # Returns completed data in format suitable for MAP estimator
end

# Define or load MAP estimator
MAP_estimator = ... # Neural MAP estimator

# Create EM object
em = EM(sim_conditional, MAP_estimator, θ₀ = [1.0, 2.0])

# Apply to data with missing values
Z = ... # Array with Missing entries
result = em(Z, niterations = 100, nsims = 5, tol = 0.001, verbose = true)

# Access results
θ_final = result.estimate
θ_sequence = result.iterates

# Multiple datasets
Z_list = [Z1, Z2, Z3]
estimates = em(Z_list, θ₀ = [1.0, 2.0])  # Matrix with 3 columns
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/missingdata.jl#L1-L120" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Utility functions {#Utility-functions}
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.adjacencymatrix' href='#NeuralEstimators.adjacencymatrix'><span class="jlbinding">NeuralEstimators.adjacencymatrix</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
adjacencymatrix(S::Matrix, k::Integer)
adjacencymatrix(S::Matrix, r::AbstractFloat)
adjacencymatrix(S::Matrix, r::AbstractFloat, k::Integer; random = true)
adjacencymatrix(M::Matrix; k, r, kwargs...)
```


Computes a spatially weighted adjacency matrix from spatial locations `S` based  on either the `k`-nearest neighbours of each location; all nodes within a disc of fixed radius `r`; or, if both `r` and `k` are provided, a subset of `k` neighbours within a disc of fixed radius `r`.

If `S` is a square matrix, it is treated as a distance matrix; otherwise, it should be an $n$ x $d$ matrix, where $n$ is the number of spatial locations and $d$ is the spatial dimension (typically $d$ = 2). In the latter case, the distance metric is taken to be the Euclidean distance.

Two subsampling strategies are implemented when choosing a subset of `k` neighbours within  a disc of fixed radius `r`. If `random=true` (default), the neighbours are randomly selected from  within the disc. If `random=false`, a deterministic algorithm is used  that aims to preserve the distribution of distances within the neighbourhood set, by choosing  those nodes with distances to the central node corresponding to the  $\{0, \frac{1}{k}, \frac{2}{k}, \dots, \frac{k-1}{k}, 1\}$ quantiles of the empirical  distribution function of distances within the disc (this in fact yields up to $k+1$ neighbours,  since both the closest and furthest nodes are always included). 

By convention with the functionality in `GraphNeuralNetworks.jl` which is based on directed graphs,  the neighbours of location `i` are stored in the column `A[:, i]` where `A` is the  returned adjacency matrix. Therefore, the number of neighbours for each location is given by `collect(mapslices(nnz, A; dims = 1))`, and the number of times each node is  a neighbour of another node is given by `collect(mapslices(nnz, A; dims = 2))`.

By convention, we do not consider a location to neighbour itself (i.e., the diagonal elements of the adjacency matrix are zero). 

**Examples**

```julia
using NeuralEstimators, Distances, SparseArrays

n = 250
d = 2
S = rand(Float32, n, d)
k = 10
r = 0.10

# Memory efficient constructors
adjacencymatrix(S, k)
adjacencymatrix(S, r)
adjacencymatrix(S, r, k)
adjacencymatrix(S, r, k; random = false)

# Construct from full distance matrix D
D = pairwise(Euclidean(), S, dims = 1)
adjacencymatrix(D, k)
adjacencymatrix(D, r)
adjacencymatrix(D, r, k)
adjacencymatrix(D, r, k; random = false)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/Graphs.jl#L331-L387" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.containertype' href='#NeuralEstimators.containertype'><span class="jlbinding">NeuralEstimators.containertype</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
containertype(A::Type)
containertype(::Type{A}) where A <: SubArray
containertype(a::A) where A
```


Returns the container type of its argument.

If given a `SubArray`, returns the container type of the parent array.

**Examples**

```julia
a = rand(3, 4)
containertype(a)
containertype(typeof(a))
[containertype(x) for x ∈ eachcol(a)]
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/utility.jl#L205-L220" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.encodedata' href='#NeuralEstimators.encodedata'><span class="jlbinding">NeuralEstimators.encodedata</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
encodedata(Z::A; c::T = zero(T)) where {A <: AbstractArray{Union{Missing, T}, N}} where T, N
```


For data `Z` with missing entries, returns an encoded data set `(U, W)` where  `U` is the original data `Z` with missing entries replaced by a fixed constant `c`,  and `W` encodes the missingness pattern as an indicator array  equal to one if the corresponding element of `Z` is observed and zero otherwise.

The behavior depends on the dimensionality of `Z`. If `Z` has 1 or 2 dimensions,  the indicator array `W` is concatenated along the first dimension of `Z`. If `Z` has more than 2  dimensions, `W` is concatenated along the second-to-last dimension of `Z`. 

**Examples**

```julia
using NeuralEstimators

Z = rand(16, 16, 1, 1)
Z = removedata(Z, 0.25)	# remove 25% of the data at random
UW = encodedata(Z)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/missingdata.jl#L408-L427" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.expandgrid' href='#NeuralEstimators.expandgrid'><span class="jlbinding">NeuralEstimators.expandgrid</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
expandgrid(xs, ys)
```


Generates a grid of all possible combinations of the elements from two input vectors, `xs` and `ys`. 

Same as `expand.grid()` in `R`, but currently caters for two dimensions only.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/utility.jl#L361-L366" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.maternchols' href='#NeuralEstimators.maternchols'><span class="jlbinding">NeuralEstimators.maternchols</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
maternchols(D, ρ, ν, σ² = 1; stack = true)
```


Given a matrix `D` of distances, constructs the Cholesky factor of the covariance matrix under the Matérn covariance function with range parameter `ρ`, smoothness parameter `ν`, and marginal variance `σ²`.

Providing vectors of parameters will yield a three-dimensional array of Cholesky factors (note that the vectors must of the same length, but a mix of vectors and scalars is allowed). A vector of distance matrices `D` may also be provided.

If `stack = true`, the Cholesky factors will be &quot;stacked&quot; into a three-dimensional array (this is only possible if all distance matrices in `D` are the same size).

**Examples**

```julia
using NeuralEstimators
using LinearAlgebra: norm
n  = 10
S  = rand(n, 2)
D  = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S), sⱼ ∈ eachrow(S)]
ρ  = [0.6, 0.5]
ν  = [0.7, 1.2]
σ² = [0.2, 0.4]
maternchols(D, ρ, ν)
maternchols([D], ρ, ν)
maternchols(D, ρ, ν, σ²; stack = false)

S̃  = rand(n, 2)
D̃  = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S̃), sⱼ ∈ eachrow(S̃)]
maternchols([D, D̃], ρ, ν, σ²)
maternchols([D, D̃], ρ, ν, σ²; stack = false)

S̃  = rand(2n, 2)
D̃  = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S̃), sⱼ ∈ eachrow(S̃)]
maternchols([D, D̃], ρ, ν, σ²; stack = false)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/modelspecificfunctions.jl#L202-L239" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.removedata' href='#NeuralEstimators.removedata'><span class="jlbinding">NeuralEstimators.removedata</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
removedata(Z::Array, Iᵤ::Vector{T}) where T <: Union{Integer, CartesianIndex}
removedata(Z::Array, p::Union{Float, Vector{Float}}; prevent_complete_missing = true)
removedata(Z::Array, n::Integer; fixed_pattern = false, contiguous_pattern = false)
```


Replaces elements of `Z` with `missing`.

The simplest method accepts a vector `Iᵤ` that specifes the indices of the data to be removed.

Alternatively, there are two methods available to randomly generate missing data.

First, a vector `p` may be given that specifies the proportion of missingness for each element in the response vector. Hence, `p` should have length equal to the dimension of the response vector. If a single proportion is given, it will be replicated accordingly. If `prevent_complete_missing = true`, no replicates will contain 100% missingness (note that this can slightly alter the effective values of `p`).

Second, if an integer `n` is provided, all replicates will contain `n` observations after the data are removed. If `fixed_pattern = true`, the missingness pattern is fixed for all replicates. If `contiguous_pattern = true`, the data will be removed in a contiguous block based on a randomly selected starting index. 

The return type is `Array{Union{T, Missing}}`.

**Examples**

```julia
d = 5           # dimension of each replicate
m = 2000        # number of replicates
Z = rand(d, m)  # simulated data

# Passing a desired proportion of missingness
p = rand(d)
removedata(Z, p)

# Passing a desired final sample size
n = 3  # number of observed elements of each replicate: must have n <= d
removedata(Z, n)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/missingdata.jl#L266-L305" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.rowwisenorm' href='#NeuralEstimators.rowwisenorm'><span class="jlbinding">NeuralEstimators.rowwisenorm</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
rowwisenorm(A)
```


Computes the row-wise norm of a matrix `A`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/utility.jl#L110-L113" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.spatialgraph' href='#NeuralEstimators.spatialgraph'><span class="jlbinding">NeuralEstimators.spatialgraph</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
spatialgraph(S)
spatialgraph(S, Z)
spatialgraph(g::GNNGraph, Z)
```


Given spatial data `Z` measured at spatial locations `S`, constructs a [`GNNGraph`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/stable/api/gnngraph/#GNNGraph-type) ready for use in a graph neural network that employs [`SpatialGraphConv`](/API/architectures#NeuralEstimators.SpatialGraphConv) layers. 

When $m$ independent replicates are collected over the same set of $n$ spatial locations,

$$\{\boldsymbol{s}_1, \dots, \boldsymbol{s}_n\} \subset \mathcal{D},$$

where $\mathcal{D} \subset \mathbb{R}^d$ denotes the spatial domain of interest,  `Z` should be given as an $n \times m$ matrix and `S` should be given as an $n \times d$ matrix.  Otherwise, when $m$ independent replicates are collected over differing sets of spatial locations,

$$\{\boldsymbol{s}_{ij}, \dots, \boldsymbol{s}_{in_i}\} \subset \mathcal{D}, \quad i = 1, \dots, m,$$

`Z` should be given as an $m$-vector of $n_i$-vectors, and `S` should be given as an $m$-vector of $n_i \times d$ matrices.

The spatial information between neighbours is stored as an edge feature, with the specific  information controlled by the keyword arguments `stationary` and `isotropic`.  Specifically, the edge feature between node $j$ and node $j'$ stores the spatial  distance $\|\boldsymbol{s}_{j'} - \boldsymbol{s}_j\|$ (if `isotropic`), the spatial  displacement $\boldsymbol{s}_{j'} - \boldsymbol{s}_j$ (if `stationary`), or the matrix of   locations $(\boldsymbol{s}_{j'}, \boldsymbol{s}_j)$ (if `!stationary`).  

Additional keyword arguments inherit from [`adjacencymatrix()`](/API/miscellaneous#NeuralEstimators.adjacencymatrix) to determine the neighbourhood of each node, with the default being a randomly selected set of  `k=30` neighbours within a disc of radius `r=0.15` units.

**Examples**

```julia
using NeuralEstimators, GraphNeuralNetworks

# Number of replicates and spatial dimension
m = 5  
d = 2  

# Spatial locations fixed for all replicates
n = 100
S = rand(n, d)
Z = rand(n, m)
g = spatialgraph(S, Z)

# Spatial locations varying between replicates
n = rand(50:100, m)
S = rand.(n, d)
Z = rand.(n)
g = spatialgraph(S, Z)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/Graphs.jl#L3-L54" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.stackarrays' href='#NeuralEstimators.stackarrays'><span class="jlbinding">NeuralEstimators.stackarrays</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
stackarrays(v::Vector{<:AbstractArray}; merge::Bool = true)
```


Stack a vector of arrays `v` into a single higher-dimensional array.

If all arrays have the same size along the last dimension, stacks along a new final dimension. Then, if `merge = true`, merges the last two dimensions into one.

Alternatively, if sizes differ along the last dimension, concatenates along the last dimension.

**Examples**

```julia
# Vector containing arrays of the same size:
Z = [rand(2, 3, m) for m ∈ (1, 1)];
stackarrays(Z)
stackarrays(Z, merge = false)

# Vector containing arrays with differing final dimension size:
Z = [rand(2, 3, m) for m ∈ (1, 2)];
stackarrays(Z)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/utility.jl#L381-L400" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.vectotril' href='#NeuralEstimators.vectotril'><span class="jlbinding">NeuralEstimators.vectotril</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
vectotril(v; strict = false)
vectotriu(v; strict = false)
```


Converts a vector `v` of length $d(d+1)÷2$ (a triangular number) into a $d × d$ lower or upper triangular matrix.

If `strict = true`, the matrix will be _strictly_ lower or upper triangular, that is, a $(d+1) × (d+1)$ triangular matrix with zero diagonal.

Note that the triangular matrix is constructed on the CPU, but the returned matrix will be a GPU array if `v` is a GPU array. Note also that the return type is not of type `Triangular` matrix (i.e., the zeros are materialised) since `Triangular` matrices are not always compatible with other GPU operations.

**Examples**

```julia
using NeuralEstimators

d = 4
n = d*(d+1)÷2
v = collect(range(1, n))
vectotril(v)
vectotriu(v)
vectotril(v; strict = true)
vectotriu(v; strict = true)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/utility.jl#L146-L173" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Model-specific functions {#Model-specific-functions}

### Data simulators {#Data-simulators}

The philosophy of `NeuralEstimators` is to cater for any model for which simulation is feasible by allowing users to define their model implicitly through simulated data. However, the following functions have been included as they may be helpful to others, and their source code illustrates how a user could formulate code for their own model.

See also [Distributions.jl](https://juliastats.org/Distributions.jl/stable/) for a range of distributions implemented in Julia, and the package [RCall](https://juliainterop.github.io/RCall.jl/stable/) for calling R functions within Julia. 
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.simulategaussian' href='#NeuralEstimators.simulategaussian'><span class="jlbinding">NeuralEstimators.simulategaussian</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
simulategaussian(L::AbstractMatrix, m = 1)
```


Simulates `m` independent and identically distributed realisations from a mean-zero multivariate Gaussian random vector with associated lower Cholesky  factor `L`. 

If `m` is not specified, the simulated data are returned as a vector with length equal to the number of spatial locations, $n$; otherwise, the data are returned as an $n$x`m` matrix.

**Examples**

```julia
using NeuralEstimators, Distances, LinearAlgebra

n = 500
ρ = 0.6
ν = 1.0
S = rand(n, 2)
D = pairwise(Euclidean(), S, dims = 1)
Σ = Symmetric(matern.(D, ρ, ν))
L = cholesky(Σ).L
simulategaussian(L)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/modelspecificfunctions.jl#L1-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.simulatepotts' href='#NeuralEstimators.simulatepotts'><span class="jlbinding">NeuralEstimators.simulatepotts</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
simulatepotts(grid::Matrix{Int}, β)
simulatepotts(grid::Matrix{Union{Int, Nothing}}, β)
simulatepotts(nrows::Int, ncols::Int, num_states::Int, β)
```


Chequerboard Gibbs sampling from a spatial Potts model with parameter `β`&gt;0 (see, e.g., [Sainsbury-Dale et al., 2025, Sec. 3.3](https://arxiv.org/abs/2501.04330), and the references therein).

Approximately independent simulations can be obtained by setting  `nsims` &gt; 1 or `num_iterations > burn`. The degree to which the  resulting simulations can be considered independent depends on the  thinning factor (`thin`) and the burn-in (`burn`).

**Keyword arguments**
- `nsims = 1`: number of approximately independent replicates. 
  
- `num_iterations = 2000`: number of MCMC iterations.
  
- `burn = num_iterations`: burn-in.
  
- `thin = 10`: thinning factor.
  

**Examples**

```julia
using NeuralEstimators 

## Marginal simulation 
β = 0.8
simulatepotts(10, 10, 3, β)

## Marginal simulation: approximately independent samples 
simulatepotts(10, 10, 3, β; nsims = 100, thin = 10)

## Conditional simulation 
β = 0.8
complete_grid   = simulatepotts(100, 100, 3, β)      # simulate marginally 
incomplete_grid = removedata(complete_grid, 0.1)     # randomly remove 10% of the pixels 
imputed_grid    = simulatepotts(incomplete_grid, β)  # conditionally simulate over missing pixels

## Multiple conditional simulations 
imputed_grids   = simulatepotts(incomplete_grid, β; num_iterations = 2000, burn = 1000, thin = 10)

## Recreate Fig. 8.8 of Marin & Robert (2007) “Bayesian Core”
using Plots 
grids = [simulatepotts(100, 100, 2, β) for β ∈ 0.3:0.1:1.2]
heatmaps = heatmap.(grids, legend = false, aspect_ratio=1)
Plots.plot(heatmaps...)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/modelspecificfunctions.jl#L305-L348" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.simulateschlather' href='#NeuralEstimators.simulateschlather'><span class="jlbinding">NeuralEstimators.simulateschlather</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
simulateschlather(L::Matrix, m = 1; C = 3.5, Gumbel::Bool = false)
```


Simulates `m` independent and identically distributed realisations from [Schlather&#39;s (2002)](https://link.springer.com/article/10.1023/A:1020977924878) max-stable model given the lower Cholesky factor `L` of the covariance matrix of the underlying Gaussian process. 

The function uses the algorithm for approximate simulation given by [Schlather (2002)](https://link.springer.com/article/10.1023/A:1020977924878).

If `m` is not specified, the simulated data are returned as a vector with length equal to the number of spatial locations, $n$; otherwise, the data are  returned as an $n$x`m` matrix.

**Keyword arguments**
- `C = 3.5`: a tuning parameter that controls the accuracy of the algorithm. Small `C` favours computational efficiency, while large `C` favours accuracy. 
  
- `Gumbel = true`: flag indicating whether the data should be log-transformed from the unit Fréchet scale to the Gumbel scale.
  

**Examples**

```julia
using NeuralEstimators, Distances, LinearAlgebra

n = 500
ρ = 0.6
ν = 1.0
S = rand(n, 2)
D = pairwise(Euclidean(), S, dims = 1)
Σ = Symmetric(matern.(D, ρ, ν))
L = cholesky(Σ).L
simulateschlather(L)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/modelspecificfunctions.jl#L35-L64" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Spatial point processes {#Spatial-point-processes}
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.maternclusterprocess' href='#NeuralEstimators.maternclusterprocess'><span class="jlbinding">NeuralEstimators.maternclusterprocess</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
maternclusterprocess(; λ=10, μ=10, r=0.1, xmin=0, xmax=1, ymin=0, ymax=1, unit_bounding_box=false)
```


Generates a realisation from a Matérn cluster process (e.g., [Baddeley et al., 2015](https://www.taylorfrancis.com/books/mono/10.1201/b19708/spatial-point-patterns-adrian-baddeley-rolf-turner-ege-rubak), Ch. 12). 

The process is defined by a parent homogenous Poisson point process with intensity `λ` &gt; 0, a mean number of daughter points `μ` &gt; 0, and a cluster radius `r` &gt; 0. The simulation is performed over a rectangular window defined by [`xmin, xmax`] × [`ymin`, `ymax`].

If `unit_bounding_box = true`, the simulated points will be scaled so that the longest side of their bounding box is equal to one (this may change the simulation window). 

See also the R package [`spatstat`](https://cran.r-project.org/web/packages/spatstat/index.html), which provides functions for simulating from a range of point processes and which can be interfaced from Julia using [`RCall`](https://juliainterop.github.io/RCall.jl/stable/).

**Examples**

```julia
using NeuralEstimators

# Simulate a realisation from a Matérn cluster process
S = maternclusterprocess()

# Visualise realisation (requires UnicodePlots)
using UnicodePlots
scatterplot(S[:, 1], S[:, 2])

# Visualise realisations from the cluster process with varying parameters
n = 250
λ = [10, 25, 50, 90]
μ = n ./ λ
plots = map(eachindex(λ)) do i
	S = maternclusterprocess(λ = λ[i], μ = μ[i])
	scatterplot(S[:, 1], S[:, 2])
end
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/Graphs.jl#L591-L626" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Covariance functions {#Covariance-functions}

These covariance functions may be of use for various models.
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.matern' href='#NeuralEstimators.matern'><span class="jlbinding">NeuralEstimators.matern</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
matern(h, ρ, ν, σ² = 1)
```


Given distance $\|\boldsymbol{h}\|$ (`h`), computes the Matérn covariance function

$$C(\|\boldsymbol{h}\|) = \sigma^2 \frac{2^{1 - \nu}}{\Gamma(\nu)} \left(\frac{\|\boldsymbol{h}\|}{\rho}\right)^\nu K_\nu \left(\frac{\|\boldsymbol{h}\|}{\rho}\right),$$

where `ρ` is a range parameter, `ν` is a smoothness parameter, `σ²` is the marginal variance,  $\Gamma(\cdot)$ is the gamma function, and $K_\nu(\cdot)$ is the modified Bessel function of the second kind of order $\nu$.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/modelspecificfunctions.jl#L115-L126" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.paciorek' href='#NeuralEstimators.paciorek'><span class="jlbinding">NeuralEstimators.paciorek</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
paciorek(s, r, ω₁, ω₂, ρ, β)
```


Given spatial locations `s` and `r`, computes the nonstationary covariance function 

$$C(\boldsymbol{s}, \boldsymbol{r}) = 
|\boldsymbol{\Sigma}(\boldsymbol{s})|^{1/4}
|\boldsymbol{\Sigma}(\boldsymbol{r})|^{1/4}
\left|\frac{\boldsymbol{\Sigma}(\boldsymbol{s}) + \boldsymbol{\Sigma}(\boldsymbol{r})}{2}\right|^{-1/2}
C^0\big(\sqrt{Q(\boldsymbol{s}, \boldsymbol{r})}\big), $$

where $C^0(h) = \exp\{-(h/\rho)^{3/2}\}$ for range parameter $\rho > 0$,  the matrix $\boldsymbol{\Sigma}(\boldsymbol{s}) = \exp(\beta\|\boldsymbol{s} - \boldsymbol{\omega}\|)\boldsymbol{I}$  is a kernel matrix ([Paciorek and Schervish, 2006](https://onlinelibrary.wiley.com/doi/abs/10.1002/env.785))  with scale parameter $\beta > 0$ and reference point $\boldsymbol{\omega} \equiv (\omega_1, \omega_2)' \in \mathbb{R}^2$, and 

$$Q(\boldsymbol{s}, \boldsymbol{r}) = 
(\boldsymbol{s} - \boldsymbol{r})'
\left(\frac{\boldsymbol{\Sigma}(\boldsymbol{s}) + \boldsymbol{\Sigma}(\boldsymbol{r})}{2}\right)^{-1}
(\boldsymbol{s} - \boldsymbol{r})$$

is the squared Mahalanobis distance between $\boldsymbol{s}$ and $\boldsymbol{r}$. 

Note that, in practical applications, the reference point $\boldsymbol{\omega}$ is often taken to be an estimable parameter rather than fixed and known. 


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/modelspecificfunctions.jl#L145-L169" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Density functions {#Density-functions}

Density functions are not needed in the workflow of `NeuralEstimators`. However, as part of a series of comparison studies between neural estimators and likelihood-based estimators given in various paper, we have developed the following functions for evaluating the density function for several popular distributions. We include these in `NeuralEstimators` to cater for the possibility that they may be of use in future comparison studies.
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.gaussiandensity' href='#NeuralEstimators.gaussiandensity'><span class="jlbinding">NeuralEstimators.gaussiandensity</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
gaussiandensity(Z::V, L::LT) where {V <: AbstractVector, LT <: LowerTriangular}
gaussiandensity(Z::A, L::LT) where {A <: AbstractArray, LT <: LowerTriangular}
gaussiandensity(Z::A, Σ::M) where {A <: AbstractArray, M <: AbstractMatrix}
```


Efficiently computes the density function for `Z` ~ 𝑁(0, `Σ`), namely,  

$$|2\pi\boldsymbol{\Sigma}|^{-1/2} \exp\{-\frac{1}{2}\boldsymbol{Z}^\top \boldsymbol{\Sigma}^{-1}\boldsymbol{Z}\},$$

for covariance matrix `Σ`, and where `L` is lower Cholesky factor of `Σ`.

The method `gaussiandensity(Z::A, L::LT)` assumes that the last dimension of `Z` contains independent and identically distributed replicates.

If `logdensity = true` (default), the log-density is returned.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/modelspecificfunctions.jl#L573-L587" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.schlatherbivariatedensity' href='#NeuralEstimators.schlatherbivariatedensity'><span class="jlbinding">NeuralEstimators.schlatherbivariatedensity</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
schlatherbivariatedensity(z₁, z₂, ψ₁₂; logdensity = true)
```


The bivariate density function (see, e.g., [Sainsbury-Dale et al., 2024](https://www.tandfonline.com/doi/suppl/10.1080/00031305.2023.2249522?scroll=top), Sec. S6.2) for [Schlather&#39;s (2002)](https://link.springer.com/article/10.1023/A:1020977924878) max-stable model, where `ψ₁₂` denotes the spatial correlation function evaluated at the locations of observations `z₁` and `z₂`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/modelspecificfunctions.jl#L605-L608" target="_blank" rel="noreferrer">source</a></Badge>

</details>

