


# Neural-network building blocks {#Neural-network-building-blocks}

Any [Flux](https://fluxml.ai/Flux.jl/stable/) model can be used to construct a neural network when using the package. In addition to the standard Flux layers and architectures, the following components can be useful.

## Modules {#Modules}

The structures listed below are often useful when constructing neural estimators. In particular, [`DeepSet`](/API/architectures#NeuralEstimators.DeepSet) provides a convenient wrapper for embedding standard neural networks (e.g., MLPs, CNNs, GNNs) into a framework suited to making inference with an arbitrary number of independent replicates. 
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.DeepSet' href='#NeuralEstimators.DeepSet'><span class="jlbinding">NeuralEstimators.DeepSet</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DeepSet(ψ, ϕ, a = mean; S = nothing)
(ds::DeepSet)(Z::Vector{A}) where A <: Any
(ds::DeepSet)(tuple::Tuple{Vector{A}, Vector{Vector}}) where A <: Any
```


The DeepSets representation ([Zaheer et al., 2017](https://arxiv.org/abs/1703.06114); [Sainsbury-Dale et al., 2024](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522)),

$$\hat{\boldsymbol{\theta}}(\mathbf{Z}) = \boldsymbol{\phi}(\mathbf{T}(\mathbf{Z})), \quad
\mathbf{T}(\mathbf{Z}) = \mathbf{a}(\{\boldsymbol{\psi}(\mathbf{Z}_i) : i = 1, \dots, m\}),$$

where 𝐙 ≡ (𝐙₁&#39;, …, 𝐙ₘ&#39;)&#39; are independent replicates of data,  `ψ` and `ϕ` are neural networks, and `a` is a permutation-invariant aggregation function. 

The function `a` must operate on arrays and have a keyword argument `dims` for  specifying the dimension of aggregation (e.g., `mean`, `sum`, `maximum`, `minimum`, `logsumexp`).

`DeepSet` objects act on data of type `Vector{A}`, where each element of the vector is associated with one data set (i.e., one set of independent replicates), and where `A` depends on the chosen architecture for `ψ`.  Independent replicates within each data set are stored in the batch dimension.  For example, data collected over a two-dimensional grid and `ψ` a CNN, `A` should be a 4-dimensional array,  with replicates stored in the 4ᵗʰ dimension. 

For computational efficiency,  array data are first concatenated along their final dimension  (i.e., the replicates dimension) before being passed into the inner network `ψ`,  thereby ensuring that `ψ` is applied to a single large array, rather than multiple small ones. 

Fixed (non-trainable) transformations of the data can be incorporated alongside the learned summaries via the `S` argument:

$$\hat{\boldsymbol{\theta}}(\mathbf{Z}) = \boldsymbol{\phi}((\mathbf{T}(\mathbf{Z})', \mathbf{S}(\mathbf{Z})')'),$$

where `S` is a function (or vector of functions) that maps data to a vector of fixed summary statistics. These are not differentiated through during training. In the case that `ψ` is set to `nothing`, only the fixed summaries will be used. For the common case where summary statistics are precomputed and stored alongside the data, see [`DataSet`](/API/parametersdata#NeuralEstimators.DataSet) as an alternative approach.

Set-level inputs (e.g., covariates) $𝐗$ can be passed directly into the outer network `ϕ` in the following manner: 

$$\hat{\boldsymbol{\theta}}(\mathbf{Z}) = \boldsymbol{\phi}((\mathbf{T}(\mathbf{Z})', \mathbf{X}')'),$$

or, when fixed transformations are also used,

$$\hat{\boldsymbol{\theta}}(\mathbf{Z}) = \boldsymbol{\phi}((\mathbf{T}(\mathbf{Z})', \mathbf{S}(\mathbf{Z})', \mathbf{X}')').$$

This is done by calling the `DeepSet` object on a `Tuple{Vector{A}, Vector{Vector}}`, where the first element of the tuple contains a vector of data sets and the second element contains a vector of set-level inputs (i.e., one vector for each data set).

**Examples**

```julia
using NeuralEstimators, Flux

# Two data sets containing 3 and 4 replicates
d = 5  # number of parameters in the model
n = 10 # dimension of each replicate
Z = [rand32(n, m) for m ∈ (3, 4)]

# Construct DeepSet object
dₜ = 16  # dimension of neural summary statistic
w  = 32  # width of hidden layers
ψ  = Chain(Dense(n, w, relu), Dense(w, dₜ, relu))
ϕ  = Chain(Dense(dₜ, w, relu), Dense(w, d))
ds = DeepSet(ψ, ϕ)

# Apply DeepSet object to data
ds(Z)

# With fixed transformations S
dₛ = 1   # dimension of fixed summary statistic
ϕ  = Chain(Dense(dₜ + dₛ, w, relu), Dense(w, d))
ds = DeepSet(ψ, ϕ; S = logsamplesize)
ds(Z)

# With set-level inputs 
dₓ = 2 # dimension of set-level inputs 
ϕ  = Chain(Dense(dₜ + dₓ, w, relu), Dense(w, d))
ds = DeepSet(ψ, ϕ)
X  = [rand32(dₓ) for _ ∈ eachindex(Z)]
ds((Z, X))
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Architectures.jl#L42-L122" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.GNNSummary' href='#NeuralEstimators.GNNSummary'><span class="jlbinding">NeuralEstimators.GNNSummary</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
GNNSummary(propagation, readout)
```


A graph neural network (GNN) module designed to serve as the inner network `ψ` in the [`DeepSet`](/API/architectures#NeuralEstimators.DeepSet) representation when the data are graphical (e.g., irregularly observed spatial data).

The `propagation` module transforms graph data into a set of hidden-feature graphs. The `readout` module aggregates these feature graphs into a single hidden feature vector of fixed length. The network `ψ` is then defined as the composition of the propagation and readout modules.

The data should be stored as a `GNNGraph` or `Vector{GNNGraph}`, where each graph is associated with a single parameter vector. The graphs may contain subgraphs corresponding to independent replicates.

**Examples**

```julia
using NeuralEstimators, Flux, GraphNeuralNetworks
using Statistics: mean

# Spatial data
n = 100                # number of spatial locations
m = 50                 # number of independent replicates
S = rand(n, 2)         # spatial locations
Z = rand(n, m)         # observed data
g = spatialgraph(S, Z) # construct the graph

# Propagation module
nₕ = 32    # dimension of node feature vectors
propagation = Chain(SpatialGraphConv(1 => nₕ), SpatialGraphConv(nₕ => nₕ))

# Readout module
readout = GlobalPool(mean)

# Inner network
ψ = GNNSummary(propagation, readout)

# Outer network
d = 3     # number of parameters
w = 64    # width of hidden layer
ϕ = Chain(Dense(nₕ, w, relu), Dense(w, d))

# DeepSet object 
ds = DeepSet(ψ, ϕ)

# Apply to data 
ds(g)        # single graph with subgraphs corresponding to independent replicates
ds([g, g])   # vector of graphs, corresponding to multiple data sets 
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Graphs.jl#L134-L183" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.MLP' href='#NeuralEstimators.MLP'><span class="jlbinding">NeuralEstimators.MLP</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MLP(in::Integer, out::Integer; kwargs...)
(mlp::MLP)(x)
(mlp::MLP)(x, y)
```


A traditional fully-connected multilayer perceptron (MLP) with input dimension `in` and output dimension `out`.

The method `(mlp::MLP)(x, y)` concatenates `x` and `y` along their first dimension before passing the result through the neural network. This functionality is used in constructs such as [`AffineCouplingBlock`](/API/approximatedistributions#NeuralEstimators.AffineCouplingBlock). 

**Keyword arguments**
- `depth::Integer = 2`: the number of hidden layers.
  
- `width::Integer = 128`: the width of each hidden layer.
  
- `activation::Function = relu`: the (non-linear) activation function used in each hidden layer.
  
- `output_activation = identity`: the activation function used in the output layer.
  
- `final_layer = nothing`: an optional final layer to append to the network. If provided, it must accept `width` inputs. When set, `output_activation` is ignored and replaced with identity. The effective depth of the network becomes `depth + 1`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Architectures.jl#L745-L759" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## User-defined summary statistics {#User-defined-summary-statistics}


The following functions correspond to summary statistics that are often useful as user-defined summaries in [`DeepSet`](/API/architectures#NeuralEstimators.DeepSet) objects.
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.samplesize' href='#NeuralEstimators.samplesize'><span class="jlbinding">NeuralEstimators.samplesize</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
samplesize(Z)
```


Computes the number of independent replicates in the data set `Z`. 

Note that this function is a wrapper around [`numberreplicates`](/API/miscellaneous#NeuralEstimators.numberreplicates) with return type equal to the eltype of `Z`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/summarystatistics.jl#L1-L6" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.logsamplesize' href='#NeuralEstimators.logsamplesize'><span class="jlbinding">NeuralEstimators.logsamplesize</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
logsamplesize(Z)
```


Computes the log of the number of independent replicates in the data set `Z`. 


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/summarystatistics.jl#L9-L12" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.invsqrtsamplesize' href='#NeuralEstimators.invsqrtsamplesize'><span class="jlbinding">NeuralEstimators.invsqrtsamplesize</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
invsqrtsamplesize(Z)
```


Computes the inverse of the square root of the number of independent replicates in the data set `Z`. 


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/summarystatistics.jl#L15-L18" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.samplecorrelation' href='#NeuralEstimators.samplecorrelation'><span class="jlbinding">NeuralEstimators.samplecorrelation</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
samplecorrelation(Z::AbstractArray)
```


Computes the sample correlation matrix, R̂, and returns the vectorised strict lower triangle of R̂.

**Examples**

```julia
# 5 independent replicates of a 3-dimensional vector
z = rand(3, 5)
samplecorrelation(z)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/summarystatistics.jl#L44-L56" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.samplecovariance' href='#NeuralEstimators.samplecovariance'><span class="jlbinding">NeuralEstimators.samplecovariance</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
samplecovariance(Z::AbstractArray)
```


Computes the [sample covariance matrix](https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Definition_of_sample_covariance), Σ̂, and returns the vectorised lower triangle of Σ̂.

**Examples**

```julia
# 5 independent replicates of a 3-dimensional vector
z = rand(3, 5)
samplecovariance(z)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/summarystatistics.jl#L21-L33" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.NeighbourhoodVariogram' href='#NeuralEstimators.NeighbourhoodVariogram'><span class="jlbinding">NeuralEstimators.NeighbourhoodVariogram</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
NeighbourhoodVariogram(h_max, n_bins) 
(l::NeighbourhoodVariogram)(g::GNNGraph)
```


Computes the empirical variogram, 

$$\hat{\gamma}(h \pm \delta) = \frac{1}{2|N(h \pm \delta)|} \sum_{(i,j) \in N(h \pm \delta)} (Z_i - Z_j)^2$$

where $N(h \pm \delta) \equiv \left\{(i,j) : \|\boldsymbol{s}_i - \boldsymbol{s}_j\| \in (h-\delta, h+\delta)\right\}$  is the set of pairs of locations separated by a distance within $(h-\delta, h+\delta)$, and $|\cdot|$ denotes set cardinality. 

The distance bins are constructed to have constant width $2\delta$, chosen based on the maximum distance  `h_max` to be considered, and the specified number of bins `n_bins`. 

The input type is a `GNNGraph`, and the empirical variogram is computed based on the corresponding graph structure.  Specifically, only locations that are considered neighbours will be used when computing the empirical variogram. 

**Examples**

```julia
using NeuralEstimators, GraphNeuralNetworks, Distances, LinearAlgebra
  
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



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Graphs.jl#L193-L236" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Layers {#Layers}

In addition to the [built-in layers](https://fluxml.ai/Flux.jl/stable/reference/models/layers/) provided by Flux, the following layers may be used when building a neural-network architecture.
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.DensePositive' href='#NeuralEstimators.DensePositive'><span class="jlbinding">NeuralEstimators.DensePositive</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DensePositive(layer::Dense; g::Function = relu, last_only::Bool = false)
```


Wrapper around the standard [Dense](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense) layer that ensures positive weights (biases are left unconstrained).

This layer can be useful for constucting (partially) monotonic neural networks. 

**Examples**

```julia
using NeuralEstimators, Flux

l = DensePositive(Dense(5 => 2))
x = rand32(5, 64)
l(x)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Architectures.jl#L601-L617" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.ResidualBlock' href='#NeuralEstimators.ResidualBlock'><span class="jlbinding">NeuralEstimators.ResidualBlock</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ResidualBlock(filter, in => out; stride = 1)
```


Basic residual block (see [here](https://en.wikipedia.org/wiki/Residual_neural_network#Basic_block)), consisting of two sequential convolutional layers and a skip (shortcut) connection that connects the input of the block directly to the output, facilitating the training of deep networks.

**Examples**

```julia
using NeuralEstimators
z = rand(16, 16, 1, 1)
b = ResidualBlock((3, 3), 1 => 32)
b(z)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Architectures.jl#L696-L711" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.SpatialGraphConv' href='#NeuralEstimators.SpatialGraphConv'><span class="jlbinding">NeuralEstimators.SpatialGraphConv</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SpatialGraphConv(in => out, g=relu; args...)
```


Implements a spatial graph convolution for isotropic spatial processes [(Sainsbury-Dale et al., 2025)](https://arxiv.org/abs/2310.02600), 

$$ \boldsymbol{h}^{(l)}_{j} =
 g\Big(
 \boldsymbol{\Gamma}_{\!1}^{(l)} \boldsymbol{h}^{(l-1)}_{j}
 +
 \boldsymbol{\Gamma}_{\!2}^{(l)} \bar{\boldsymbol{h}}^{(l)}_{j}
 +
 \boldsymbol{\gamma}^{(l)}
 \Big),
 \quad
 \bar{\boldsymbol{h}}^{(l)}_{j} = \sum_{j' \in \mathcal{N}(j)}\boldsymbol{w}^{(l)}(\|\boldsymbol{s}_{j'} - \boldsymbol{s}_j\|) \odot f^{(l)}(\boldsymbol{h}^{(l-1)}_{j}, \boldsymbol{h}^{(l-1)}_{j'}),$$

where $\boldsymbol{h}^{(l)}_{j}$ is the hidden feature vector at location $\boldsymbol{s}_j$ at layer $l$, $g(\cdot)$ is a non-linear activation function applied elementwise, $\boldsymbol{\Gamma}_{\!1}^{(l)}$ and $\boldsymbol{\Gamma}_{\!2}^{(l)}$ are trainable parameter matrices, $\boldsymbol{\gamma}^{(l)}$ is a trainable bias vector, $\mathcal{N}(j)$ denotes the indices of neighbours of $\boldsymbol{s}_j$, $\boldsymbol{w}^{(l)}(\cdot)$ is a (learnable) spatial weighting function, $\odot$ denotes elementwise multiplication,  and $f^{(l)}(\cdot, \cdot)$ is a (learnable) function. 

By default, the function $f^{(l)}(\cdot, \cdot)$ is modelled using a [`PowerDifference`](/API/architectures#NeuralEstimators.PowerDifference) function.  One may alternatively employ a nonlearnable function, for example, `f = (hᵢ, hⱼ) -> (hᵢ - hⱼ).^2`,  specified through the keyword argument `f`.  

The spatial distances between locations must be stored as an edge feature, as facilitated by [`spatialgraph()`](/API/miscellaneous#NeuralEstimators.spatialgraph).  The input to $\boldsymbol{w}^{(l)}(\cdot)$ is a $1 \times n$ matrix (i.e., a row vector) of spatial distances.  The output of $\boldsymbol{w}^{(l)}(\cdot)$ must be either a scalar; a vector of the same dimension as the feature vectors of the previous layer;  or, if the features vectors of the previous layer are scalars, a vector of arbitrary dimension.  To promote identifiability, the weights are normalised to sum to one (row-wise) within each neighbourhood set.  By default, $\boldsymbol{w}^{(l)}(\cdot)$ is taken to be a multilayer perceptron with a single hidden layer,  although a custom choice for this function can be provided using the keyword argument `w`. 

**Arguments**
- `in`: dimension of input features.
  
- `out`: dimension of output features.
  
- `g = relu`: activation function.
  
- `bias = true`: add learnable bias?
  
- `init = glorot_uniform`: initialiser for $\boldsymbol{\Gamma}_{\!1}^{(l)}$, $\boldsymbol{\Gamma}_{\!2}^{(l)}$, and $\boldsymbol{\gamma}^{(l)}$. 
  
- `f = nothing`
  
- `w = nothing` 
  
- `w_width = 128` (applicable only if `w = nothing`): the width of the hidden layer in the MLP used to model $\boldsymbol{w}^{(l)}(\cdot, \cdot)$. 
  
- `w_out = in` (applicable only if `w = nothing`): the output dimension of $\boldsymbol{w}^{(l)}(\cdot, \cdot)$.  
  

**Examples**

```julia
using NeuralEstimators, Flux, GraphNeuralNetworks

# Toy spatial data
n = 250                # number of spatial locations
m = 5                  # number of independent replicates
S = rand(n, 2)         # spatial locations
Z = rand(n, m)         # data
g = spatialgraph(S, Z) # construct the graph

# Construct and apply spatial graph convolution layer
l = SpatialGraphConv(1 => 10)
l(g)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Graphs.jl#L57-L122" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Output layers {#Output-layers}


In addition to the [standard activation functions](https://fluxml.ai/Flux.jl/stable/models/activation/) provided by Flux (e.g., `relu`, `softplus`), the following layers can be used at the end of an architecture to ensure valid estimates for certain models. Note that the Flux layer `Parallel` can be useful for applying several different parameter constraints.

::: tip Layers vs. activation functions

Although we may conceptualise the following types as &quot;output activation functions&quot;, they should be treated as separate layers included in the final stage of a Flux `Chain()`. In particular, they cannot be used as the activation function of a `Dense` layer. 

:::
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.Compress' href='#NeuralEstimators.Compress'><span class="jlbinding">NeuralEstimators.Compress</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Compress(a, b, k = 1)
```


Layer that compresses its input to be within the range `a` and `b`, where each element of `a` is less than the corresponding element of `b`.

The layer uses a logistic function,

$$l(θ) = a + \frac{b - a}{1 + e^{-kθ}},$$

where the arguments `a` and `b` together combine to shift and scale the logistic function to the range (`a`, `b`), and the growth rate `k` controls the steepness of the curve.

The logistic function given [here](https://en.wikipedia.org/wiki/Logistic_function) contains an additional parameter, θ₀, which is the input value corresponding to the functions midpoint. In `Compress`, we fix θ₀ = 0, since the output of a randomly initialised neural network is typically around zero.

**Examples**

```julia
using NeuralEstimators, Flux

a = [25, 0.5, -pi/2]
b = [500, 2.5, 0]
p = length(a)
K = 100
θ = randn(p, K)
l = Compress(a, b)
l(θ)

n = 20
θ̂ = Chain(Dense(n, p), l)
Z = randn(n, K)
θ̂(Z)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Architectures.jl#L265-L302" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.CorrelationMatrix' href='#NeuralEstimators.CorrelationMatrix'><span class="jlbinding">NeuralEstimators.CorrelationMatrix</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CorrelationMatrix(d)
(object::CorrelationMatrix)(x::Matrix, cholesky::Bool = false)
```


Transforms a vector 𝐯 ∈ ℝᵈ to the parameters of an unconstrained `d`×`d` correlation matrix or, if `cholesky = true`, the lower Cholesky factor of an unconstrained `d`×`d` correlation matrix.

The expected input is a `Matrix` with T(`d`-1) = (`d`-1)`d`÷2 rows, where T(`d`-1) is the (`d`-1)th triangular number (the number of free parameters in an unconstrained `d`×`d` correlation matrix), and the output is a `Matrix` of the same dimension. The columns of the input and output matrices correspond to independent parameter configurations (i.e., different correlation matrices).

Internally, the layer constructs a valid Cholesky factor 𝐋 for a correlation matrix, and then extracts the strict lower triangle from the correlation matrix 𝐑 = 𝐋𝐋&#39;. The lower triangle is extracted and vectorised in line with Julia&#39;s column-major ordering: for example, when modelling the correlation matrix

$$\begin{bmatrix}
1   & R₁₂ &  R₁₃ \\
R₂₁ & 1   &  R₂₃\\
R₃₁ & R₃₂ & 1\\
\end{bmatrix},$$

the rows of the matrix returned by a `CorrelationMatrix` layer are ordered as

$$\begin{bmatrix}
R₂₁ \\
R₃₁ \\
R₃₂ \\
\end{bmatrix},$$

which means that the output can easily be transformed into the implied correlation matrices using [`vectotril`](/API/miscellaneous#NeuralEstimators.vectotril) and `Symmetric`.

See also [`CovarianceMatrix`](/API/architectures#NeuralEstimators.CovarianceMatrix).

**Examples**

```julia
using NeuralEstimators, LinearAlgebra, Flux

d  = 4
l  = CorrelationMatrix(d)
p  = (d-1)*d÷2
θ  = randn(p, 100)

# Returns a matrix of parameters, which can be converted to correlation matrices
R = l(θ)
R = map(eachcol(R)) do r
	R = Symmetric(cpu(vectotril(r, strict = true)), :L)
	R[diagind(R)] .= 1
	R
end

# Obtain the Cholesky factor directly
L = l(θ, true)
L = map(eachcol(L)) do x
	# Only the strict lower diagonal elements are returned
	L = LowerTriangular(cpu(vectotril(x, strict = true)))

	# Diagonal elements are determined under the constraint diag(L*L') = 𝟏
	L[diagind(L)] .= sqrt.(1 .- rowwisenorm(L).^2)
	L
end
L[1] * L[1]'
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Architectures.jl#L449-L519" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.CovarianceMatrix' href='#NeuralEstimators.CovarianceMatrix'><span class="jlbinding">NeuralEstimators.CovarianceMatrix</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CovarianceMatrix(d)
(object::CovarianceMatrix)(x::Matrix, cholesky::Bool = false)
```


Transforms a vector 𝐯 ∈ ℝᵈ to the parameters of an unconstrained `d`×`d` covariance matrix or, if `cholesky = true`, the lower Cholesky factor of an unconstrained `d`×`d` covariance matrix.

The expected input is a `Matrix` with T(`d`) = `d`(`d`+1)÷2 rows, where T(`d`) is the `d`th triangular number (the number of free parameters in an unconstrained `d`×`d` covariance matrix), and the output is a `Matrix` of the same dimension. The columns of the input and output matrices correspond to independent parameter configurations (i.e., different covariance matrices).

Internally, the layer constructs a valid Cholesky factor 𝐋 and then extracts the lower triangle from the positive-definite covariance matrix 𝚺 = 𝐋𝐋&#39;. The lower triangle is extracted and vectorised in line with Julia&#39;s column-major ordering: for example, when modelling the covariance matrix

$$\begin{bmatrix}
Σ₁₁ & Σ₁₂ & Σ₁₃ \\
Σ₂₁ & Σ₂₂ & Σ₂₃ \\
Σ₃₁ & Σ₃₂ & Σ₃₃ \\
\end{bmatrix},$$

the rows of the matrix returned by a `CovarianceMatrix` are ordered as

$$\begin{bmatrix}
Σ₁₁ \\
Σ₂₁ \\
Σ₃₁ \\
Σ₂₂ \\
Σ₃₂ \\
Σ₃₃ \\
\end{bmatrix},$$

which means that the output can easily be transformed into the implied covariance matrices using [`vectotril`](/API/miscellaneous#NeuralEstimators.vectotril) and `Symmetric`.

See also [`CorrelationMatrix`](/API/architectures#NeuralEstimators.CorrelationMatrix).

**Examples**

```julia
using NeuralEstimators, Flux, LinearAlgebra

d = 4
l = CovarianceMatrix(d)
p = d*(d+1)÷2
θ = randn(p, 50)

# Returns a matrix of parameters, which can be converted to covariance matrices
Σ = l(θ)
Σ = [Symmetric(cpu(vectotril(x)), :L) for x ∈ eachcol(Σ)]

# Obtain the Cholesky factor directly
L = l(θ, true)
L = [LowerTriangular(cpu(vectotril(x))) for x ∈ eachcol(L)]
L[1] * L[1]'
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Architectures.jl#L340-L402" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Miscellaneous {#Miscellaneous}
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.IndicatorWeights' href='#NeuralEstimators.IndicatorWeights'><span class="jlbinding">NeuralEstimators.IndicatorWeights</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
IndicatorWeights(h_max, n_bins::Integer)
(w::IndicatorWeights)(h::Matrix)
```


For spatial locations $\boldsymbol{s}$ and  $\boldsymbol{u}$, creates a spatial weight function defined as

$$\boldsymbol{w}(\boldsymbol{s}, \boldsymbol{u}) \equiv (\mathbb{I}(h \in B_k) : k = 1, \dots, K)',$$

where $\mathbb{I}(\cdot)$ denotes the indicator function,  $h \equiv \|\boldsymbol{s} - \boldsymbol{u} \|$ is the spatial distance between $\boldsymbol{s}$ and  $\boldsymbol{u}$, and $\{B_k : k = 1, \dots, K\}$ is a set of $K =$`n_bins` equally-sized distance bins covering the spatial distances between 0 and `h_max`. 

**Examples**

```julia
using NeuralEstimators, GraphNeuralNetworks

h_max = 1
n_bins = 10
w = IndicatorWeights(h_max, n_bins)
h = rand(1, 30) # distances between 30 pairs of spatial locations 
w(h)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Graphs.jl#L243-L266" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.KernelWeights' href='#NeuralEstimators.KernelWeights'><span class="jlbinding">NeuralEstimators.KernelWeights</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
KernelWeights(h_max, n_bins::Integer)
(w::KernelWeights)(h::Matrix)
```


For spatial locations $\boldsymbol{s}$ and  $\boldsymbol{u}$, creates a spatial weight function defined as

$$\boldsymbol{w}(\boldsymbol{s}, \boldsymbol{u}) \equiv (\exp(-(h - \mu_k)^2 / (2\sigma_k^2)) : k = 1, \dots, K)',$$

where $h \equiv \|\boldsymbol{s} - \boldsymbol{u}\|$ is the spatial distance between $\boldsymbol{s}$ and $\boldsymbol{u}$, and ${\mu_k : k = 1, \dots, K}$ and ${\sigma_k : k = 1, \dots, K}$ are the means and standard deviations of the Gaussian kernels for each bin, covering the spatial distances between 0 and h_max.

**Examples**

```julia
using NeuralEstimators, GraphNeuralNetworks

h_max = 1
n_bins = 10
w = KernelWeights(h_max, n_bins)
h = rand(1, 30) # distances between 30 pairs of spatial locations 
w(h)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Graphs.jl#L285-L306" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.PowerDifference' href='#NeuralEstimators.PowerDifference'><span class="jlbinding">NeuralEstimators.PowerDifference</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
PowerDifference(a, b)
```


Function $f(x, y) = |ax - (1-a)y|^b$ for trainable parameters a ∈ [0, 1] and b &gt; 0.

**Examples**

```julia
using NeuralEstimators, Flux

# Generate some data
d = 5
K = 10000
X = randn32(d, K)
Y = randn32(d, K)
XY = (X, Y)
a = 0.2f0
b = 1.3f0
Z = (abs.(a .* X - (1 .- a) .* Y)).^b

# Initialise layer
f = PowerDifference([0.5f0], [2.0f0])

# Optimise the layer
loader = Flux.DataLoader((XY, Z), batchsize=32, shuffle=false)
optim = Flux.setup(Flux.Adam(0.01), f)
for epoch in 1:100
    for (xy, z) in loader
        loss, grads = Flux.withgradient(f) do m
            Flux.mae(m(xy), z)
        end
        Flux.update!(optim, f, grads[1])
    end
end

# Estimates of a and b
f.a
f.b
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Architectures.jl#L646-L683" target="_blank" rel="noreferrer">source</a></Badge>

</details>

