# Examples

We first load the required packages, the following of which are used throughout these examples:

```
using NeuralEstimators
using Flux                 # Julia's deep-learning library
using Distributions        # sampling from probability distributions
using AlgebraOfGraphics    # visualisation
using CairoMakie           # visualisation
```

The following packages will be used in the examples with [Gridded data](@ref) and [Irregular spatial data](@ref):  

```
using Distances            # computing distance matrices 
using Folds                # parallel simulation (start Julia with --threads=auto)
using LinearAlgebra        # Cholesky factorisation
```

The following packages are used only in the example with [Irregular spatial data](@ref): 

```
using GraphNeuralNetworks  # GNN architecture
using Statistics           # mean()
```

Finally, various GPU backends can be used (see the [Flux documentation](https://fluxml.ai/Flux.jl/stable/guide/gpu/#GPU-Support) for details). For instance, to use an NVIDIA GPU in the following examples, simply the load the `CUDA.jl` package:  

```
using CUDA
```

Once a GPU package is loaded and an appropriate GPU is available, the functions in `NeuralEstimators` will automatically leverage the GPU to improve computational efficiency, while maintaining memory safety through the use of batched operations.

## Univariate data

Here, we develop a neural Bayes estimator  for $\boldsymbol{\theta} \equiv (\mu, \sigma)'$ from data $Z_1, \dots, Z_m$ that are independent and identically distributed realisations from the distribution $N(\mu, \sigma^2)$. (See [Estimators](@ref) for a list of other classes of estimators available in the package.)

We begin by defining a function to sample parameters from the prior distribution. Assuming prior independence, we adopt the marginal priors $\mu \sim N(0, 1)$ and $\sigma \sim IG(3, 1)$:

```
function sample(K)
	μ = rand(Normal(0, 1), K)
	σ = rand(InverseGamma(3, 1), K)
	θ = vcat(μ', σ')
	return θ
end
```

Next, we define the statistical model implicitly through data simulation. The simulated data are stored as a `Vector{A}`, where each element corresponds to one parameter vector. The type `A` reflects the multivariate structure of the data. In this example, each replicate $Z_1, \dots, Z_m$ is univariate, so `A` is a Matrix with $n = 1$ row and $m$ columns:

```
function simulate(θ, m)
    [ϑ[1] .+ ϑ[2] .* randn(1, m) for ϑ in eachcol(θ)]
end
```

We now design our neural network. 

As we are constructing a neural Bayes estimator, the neural network is a mapping $\mathcal{Z}\to\Theta$, and the dimensionality of the neural-network output is therefore $d \equiv \rm{dim}(\Theta) = 2$. 

Since our data are replicated, we adopt the DeepSets framework, implemented via the type [`DeepSet`](@ref). DeepSets consist of two neural networks: an inner network and an outer network. The inner network extracts summary statistics from the data, and its architecture depends on the multivariate structure of the data. For unstructured data (i.e., no spatial or temporal correlation within each replicate), we use a multilayer perceptron (MLP) with an input dimension equal to the dimensionality of each data replicate. The outer network maps the learned summary statistics to the output space (here, the parameter space, $\Theta$). The outer network is always an MLP. 

Below is an example of a DeepSets architecture for neural Bayes estimation in this example. Note that many models have parameter constraints (e.g., variance and range parameters that must be strictly positive). These constraints can be incorporated in the final layer of the neural network by choosing appropriate activation functions for each parameter. Here, we enforce the constraint $\sigma > 0$ by applying the softplus activation function in the final layer of the outer network, ensuring that all parameter estimates are valid:

```
# Hidden layer width
w = 128  

# Final layer has output dimension d = dim(Θ) = 2 and enforces parameter constraints
final_layer = Parallel(
    vcat,
    Dense(w, 1, identity),     # μ ∈ ℝ
    Dense(w, 1, softplus)      # σ > 0
)

# Inner and outer networks
ψ = Chain(Dense(1, w, relu), Dense(w, w, relu))    
ϕ = Chain(Dense(w, w, relu), final_layer)          

# Combine into a DeepSet
network = DeepSet(ψ, ϕ)
```



We then initialise the neural Bayes estimator by wrapping the neural network in a [`PointEstimator`](@ref): 

```
θ̂ = PointEstimator(network)
```

Next, we train the estimator using [`train()`](@ref), here using the default absolute-error loss. We'll train the estimator using 50 independent replicates per parameter configuration. Below, we pass our user-defined functions for sampling parameters and simulating data, but one may also pass parameter or data instances, which will be held fixed during training:

```
m = 50
θ̂ = train(θ̂, sample, simulate, m = m)
```

One may wish to save a trained estimator and load it in a later session: see [Saving and loading neural estimators](@ref) for details on how this can be done. 

The function [`assess()`](@ref) can be used to assess the trained estimator. Parametric and non-parametric bootstrap-based uncertainty quantification are facilitated by [`bootstrap()`](@ref) and [`interval()`](@ref), and this can also be included in the assessment stage through the keyword argument `boot`:

```
θ_test = sample(1000)
Z_test = simulate(θ_test, m)
assessment = assess(θ̂, θ_test, Z_test, boot = true)
```

The resulting [`Assessment`](@ref) object contains the sampled parameters, the corresponding point estimates, and the corresponding lower and upper bounds of the bootstrap intervals. This object can be used to compute various diagnostics:

```
bias(assessment)      # μ = 0.002, σ = 0.017
rmse(assessment)      # μ = 0.086, σ = 0.078
risk(assessment)      # μ = 0.055, σ = 0.056
plot(assessment)
```

![Univariate Gaussian example: Estimates vs. truth](../assets/figures/univariate.png)

As an alternative form of uncertainty quantification with neural Bayes estimators, one may approximate a set of marginal posterior quantiles by training a neural Bayes estimator under the quantile loss function, which allows one to generate approximate marginal posterior credible intervals. This is facilitated with [`IntervalEstimator`](@ref) which, by default, targets 95% central credible intervals:

```
q̂ = IntervalEstimator(network)
q̂ = train(q̂, sample, simulate, m = m)
```


The resulting posterior credible-interval estimator can also be assessed using [`assess()`](@ref). Often, these intervals have better coverage than bootstrap-based intervals.

Once an estimator is deemed to be well calibrated, it may be applied to observed data (below, we use simulated data as a substitute for observed data):

```
θ = sample(1)               # true parameters
Z = simulate(θ, m)          # "observed" data
estimate(θ̂, Z)              # point estimates
interval(bootstrap(θ̂, Z))   # 95% non-parametric bootstrap intervals
interval(q̂, Z)              # 95% marginal posterior credible intervals
```

## Unstructured multivariate data

Suppose now that each data set consists of $m$ replicates $\boldsymbol{Z}_1, \dots, \boldsymbol{Z}_m$ of an $n$-dimensional multivariate distribution. Everything remains as given in the univariate example above, except that we now store each data set as a $n \times m$ matrix (previously they were stored as $1\times m$ matrices), and the inner network of the DeepSets representation takes a $n$-dimensional input (previously it took a 1-dimensional input).

Note that, when estimating a full covariance matrix, one may wish to constrain the neural estimator to only produce parameters that imply a valid (i.e., positive definite) covariance matrix. This can be achieved by appending a  [`CovarianceMatrix`](@ref) layer to the end of the outer network of the DeepSets representation. However, the estimator will often learn to provide valid estimates, even if not constrained to do so.


## Gridded data

For data collected over a regular grid, neural estimators are typically based on a convolutional neural network (CNN; see, e.g., [Dumoulin and Visin, 2016](https://arxiv.org/abs/1603.07285)). 

When using CNNs with `NeuralEstimators`, each data set must be stored as a multi-dimensional array. The penultimate dimension stores the so-called "channels" (this dimension is singleton for univariate processes, two for bivariate processes, etc.), while the final dimension stores independent replicates. For example, to store $50$ independent replicates of a bivariate spatial process measured over a $10\times15$ grid, one would construct an array of dimension $10\times15\times2\times50$.

 For illustration, here we develop a neural Bayes estimator for the spatial Gaussian process model with exponential covariance function and unknown range parameter $\theta$. The spatial domain is taken to be the unit square, and we adopt the prior $\theta \sim U(0.05, 0.5)$. 
 
 Simulation from Gaussian processes typically involves the computation of an expensive intermediate object, namely, the Cholesky factor of a covariance matrix. Storing intermediate objects can enable the fast simulation of new data sets when the parameters are held fixed. Hence, in this example, we define a custom type `Parameters` subtyping [`ParameterConfigurations`](@ref) for storing the matrix of parameters and the corresponding Cholesky factors: 

```
struct Parameters{T} <: ParameterConfigurations
	θ::Matrix{T}
	L
end
```
 
 Further, we define two constructors for our custom type: one that accepts an integer $K$, and another that accepts a $d\times K$ matrix of parameters. The former constructor will be useful during the training stage for sampling from the prior distribution, while the latter constructor will be useful for parametric bootstrap (since this involves repeated simulation from the fitted model):

```
function sample(K::Integer)

	# Sample parameters from the prior 
	θ = rand(Uniform(0.05, 0.5), 1, K)

	# Pass to matrix constructor
	Parameters(θ)
end

function Parameters(θ::Matrix)

	# Spatial locations, a 16x16 grid over the unit square
	pts = range(0, 1, length = 16)
	S = expandgrid(pts, pts)

	# Distance matrix, covariance matrices, and Cholesky factors
	D = pairwise(Euclidean(), S, dims = 1)
	K = size(θ, 2)
	L = Folds.map(1:K) do k
		Σ = exp.(-D ./ θ[k])
		cholesky(Symmetric(Σ)).L
	end

	Parameters(θ, L)
end
```

Next, we define the model simulator: 

```
function simulate(parameters::Parameters, m = 1) 
	Z = Folds.map(parameters.L) do L
		n = size(L, 1)
		z = L * randn(n, m)
		z = reshape(z, 16, 16, 1, m) # reshape to 16x16 images
		z
	end
	Z
end
```

A possible architecture is as follows:

```
ψ = Chain(
	Conv((3, 3), 1 => 32, relu),
	MaxPool((2, 2)),
	Conv((3, 3),  32 => 64, relu),
	MaxPool((2, 2)),
	Flux.flatten
	)
ϕ = Chain(Dense(256, 64, relu), Dense(64, 1))
network = DeepSet(ψ, ϕ)
```

Next, we initialise a point estimator and a posterior credible-interval estimator:

```
θ̂ = PointEstimator(network)
q̂ = IntervalEstimator(network)
```

Now we train the estimators, here using fixed parameter instances to avoid repeated Cholesky factorisations (see [Storing expensive intermediate objects for data simulation](@ref) and [On-the-fly and just-in-time simulation](@ref) for further discussion):

```
K = 10000  # number of training parameter vectors
m = 1      # number of independent replicates in each data set
θ_train = sample(K)
θ_val = sample(K ÷ 10)
θ̂ = train(θ̂, θ_train, θ_val, simulate, m = m)
q̂ = train(q̂, θ_train, θ_val, simulate, m = m)
```

Once the estimators have been trained, we assess them using empirical simulation-based methods:

```
θ_test = sample(1000)
Z_test = simulate(θ_test)
assessment = assess([θ̂, q̂], θ_test, Z_test)

bias(assessment)       # 0.005
rmse(assessment)       # 0.032
coverage(assessment)   # 0.953
plot(assessment)       
```

![Gridded spatial Gaussian process example: Estimates vs. truth](../assets/figures/gridded.png)

Finally, we can apply our estimators to observed data. Note that when we have a single replicate only (which is often the case in spatial statistics), non-parametric bootstrap is not possible, and we instead use parametric bootstrap:

```
θ = sample(1)                          # true parameter
Z = simulate(θ)                        # "observed" data
θ̂(Z)                                   # point estimates
interval(q̂, Z)                         # 95% marginal posterior credible intervals
bs = bootstrap(θ̂, θ̂(Z), simulate, m)   # parametric bootstrap intervals
interval(bs)                           # 95% parametric bootstrap intervals
```

## Irregular spatial data

To cater for spatial data collected over arbitrary spatial locations, one may construct a neural estimator with a graph neural network (GNN; see [Sainsbury-Dale, Zammit-Mangion, Richards, and Huser, 2025](https://doi.org/10.1080/10618600.2024.2433671)). The overall workflow remains as given in previous examples, with two key additional steps:

- Sampling spatial configurations during the training phase, possibly using an appropriately chosen spatial point process: see, for example, [`maternclusterprocess`](@ref).
- Storing the spatial data as a graph: see [`spatialgraph`](@ref).

For illustration, we again consider a spatial Gaussian process model with exponential covariance function, and we define a struct for storing expensive intermediate objects needed for data simulation. In this example, these objects include Cholesky factors, and spatial graphs which store the adjacency matrices needed to perform graph convolution: 


```
struct Parameters{T} <: ParameterConfigurations
	θ::Matrix{T}   # true parameters  
	L              # Cholesky factors
	g              # spatial graphs
	S              # spatial locations 
end
```

Again, we define two constructors, which will be convenient for sampling parameters from the prior during training and assessment, and for parametric bootstrap sampling when making inferences from observed data:

```
function sample(K::Integer)

	# Sample parameters from the prior 
	θ = rand(Uniform(0.05, 0.5), 1, K)

	# Simulate spatial configurations over the unit square
	n = rand(200:300, K)
	λ = rand(Uniform(10, 50), K)
	S = [maternclusterprocess(λ = λ[k], μ = n[k]/λ[k]) for k ∈ 1:K]

	# Pass to constructor
	Parameters(θ, S)
end

function Parameters(θ::Matrix, S)

	# Number of parameter vectors
	K = size(θ, 2)

	# Distance matrices, covariance matrices, and Cholesky factors
	D = pairwise.(Ref(Euclidean()), S, dims = 1)
	L = Folds.map(1:K) do k
		Σ = exp.(-D[k] ./ θ[k])
		cholesky(Symmetric(Σ)).L
	end

	# Construct spatial graphs
	g = spatialgraph.(S)

	Parameters(θ, L, g, S)
end
```

Next, we define a function for simulating from the model given an object of type `Parameters`. The code below enables simulation of an arbitrary number of independent replicates `m`, and one may provide a single integer for `m`, or any object that can be sampled using `rand(m, K)` (e.g., an integer range or some distribution over the possible sample sizes):

```
function simulate(parameters::Parameters, m)
	K = size(parameters, 2)
	m = rand(m, K)
	map(1:K) do k
		L = parameters.L[k]
		g = parameters.g[k]
		n = size(L, 1)
		Z = L * randn(n, m[k])      
		spatialgraph(g, Z)            
	end
end
simulate(parameters::Parameters, m::Integer = 1) = simulate(parameters, range(m, m))
```

Next, we construct our GNN architectur. Here, we use an architecture tailored to isotropic spatial dependence models; for further details, see Section 2.2 of [Sainsbury-Dale et al. (2025)](https://doi.org/10.1080/10618600.2024.2433671). In this example our goal is to construct a point estimator, however any other kind of estimator (see [Estimators](@ref)) can be constructed by simply substituting the appropriate estimator class in the final line below:

```
# Spatial weight functions: continuous surrogates for 0-1 basis functions 
h_max = 0.15 # maximum distance to consider 
q = 10       # output dimension of the spatial weights
w = KernelWeights(h_max, q)

# Propagation module
propagation = GNNChain(
	SpatialGraphConv(1 => q, relu, w = w, w_out = q),
	SpatialGraphConv(q => q, relu, w = w, w_out = q)
)

# Readout module
readout = GlobalPool(mean)

# Summary network
ψ = GNNSummary(propagation, readout)

# Expert summary statistics, the empirical variogram
S = NeighbourhoodVariogram(h_max, q)

# Inference network
ϕ = Chain(
	Dense(2q => 128, relu), 
	Dense(128 => 128, relu), 
	Dense(128 => 1, identity)
)

# DeepSet object
deepset = DeepSet(ψ, ϕ; S = S)

# Point estimator
θ̂ = PointEstimator(deepset)
```

Next, we train the estimator. 

```
m = 1
K = 5000
θ_train = sample(K)
θ_val   = sample(K÷5)
θ̂ = train(θ̂, θ_train, θ_val, simulate, m = m, epochs = 20)
``` 

Note that the computations in GNNs are performed in parallel, making them particularly well-suited for GPUs, which typically contain thousands of cores. If you have access to an NVIDIA GPU, you can utilise it by simply loading the Julia package `CUDA`. 

Next, we assess our trained estimator: 

```
θ_test = sample(1000)
Z_test = simulate(θ_test, m)
assessment = assess(θ̂, θ_test, Z_test)
bias(assessment)   
rmse(assessment)    
risk(assessment)   
plot(assessment)   
```

![Estimates from a graph neural network (GNN) based neural Bayes estimator](../assets/figures/spatial.png)

Finally, once the estimator has been assessed, it may be applied to observed data, with bootstrap-based uncertainty quantification facilitated by [`bootstrap`](@ref) and [`interval`](@ref). Below, we use simulated data as a substitute for observed data:

```
parameters = sample(1)             # sample a single parameter vector
Z = simulate(parameters)           # simulate data                  
θ = parameters.θ                   # true parameters used to generate data
S = parameters.S                   # observed locations
θ̂(Z)                               # point estimates
θ̃ = Parameters(θ̂(Z), S)            # construct Parameters object from the point estimates
bs = bootstrap(θ̂, θ̃, simulate, m)  # bootstrap estimates
interval(bs)                       # parametric bootstrap confidence interval              
```
