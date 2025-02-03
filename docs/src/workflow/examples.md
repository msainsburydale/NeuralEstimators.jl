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
using Distances            # distance matrices 
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

As we are constructing a neural Bayes estimator, the neural network is a mapping $\mathcal{Z}\to\Theta$, and the dimensionality of the neural-network output is therefore $d \equiv \textrm{dim}(\Theta) = 2$. 

Since our data are replicated, we adopt the DeepSets framework, implemented via the type [`DeepSet`](@ref). DeepSets consist of two neural networks: an inner network and an outer network. The inner network extracts summary statistics from the data, and its architecture depends on the multivariate structure of the data. For unstructured data (i.e., data without spatial or temporal correlation within each replicate), we use a multilayer perceptron (MLP). The input dimension matches the dimensionality of each data replicate, while the output dimension corresponds to the number of summary statistics appropriate for the model (a common choice is $d$). The outer network maps the learned summary statistics to the output space (here, the parameter space, $\Theta$). The outer network is always an MLP. 

Below is an example of a DeepSets architecture for neural Bayes estimation in this example. Note that many models have parameter constraints (e.g., variance and range parameters that must be strictly positive). These constraints can be incorporated in the final layer of the neural network by choosing appropriate activation functions for each parameter. Here, we enforce the constraint $\sigma > 0$ by applying the softplus activation function in the final layer of the outer network, ensuring that all parameter estimates are valid. For some additional ways to constrain parameter estimates, see [Output layers](@ref). 

```
n = 1    # dimension of each data replicate (univariate)
d = 2    # dimension of the parameter vector θ
w = 128  # width of each hidden layer 

# Final layer has output dimension d and enforces parameter constraints
final_layer = Parallel(
    vcat,
    Dense(w, 1, identity),     # μ ∈ ℝ
    Dense(w, 1, softplus)      # σ > 0
)

# Inner and outer networks
ψ = Chain(Dense(n, w, relu), Dense(w, d, relu))    
ϕ = Chain(Dense(d, w, relu), final_layer)          

# Combine into a DeepSet
network = DeepSet(ψ, ϕ)
```

We then initialise the neural Bayes estimator by wrapping the neural network in a [`PointEstimator`](@ref): 

```
estimator = PointEstimator(network)
```

Next, we train the estimator using [`train()`](@ref), here using the default absolute-error loss. We'll train the estimator using 50 independent replicates per parameter configuration. Below, we pass our user-defined functions for sampling parameters and simulating data, but one may also pass parameter or data instances, which will be held fixed during training:

```
m = 50
estimator = train(estimator, sample, simulate, m = m)
```

One may wish to save a trained estimator and load it in a later session: see [Saving and loading neural estimators](@ref) for details on how this can be done. 

The function [`assess()`](@ref) can be used to assess the trained estimator. Parametric and non-parametric bootstrap estimates can be obtained via [`bootstrap()`](@ref), with corresponding confidence intervals computed using [`interval()`](@ref). Additionally, bootstrap-based uncertainty quantification can be included in the assessment stage through the keyword argument `boot`:

```
θ_test = sample(1000)
Z_test = simulate(θ_test, m)
assessment = assess(estimator, θ_test, Z_test, boot = true)
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
θ = sample(1)                       # true parameters
Z = simulate(θ, m)                  # "observed" data
estimate(estimator, Z)              # point estimate
interval(bootstrap(estimator, Z))   # 95% non-parametric bootstrap intervals
interval(q̂, Z)                      # 95% marginal posterior credible intervals
```

## Gridded data

For data collected over a regular grid, neural estimators are typically based on a convolutional neural network (CNN; see, e.g., [Dumoulin and Visin, 2016](https://arxiv.org/abs/1603.07285)). 

When using CNNs with `NeuralEstimators`, each data set must be stored as a multi-dimensional array. The penultimate dimension stores the so-called "channels" (this dimension is singleton for univariate processes, two for bivariate processes), while the final dimension stores independent replicates. For example, to store $50$ independent replicates of a bivariate spatial process measured over a $10\times15$ grid, one would construct an array of dimension $10\times15\times2\times50$.

 For illustration, here we develop a neural Bayes estimator for the (univariate) spatial Gaussian process model with exponential covariance function and unknown range parameter $\theta > 0$. The spatial domain is taken to be the unit square, and we adopt the prior $\theta \sim U(0.05, 0.5)$. 
 
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

A possible architecture is as follows. Note that deeper architectures that employ residual connections (see [ResidualBlock](@ref)) often lead to improved performance, and certain pooling layers (e.g., [GlobalMeanPool](https://fluxml.ai/Flux.jl/stable/reference/models/layers/#Flux.GlobalMeanPool)) allow the neural network to accommodate grids of varying dimension; for further discussion and an illustration, see [Sainsbury-Dale et al. (2025, Sec. S3, S4)](https://doi.org/10.48550/arXiv.2501.04330). 

```
# Inner network 
ψ = Chain(
      Conv((3, 3), 1 => 32, relu),   # 3x3 convolutional filter, 1 input channel to 32 output channels
      MaxPool((2, 2)),               # 2x2 max pooling for dimension reduction
      Conv((3, 3), 32 => 64, relu),  # 3x3 convolutional filter, 32 input channels to 64 output channels
      MaxPool((2, 2)),               # 2x2 max pooling for dimension reduction
      Flux.flatten                   # flatten output to feed into a fully connected layer
  )

# Outer network 
ϕ = Chain(Dense(256, 64, relu), Dense(64, 1))

# DeepSet object
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

Finally, we can apply our estimators to observed data:

```
θ = sample(1)                          # true parameter
Z = simulate(θ)                        # "observed" data
estimate(θ̂, Z)                         # point estimate
interval(q̂, Z)                         # 95% marginal posterior credible intervals
```

Note that missing data (e.g., due to cloud cover) can be accommodated using the [missing-data methods](@ref "Missing data") implemented in the package.

## Irregular spatial data

To cater for spatial data collected over arbitrary spatial locations, one may construct a neural estimator with a graph neural network (GNN; see [Sainsbury-Dale, Zammit-Mangion, Richards, and Huser, 2025](https://doi.org/10.1080/10618600.2024.2433671)). The overall workflow remains as given in previous examples, with two key additional steps:

- Sampling spatial configurations during the training phase, possibly using an appropriately chosen spatial point process; see, for example, [`maternclusterprocess()`](@ref).
- Storing the spatial data as a graph; see [`spatialgraph()`](@ref).

For illustration, we again consider a spatial Gaussian process model with exponential covariance function, and we define a type for storing expensive intermediate objects needed for data simulation. In this example, these objects include Cholesky factors, and spatial graphs which store the adjacency matrices needed to perform graph convolution: 

```
struct Parameters <: ParameterConfigurations
	θ::Matrix      # true parameters  
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

	# Sample spatial configurations from Matern cluster process on [0, 1]²
	n = rand(200:300, K)
	λ = rand(Uniform(10, 50), K)
	S = [maternclusterprocess(λ = λ[k], μ = n[k]/λ[k]) for k ∈ 1:K]

	# Pass to constructor
	Parameters(θ, S)
end

function Parameters(θ::Matrix, S)
	# Compute covariance matrices and Cholesky factors 
	L = Folds.map(axes(θ, 2)) do k
		D = pairwise(Euclidean(), S[k], dims = 1)
		Σ = Symmetric(exp.(-D ./ θ[k]))
		cholesky(Σ).L
	end

	# Construct spatial graphs
	g = spatialgraph.(S)

	# Store in Parameters object
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

Next, we construct our GNN architecture. Here, we use an architecture tailored to isotropic spatial dependence models; for further details, see [Sainsbury-Dale et al. (2025, Sec. 2.2)](https://doi.org/10.1080/10618600.2024.2433671). We also employ a sparse approximation of the empirical variogram as an expert summary statistic ([Gerber and Nychka, 2021](https://onlinelibrary.wiley.com/doi/abs/10.1002/sta4.382)).

In this example our goal is to construct a point estimator, however any other kind of estimator (see [Estimators](@ref)) can be constructed by simply substituting the appropriate estimator class in the final line below:

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

# Inner network
ψ = GNNSummary(propagation, readout)

# Expert summary statistics, the empirical variogram
S = NeighbourhoodVariogram(h_max, q)

# Outer network
ϕ = Chain(
	Dense(2q => 128, relu), 
	Dense(128 => 128, relu), 
	Dense(128 => 1, identity)
)

# DeepSet object
deepset = DeepSet(ψ, ϕ; S = S)

# Point estimator
estimator = PointEstimator(deepset)
```

Next, we train the estimator. 

```
m = 1
K = 5000
θ_train = sample(K)
θ_val   = sample(K÷5)
estimator = train(estimator, θ_train, θ_val, simulate, m = m, epochs = 20)
``` 

Note that the computations in GNNs are performed in parallel, making them particularly well-suited for GPUs, which typically contain thousands of cores. If you have access to an NVIDIA GPU, you can utilise it by simply loading the Julia package `CUDA`. 

Next, we assess our trained estimator: 

```
θ_test = sample(1000)
Z_test = simulate(θ_test, m)
assessment = assess(estimator, θ_test, Z_test)
bias(assessment)   
rmse(assessment)    
risk(assessment)   
plot(assessment)   
```

![Estimates from a graph neural network (GNN) based neural Bayes estimator](../assets/figures/spatial.png)

Finally, once the estimator has been assessed, it may be applied to observed data, with bootstrap-based uncertainty quantification facilitated by [`bootstrap`](@ref) and [`interval`](@ref). Below, we use simulated data as a substitute for observed data:

```
parameters = sample(1)             # sample a parameter vector and spatial locations              
θ = parameters.θ                   # true parameters
S = parameters.S                   # "observed" locations
Z = simulate(parameters)           # "observed" data    
θ̂ = estimate(estimator, Z)         # point estimate
ps = Parameters(θ̂, S)              # construct Parameters object from point estimate
bs = bootstrap(estimator, ps, simulate, m)  # parametric bootstrap estimates
interval(bs)                       # parametric bootstrap confidence interval              
```
