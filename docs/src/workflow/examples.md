# Examples

The following packages are used throughout these examples:

```
using NeuralEstimators, Flux, GraphNeuralNetworks, Distances, Distributions, Folds, LinearAlgebra, Statistics
```

## Univariate data

Here we develop a neural Bayes estimator for $\boldsymbol{\theta} \equiv (\mu, \sigma)'$ from data $Z_1, \dots, Z_m$ that are independent and identically distributed realisations from the distribution $N(\mu, \sigma^2)$. 

First, we define a function to sample parameters from the prior distribution. Here, we assume that the parameters are independent a priori and we adopt the marginal priors $\mu \sim N(0, 1)$ and $\sigma \sim IG(3, 1)$. The sampled parameters are stored as $p \times K$ matrices, with $p$ the number of parameters in the model and $K$ the number of sampled parameter vectors:

```
function sample(K)
	μ = rand(Normal(0, 1), 1, K)
	σ = rand(InverseGamma(3, 1), 1, K)
	θ = vcat(μ, σ)
	return θ
end
```

Next, we implicitly define the statistical model through data simulation. In this package, the data are always stored as a `Vector{A}`, where each element of the vector is associated with one parameter vector, and where the type `A` depends on the multivariate structure of the data. Since in this example each replicate $Z_1, \dots, Z_m$ is univariate, `A` should be a `Matrix` with $d=1$ row and $m$ columns. Below, we define our simulator given a single parameter vector, and given a matrix of parameter vectors (which simply applies the simulator to each column):

```
simulate(θ, m) = θ[1] .+ θ[2] .* randn(1, m)
simulate(θ::AbstractMatrix, m) = simulate.(eachcol(θ), m)
```

We now design our neural-network architecture. The workhorse of the package is the [`DeepSet`](@ref) architecture, which provides an elegant framework for making inference with an arbitrary number of independent replicates and for incorporating both neural and user-defined statistics. The DeepSets framework consists of two neural networks, a summary network and an inference network. The inference network (also known as the outer network) is always a multilayer perceptron (MLP). However, the architecture of the summary network (also known as the inner network) depends on the multivariate structure of the data. With unstructured data (i.e., when there is no spatial or temporal correlation within a replicate), we use an MLP with input dimension equal to the dimension of each replicate of the statistical model (i.e., one for univariate data): 

```
p = 2                                                # number of parameters 
ψ = Chain(Dense(1, 32, relu), Dense(32, 32, relu))   # summary network
ϕ = Chain(Dense(32, 32, relu), Dense(32, p))         # inference network
architecture = DeepSet(ψ, ϕ)
```

In this example, we wish to construct a point estimator for the unknown parameter vector, and we therefore initialise a [`PointEstimator`](@ref) object based on our chosen architecture (see [Estimators](@ref) for a list of other estimators available in the package): 

```
θ̂ = PointEstimator(architecture)
```

Next, we train the estimator using [`train()`](@ref), here using the default absolute-error loss. We'll train the estimator using 50 independent replicates per parameter configuration. Below, we pass our user-defined functions for sampling parameters and simulating data, but one may also pass parameter or data instances, which will be held fixed during training:

```
m = 50
θ̂ = train(θ̂, sample, simulate, m = m)
```

To fully exploit the amortised nature of neural estimators, one may wish to save a trained estimator and load it in later sessions: see [Saving and loading neural estimators](@ref) for details on how this can be done. 

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

As an alternative form of uncertainty quantification, one may approximate a set of marginal posterior quantiles by training a second estimator under the quantile loss function, which allows one to generate approximate marginal posterior credible intervals. This is facilitated with [`IntervalEstimator`](@ref) which, by default, targets 95% central credible intervals:

```
q̂ = IntervalEstimator(architecture)
q̂ = train(q̂, sample, simulate, m = m)
```


The resulting posterior credible-interval estimator can also be assessed with empirical simulation-based methods using [`assess()`](@ref), as we did above for the point estimator. Often, these intervals have better coverage than bootstrap-based intervals.

Once an estimator is deemed to be satisfactorily calibrated, it may be applied to observed data (below, we use simulated data as a substitute for observed data):

```
θ = sample(1)               # true parameters
Z = simulate(θ, m)          # "observed" data
θ̂(Z)                        # point estimates
interval(bootstrap(θ̂, Z))   # 95% non-parametric bootstrap intervals
interval(q̂, Z)              # 95% marginal posterior credible intervals
```

To utilise a GPU for improved computational efficiency, one may simply move the estimator and the data to the GPU through the calls `θ̂ = gpu(θ̂)` and `Z = gpu(Z)` before applying the estimator. Note that GPUs often have limited memory relative to CPUs, and this can sometimes lead to memory issues when working with very large data sets: in these cases, the function [`estimateinbatches()`](@ref) can be used to apply the estimator over batches of data to circumvent any memory concerns. 

## Unstructured multivariate data

Suppose now that each data set now consists of $m$ replicates $\mathbf{Z}_1, \dots, \mathbf{Z}_m$ of a $d$-dimensional multivariate distribution. Everything remains as given in the univariate example above, except that we now store each data set as a $d \times m$ matrix (previously they were stored as $1\times m$ matrices), and the summary network of the DeepSets representation takes a $d$-dimensional input (previously it took a 1-dimensional input).

Note that, when estimating a full covariance matrix, one may wish to constrain the neural estimator to only produce parameters that imply a valid (i.e., positive definite) covariance matrix. This can be achieved by appending a  [`CovarianceMatrix`](@ref) layer to the end of the outer network of the DeepSets representation. However, the estimator will often learn to provide valid estimates, even if not constrained to do so.


## Gridded data


For data collected over a regular grid, neural estimators are typically based on a convolutional neural network (CNN; see, for example, [Dumoulin and Visin, 2016](https://arxiv.org/abs/1603.07285)).

In these settings, each data set must be stored as a ($D + 2$)-dimensional array, where $D$ is the dimension of the grid (e.g., $D = 1$ for time series, $D = 2$ for two-dimensional spatial grids, etc.). The first $D$ dimensions of the array correspond to the dimensions of the grid; the penultimate dimension stores the so-called "channels" (this dimension is singleton for univariate processes, two for bivariate processes, etc.); and the final dimension stores the independent replicates. For example, to store 50 independent replicates of a bivariate spatial process measured over a 10x15 grid, one would construct an array of dimension 10x15x2x50.

Here, we develop a neural Bayes estimator for the spatial Gaussian process model with exponential covariance function and unknown range parameter. 

 For illustration, we develop a neural Bayes estimator for the spatial Gaussian-process model,
 ```math
 Z_{j} = Y(\boldsymbol{s}_{j}) + \epsilon_{j}, \quad j = 1, \dots, n,
 ```
 where $\boldsymbol{Z} \equiv (Z_{1}, \dots, Z_{n})'$ are data collected at locations $\{\boldsymbol{s}_{1}, \dots, \boldsymbol{s}_{n}\}$ in a spatial domain $\mathcal{D}$, $Y(\cdot)$ is a spatially-correlated mean-zero Gaussian process, and $\epsilon_j \sim N(0, \tau^2)$, $j = 1, \dots, n$ is Gaussian white noise with standard deviation $\tau > 0$. Here, we use the popular isotropic Matérn covariance function,
 ```math
 \text{cov}\big(Y(\boldsymbol{s}), Y(\boldsymbol{u})\big)
 =
 \sigma^2 \frac{2^{1 - \nu}}{\Gamma(\nu)} \Big(\frac{\|\boldsymbol{s} - \boldsymbol{u}\|}{\rho}\Big)^\nu K_\nu\Big(\frac{\|\boldsymbol{s} - \boldsymbol{u}\|}{\rho}\Big),
 \quad \boldsymbol{s}, \boldsymbol{u} \in \mathcal{D},
 ```
 where $\sigma^2$ is the marginal variance, $\Gamma(\cdot)$ is the gamma function, $K_\nu(\cdot)$ is the Bessel function of the second kind of order $\nu$, and $\rho > 0$ and $\nu > 0$ are range and smoothness parameters, respectively. For ease of illustration, we fix $\sigma^2 = 1$ and $\nu = 1$, which leaves two unknown parameters that need to be estimated, $\boldsymbol{\theta} \equiv (\tau, \rho)'$.


The spatial domain is taken to be the unit square and we adopt the prior $\theta \sim U(0.05, 0.5)$. First, we define a function for sampling from the prior distribution:

```
sample(K) = rand(Uniform(0.05, 0.5), 1, K)
```

Below, we give example code for simulating from the statistical model, where the data is collected over a 16x16 grid:

```
function simulate(θ, m = 1)

	# Spatial locations
	pts = range(0, 1, length = 16)
	S = expandgrid(pts, pts)
	n = size(S, 1)

	# Distance matrix, covariance matrix, and Cholesky factor
	D = pairwise(Euclidean(), S, dims = 1)
	Σ = exp.(-D ./ θ)
	L = cholesky(Symmetric(Σ)).L

	# Spatial field
	Z = L * randn(n)

	# Reshape to 16x16 image
	Z = reshape(Z, 16, 16, 1, 1)

	return Z
end
simulate(θ::AbstractMatrix, m) = [simulate(x, m) for x ∈ eachcol(θ)]
```

For a 16x16 grid, one possible CNN architecture is as follows:

```
p = 1 # number of parameters in the statistical model

# Summary network
ψ = Chain(
	Conv((3, 3), 1 => 32, relu),
	MaxPool((2, 2)),
	Conv((3, 3),  32 => 64, relu),
	MaxPool((2, 2)),
	Flux.flatten
	)

# Inference network
ϕ = Chain(Dense(256, 64, relu), Dense(64, p))

# DeepSet
architecture = DeepSet(ψ, ϕ)
```

Next, we initialise a point estimator and a posterior credible-interval estimator:

```
θ̂ = PointEstimator(architecture)
q̂ = IntervalEstimator(architecture)
```

Now we train the estimators. Since simulation from this statistical model involves Cholesky factorisation, which is moderately expensive with $n=256$ spatial locations, here we used fixed parameter and data instances during training. See [Storing expensive intermediate objects for data simulation](@ref) for methods that allow one to avoid repeated Cholesky factorisation when performing [On-the-fly and just-in-time simulation](@ref):

```
K = 20000
θ_train = sample(K)
θ_val   = sample(K ÷ 10)
Z_train = simulate(θ_train)
Z_val   = simulate(θ_val)

θ̂  = train(θ̂,  θ_train, θ_val, Z_train, Z_val)
q̂ = train(q̂, θ_train, θ_val, Z_train, Z_val)
```

Once the estimators have been trained, we assess them using empirical simulation-based methods:

```
θ_test = sample(1000)
Z_test = simulate(θ_test)
assessment = assess([θ̂, q̂], θ_test, Z_test)
assessment = assess(θ̂, θ_test, Z_test)

bias(assessment)
rmse(assessment)
coverage(assessment)
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

To cater for spatial data collected over arbitrary spatial locations, one may construct a neural estimator with a graph neural network (GNN) architecture (see [Sainsbury-Dale, Zammit-Mangion, Richards, and Huser, 2023](https://arxiv.org/abs/2310.02600)). The overall workflow remains as given in previous examples, with some key additional steps:

- Sampling spatial configurations during the training phase, typically using an appropriately chosen spatial point process: see, for example, [`maternclusterprocess`](@ref).
- Storing the spatial data as a graph: see [`spatialgraph`](@ref).
- Constructing an appropriate architecture: see [`GNNSummary`](@ref) and [`SpatialGraphConv`](@ref).

For illustration, in this example we again consider the spatial Gaussian process model with exponential covariance function. 

First, we define a function to sample parameters from the prior. As before, the sampled parameters are stored as $p \times K$ matrices, with $p$ the number of parameters in the model and $K$ the number of sampled parameter vectors. We use the priors $\tau \sim U(0, 1)$ and $\rho \sim U(0.05, 0.5)$, and we assume that the parameters are independent a priori. 

Simulation from spatial Gaussian process models involves the computation of an expensive intermediate object, namely, the Cholesky factor of a covariance matrix. Storing this Cholesky factor can enable the fast simulation of new data sets given the previously sampled parameters: hence, in this example, we define a a type `Parameters` subtyping [`ParameterConfigurations`](@ref) for storing the matrix of parameters and the corresponding intermediate objects needed for data simulation. Further, we define two constructors for our `Parameters` object: one that accepts an integer $K$, and another that accepts a $p\times K$ matrix of parameters and a set of spatial locations associated with each parameter vector. The former constructor will be useful during the training stage for sampling from the prior distribution, while the latter constructor will be useful for parametric bootstrap (since this involves simulation from the fitted model).

```
struct Parameters{T} <: ParameterConfigurations
	θ::Matrix{T}
	S
	chols
	graphs
end

function sample(K::Integer; cluster_process::Bool = false)

	# Sample parameters from the prior distribution
	θ = rand(Uniform(0.05, 0.5), 1, K)

	# Simulate spatial configurations over the unit square
	n = rand(200:300, K)
	if cluster_process
		λ = rand(Uniform(10, 50), K)
		S = [maternclusterprocess(λ = λ[k], μ = n[k]/λ[k]) for k ∈ 1:K]
	else
		#S = [rand(n[k], 2) for k ∈ 1:K]
		pts = range(0, 1, length = 16)
		S = expandgrid(pts, pts)
		S = [S for _ in 1:K]
	end

	Parameters(θ, S)
end

function Parameters(θ::Matrix, S)

	K = size(θ, 2)

	# Construct spatial graphs
	graph = spatialgraph(S[1])
	graphs = repeat([graph], length(S))

	# Cholesky factor of covariance matrix
	D = pairwise(Euclidean(), S[1], dims = 1)
	chols = Folds.map(1:K) do k
		ρ = θ[1, k]
		Σ = exp.(-D ./ ρ)
		L = cholesky(Symmetric(Σ)).L
	end

	# Convert to Float32 for computational efficiency
	θ = Float32.(θ)  

	Parameters(θ, S, chols, graphs)
end
```

Next, we define a function for simulating from the model given an object of type `Parameters`. Although here we are constructing an estimator for a single replicate, the code below enables simulation of an arbitrary number of independent replicates `m`: one may provide a single integer for `m`, a range of values (e.g., `1:30`), or any object that can be sampled using `rand(m, K)` (e.g., some distribution over the possible sample sizes).

```
function simulate(parameters::Parameters, m)
	θ = parameters.θ
	K = size(θ, 2)
	m = rand(m, K)
	Folds.map(1:K) do k
		L = parameters.chols[k]
		g = parameters.graphs[k]
		Z = simulategaussianprocess(L, m[k])  # simulate fields
		spatialgraph(g, Z)                    # add to graph
	end
end
simulate(parameters::Parameters, m::Integer) = simulate(parameters, range(m, m))
```

Next we construct an appropriate GNN architecture, as illustrated below. Here, our goal is to construct a point estimator, however any other kind of estimator (see [Estimators](@ref)) can be constructed by simply substituting the appropriate estimator class in the final line below.

```
# Propagation module
dₕ = 256  # dimension of final node feature vectors
propagation = GNNChain(
	SpatialGraphConv(1 => 32),
	SpatialGraphConv(32 => 64),
	SpatialGraphConv(64 => dₕ)
	)

# Readout module and dimension of readout vector
readout = GlobalPool(mean); dᵣ = dₕ  

# Summary network
ψ = GNNSummary(propagation, readout)

# Mapping module
p = 1 # number of parameters in the statistical model
ϕ = Chain(Dense(dᵣ, 64, relu), Dense(64, p))

# DeepSet object
deepset = DeepSet(ψ, ϕ)

# Point estimator
θ̂ = PointEstimator(deepset)
```

Next, we train the estimator using [`train()`](@ref), here using the default absolute-error loss a single realisation per parameter configuration (i.e., with $m = 1$). Below, we use a relatively small number of epochs and a small number of training parameter vectors to keep the run time reasonably low. In practice, one might set `K` to some large value (say, 10000), and leave `epochs` unspecified so that training halts only when the risk function ceases to decrease.

```
m = 1
K = 20000
θ_train = sample(K)
θ_val = sample(K ÷ 10)
θ̂ = train(θ̂, θ_train, θ_val, simulate, m = m, epochs_per_Z_refresh = 3)
```

The function [`assess`](@ref) can be used to assess the trained estimator.

```
θ_test = sample(1000)
Z_test = simulate(θ_test, m)
assessment = assess(θ̂, θ_test, Z_test)
bias(assessment)
rmse(assessment)
plot(assessment)
```

Finally, once the estimator has been assessed, it may be applied to observed data, with bootstrap-based uncertainty quantification facilitated by [`bootstrap`](@ref) and [`interval`](@ref). Below, we use simulated data as a substitute for observed data:

```
# Generate some toy data
parameters = sample(1)       # sample a single parameter vector
Z = simulate(parameters, m)  # simulate data                  
θ = parameters.θ             # true parameters used to generate data
S = parameters.S             # observed locations

# Point estimates
θ̂(Z)

# Parametric bootstrap sample and bootstrap confidence interval
bs = bootstrap(θ̂, Parameters(θ̂(Z), S), simulate, 1)   
interval(bs)				                
```
