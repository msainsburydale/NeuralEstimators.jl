# Examples

The following packages are used throughout these examples.
```
using NeuralEstimators, Flux, Distributions, NamedArrays
```

## Univariate data

Here we develop a neural Bayes estimator for $\boldsymbol{\theta} \equiv (\mu, \sigma)'$ from data $Z_1, \dots, Z_m$ that are independent and identically distributed realisations from the distribution $N(\mu, \sigma^2)$.

First, we define a function to sample parameters from the prior. Here, we assume that the parameters are independent a priori and we adopt the marginal priors $\mu \sim N(0, 1)$ and $\sigma \sim IG(3, 1)$. The sampled parameters are stored as $p \times K$ matrices, with $p$ the number of parameters in the model and $K$ the number of sampled parameter vectors. Note that below we store the parameters as a named matrix for convenience, but this is not a requirement of the package:

```
function sample(K)
	μ = rand(Normal(0, 1), K)
	σ = rand(InverseGamma(3, 1), K)
	θ = hcat(μ, σ)'
	θ = NamedArray(θ)
	setnames!(θ, ["μ", "σ"], 1)
	return θ
end
```

Next, we implicitly define the statistical model with simulated data. In `NeuralEstimators`, the data are always stored as a `Vector{A}`, where each element of the vector is associated with one parameter vector, and where the type `A` depends on the multivariate structure of the data. Since each replicate $Z_1, \dots, Z_m$ is univariate, `A` should be a `Matrix` with $d=1$ rows and $m$ columns.

Below, we define our simulator given a single parameter vector, and given a matrix of parameter vectors (which simply applies the simulator to each column):

```
simulate(θ, m) = θ["μ"] .+ θ["σ"] .* randn32(1, m)
simulate(θ::AbstractMatrix, m) = simulate.(eachcol(θ), m)
```

We now design a neural-network architecture. Since our data $Z_1, \dots, Z_m$ are replicated, we will use the [`DeepSet`](@ref) architecture. The outer network (also known as the inference network) is always a fully-connected network. However, the architecture of the inner network (also known as the summary network) depends on the multivariate structure of the data: with unstructured data (i.e., when there is no spatial or temporal correlation within a replicate), we use a fully-connected neural network. This architecture is then used to initialise a [`PointEstimator`](@ref) object. Note that the architecture can be defined using raw `Flux` code (see below) or with the helper function [`initialise_estimator`](@ref):

```
d = 1   # dimension of each replicate
p = 2   # number of parameters in the statistical model

ψ = Chain(Dense(d, 32, relu), Dense(32, 32, relu))     # summary network
ϕ = Chain(Dense(32, 32, relu), Dense(32, p))           # inference network
architecture = DeepSet(ψ, ϕ)

θ̂ = PointEstimator(architecture)
```

Next, we train the neural estimator using [`train`](@ref), here using the default absolute-error loss. We'll train the estimator using 50 independent replicates per parameter configuration. Below, we pass our user-defined functions for sampling parameters and simulating data, but one may also pass parameter or data instances, which will be held fixed during training:

```
m = 50
θ̂ = train(θ̂, sample, simulate, m = m)
```

Since the training stage can be computationally demanding, one may wish to save a trained estimator and load it in later sessions: see [Saving and loading neural estimators](@ref) for details on how this can be done. See also the [Regularisation](@ref) methods that can be easily applied when constructing neural Bayes estimators.

The function [`assess`](@ref) can be used to assess the trained point estimator. Parametric and non-parametric bootstrap-based uncertainty quantification are facilitated by [`bootstrap`](@ref) and [`interval`](@ref), and this can also be included in the assessment stage through the keyword argument `boot`:

```
θ_test = sample(1000)
Z_test = simulate(θ_test, m)
assessment = assess(θ̂, θ_test, Z_test, boot = true)
```

An [`Assessment`](@ref) object contains the sampled parameters, the corresponding point estimates, and the corresponding lower and upper bounds of the bootstrap intervals. This object can be used to compute various diagnostics:

```
bias(assessment)
rmse(assessment)
risk(assessment)
coverage(assessment)
intervalscore(assessment)
plot(assessment)
```

![Univariate Gaussian example: Estimates vs. truth](../assets/figures/univariate.png)

As an alternative form of uncertainty quantification, one may approximate a set of marginal posterior quantiles by training a second neural Bayes estimator under the [`quantileloss`](@ref) function, which allows one to generate approximate marginal posterior credible intervals. This is facilitated with [`IntervalEstimator`](@ref) which, by default, targets 95% central credible intervals. Below, we use the same base architecture used for point estimation, which is wrapped in a more complex architecture that ensures that the estimated credible intervals are valid (i.e., that the estimated lower bound is always less than the estimated upper bound):

```
θ̂₂ = IntervalEstimator(architecture)
θ̂₂ = train(θ̂₂, sample, simulate, m = m)
```

The resulting posterior credible-interval estimator can also be assessed with empirical simulation-based methods using the function [`assess`](@ref), as we did above for the point estimator. Often, these intervals have better coverage than bootstrap-based intervals.

Once a neural Bayes estimator is calibrated, it may be applied to observed data. Below, we use simulated data as a substitute for observed data:

```
θ = sample(1)               # true parameters
Z = simulate(θ, m)          # "observed" data
θ̂(Z)                        # point estimates
interval(bootstrap(θ̂, Z))   # 95% non-parametric bootstrap intervals
interval(θ̂₂, Z)             # 95% marginal posterior credible intervals
```

Note that one may utilise the GPU above by calling `Z = gpu(Z)` before applying the estimator.

## Multivariate data

Suppose now that our data now consists of $m$ replicates $\mathbf{Z}_1, \dots, \mathbf{Z}_m$ of a $d$-dimensional multivariate distribution. Everything remains as given in the univariate example above, except that we now store the data as a vector of $d \times m$ matrices (previously they were stored as $1\times m$ matrices), and the inner network of the DeepSets representation takes a $d$-dimensional input (previously it took a 1-dimensional input).

Note that, when estimating a full covariance matrix, one may wish to constrain the neural estimator to only produce parameters that imply a valid (i.e., positive definite) covariance matrix. This can be achieved by appending a  [`CovarianceMatrix`](@ref) layer to the end of the outer network of the DeepSets representation. However, the estimator will often learn to provide valid estimates, even if not constrained to do so.


## Gridded data

```
using NeuralEstimators, Flux, Distributions, Distances, LinearAlgebra
```

For data collected over a regular grid, the neural Bayes estimator is typically based on a convolutional neural network (CNN).

In these settings, each data set must be stored as a ($D + 2$)-dimensional array, where $D$ is the dimension of the grid (e.g., $D = 1$ for time series, $D = 2$ for two-dimensional spatial grids, etc.). The first $D$ dimensions of the array correspond to the dimensions of the grid; the penultimate dimension stores the so-called "channels" (this dimension is singleton for univariate processes, two for bivariate processes, etc.); and the final dimension stores the independent replicates. For example, to store 50 independent replicates of a bivariate spatial process measured over a 10x15 grid, one would construct an array of dimension 10x15x2x50.

Below, we develop a neural Bayes estimator for the spatial Gaussian process model with exponential covariance function and unknown range parameter. The spatial domain is taken to be the unit square and we adopt the prior $\theta \sim U(0.05, 0.5)$.

```
prior(K) = rand(Uniform(0.05, 0.5), 1, K)
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

For data collected over a regular grid, the neural Bayes estimator is based on a convolutional neural network (CNN). For a useful introduction to CNNs, see, for example, [Dumoulin and Visin (2016)](https://arxiv.org/abs/1603.07285). For a 16x16 grid, one possible architecture is as follows:

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

Next, we initialise a point estimator and a posterior credible-interval estimator using our architecture defined above:

```
θ̂  = PointEstimator(architecture)
θ̂₂ = IntervalEstimator(architecture)
```

Now we train the estimators. Since simulation from this statistical model involves Cholesky factorisation, which is moderately expensive with $n=256$ spatial locations, here we used fixed parameter and data instances during training. See [Storing expensive intermediate objects for data simulation](@ref) for methods that allow one to avoid repeated Cholesky factorisation when performing [On-the-fly and just-in-time simulation](@ref):

```
K = 20000
θ_train = prior(K)
θ_val   = prior(K ÷ 10)
Z_train = simulate(θ_train)
Z_val   = simulate(θ_val)

θ̂  = train(θ̂,  θ_train, θ_val, Z_train, Z_val)
θ̂₂ = train(θ̂₂, θ_train, θ_val, Z_train, Z_val)
```



Once the estimators have been trained, we assess them using empirical simulation-based methods:

```
θ_test = prior(1000)
Z_test = simulate(θ_test)
assessment = assess([θ̂, θ̂₂], θ_test, Z_test)
assessment = assess(θ̂, θ_test, Z_test)

bias(assessment)
rmse(assessment)
coverage(assessment)
plot(assessment)
```

![Gridded spatial Gaussian process example: Estimates vs. truth](../assets/figures/gridded.png)

Finally, we can apply our neural Bayes estimators to observed data. Note that when we have a single replicate only (which is often the case in spatial statistics), non-parametric bootstrap is not possible, and we instead use parametric bootstrap:

```
θ = prior(1)                           # true parameter
Z = simulate(θ)                        # "observed" data
θ̂(Z)                                   # point estimates
interval(θ̂₂, Z)                        # 95% marginal posterior credible intervals
bs = bootstrap(θ̂, θ̂(Z), simulate, m)   # parametric bootstrap intervals
interval(bs)                           # 95% parametric bootstrap intervals
```

## Irregular spatial data

To cater for data collected over arbitrary spatial locations, one may construct a neural Bayes estimator with a graph neural network (GNN) architecture (see [Sainsbury-Dale, Zammit-Mangion, Richards, and Huser, 2023](https://arxiv.org/abs/2310.02600)). Some key steps involve:

- Sampling spatial locations to cover a wide range of spatial configurations during the training phase. This can be done using an appropriately chosen spatial point process: see, for example, [`maternclusterprocess`](@ref).
- Computing (spatially-weighted) adjacency matrices: see [`adjacencymatrix`](@ref).
- Storing the data as a graph: see [`GNNGraph`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/stable/api/gnngraph/#GNNGraph-type).
- Constructing an appropriate architecture: see [`GNNSummary`](@ref) and [`SpatialGraphConv`](@ref).

Before proceeding, we load the required packages:

```
using NeuralEstimators, Flux, GraphNeuralNetworks, Distances, Distributions, Folds, LinearAlgebra, Statistics
```

First, we define a function to sample parameters from the prior. As before, the sampled parameters are stored as $p \times K$ matrices, with $p$ the number of parameters in the model and $K$ the number of sampled parameter vectors. We use the priors $\tau \sim U(0, 1)$ and $\rho \sim U(0.05, 0.5)$, and we assume that the parameters are independent a priori. Simulation from this model involves the computation of an expensive intermediate object, namely, the Cholesky factor of the covariance matrix. Storing this Cholesky factor for re-use can enable the fast simulation of new data sets (provided that the parameters are held fixed): hence, in this example, we define a class, `Parameters`, which is a subtype of [`ParameterConfigurations`](@ref), for storing the matrix of parameters and the corresponding intermediate objects needed for data simulation.

If one wishes to make inference from a single spatial data set only, and this data is collected before the estimator is constructed, then the data can be simulated using the observed spatial locations. Otherwise, synthetic spatial locations must be generated during the training phase. If no prior knowledge on the spatial configurations is available, then a wide variety of spatial configurations must be simulated to produce an estimator that is broadly applicable. Below, we use a Matérn cluster process (see [`maternclusterprocess`](@ref)) for this task (note that the hyper-parameters of this process govern the expected number of locations in each sampled set of spatial locations, and the degree of clustering).

Below, we define two constructors for our `Parameters` object: one that constructs a `Parameters` object given an integer `K`, and another that constructs a `Parameters` object given a $p\times K$ matrix of parameters and a set of spatial locations associated with each parameter vector. The former constructor will be useful in the training stage for sampling from the prior distribution, while the latter constructor will be useful for parametric bootstrap (since this involves simulation from the fitted model).

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
