# Advanced usage

## Saving and loading neural estimators

In regards to saving and loading, neural estimators behave in the same manner as regular Flux models. Therefore, the examples and recommendations outlined in the [Flux documentation](https://fluxml.ai/Flux.jl/stable/guide/saving/) also apply directly to neural estimators. For example, to save the model state of the neural estimator `estimator`, run:

```
using Flux
using BSON: @save, @load
model_state = Flux.state(estimator)
@save "estimator.bson" model_state
```

Then, to load it in a new session, one may initialise a neural estimator with the same architecture used previously, and load the saved model state as follows:

```
@load "estimator.bson" model_state
Flux.loadmodel!(estimator, model_state)
```

It is also straightforward to save the entire neural estimator, including its architecture (see [here](https://fluxml.ai/Flux.jl/stable/guide/saving/#Saving-Models-as-Julia-Structs)). However, the first approach outlined above is recommended for long-term storage.

For convenience, the function [`train()`](@ref) allows for the automatic saving of the model state during the training stage, via the argument `savepath`.

## Storing expensive intermediate objects for data simulation

Parameters sampled from the prior distribution may be stored in two ways. Most simply, they can be stored as a $d \times K$ matrix, where $d$ is the number of parameters in the model and $K$ is the number of parameter vectors sampled from the prior distribution. Alternatively, they can be stored in a user-defined subtype of [`ParameterConfigurations`](@ref), whose only requirement is a field `θ` that stores the $d \times K$ matrix of parameters. With this approach, one may store computationally expensive intermediate objects, such as Cholesky factors, for later use when conducting "on-the-fly" simulation, which is discussed below.

## On-the-fly and just-in-time simulation

When data simulation is (relatively) computationally inexpensive, the training data set, $\mathcal{Z}_{\text{train}}$, can be simulated continuously during training, a technique coined "simulation-on-the-fly". Regularly refreshing $\mathcal{Z}_{\text{train}}$ leads to lower out-of-sample error and to a reduction in overfitting. This strategy therefore facilitates the use of larger, more representationally-powerful networks that are prone to overfitting when $\mathcal{Z}_{\text{train}}$ is fixed. Further, this technique allows for data to be simulated "just-in-time", in the sense that they can be simulated in small batches, used to train the neural estimator, and then removed from memory. This can substantially reduce pressure on memory resources, particularly when working with large data sets.

One may also regularly refresh the set $\vartheta_{\text{train}}$ of parameter vectors used during training, and doing so leads to similar benefits. However, fixing $\vartheta_{\text{train}}$ allows computationally expensive terms, such as Cholesky factors when working with Gaussian process models, to be reused throughout training, which can substantially reduce the training time for some models. Hybrid approaches are also possible, whereby the parameters (and possibly the data) are held fixed for several epochs (i.e., several passes through the training set when performing stochastic gradient descent) before being refreshed.

The above strategies are facilitated with various methods of [`train()`](@ref).


## Regularisation

The term *regularisation* refers to a variety of techniques aimed to reduce overfitting when training a neural network, primarily by discouraging complex models.

A popular regularisation technique is known as dropout, implemented in Flux's [`Dropout`](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dropout) layer. Dropout involves temporarily dropping ("turning off") a randomly selected set of neurons (along with their connections) at each iteration of the training stage, which results in a computationally-efficient form of model (neural-network) averaging [(Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html).

Another class of regularisation techniques involve modifying the loss function. For instance, L₁ regularisation (sometimes called lasso regression) adds to the loss a penalty based on the absolute value of the neural-network parameters. Similarly, L₂ regularisation (sometimes called ridge regression) adds to the loss a penalty based on the square of the neural-network parameters. Note that these penalty terms are not functions of the data or of the statistical-model parameters that we are trying to infer. These regularisation techniques can be implemented straightforwardly by providing a custom `optimiser` to [`train()`](@ref) that includes a [`SignDecay`](https://fluxml.ai/Flux.jl/stable/reference/training/optimisers/#Optimisers.SignDecay) object for L₁ regularisation, or a [`WeightDecay`](https://fluxml.ai/Flux.jl/stable/reference/training/optimisers/#Optimisers.WeightDecay) object for L₂ regularisation. See the [Flux documentation](https://fluxml.ai/Flux.jl/stable/guide/training/training/#Regularisation) for further details. Note that, when the training data and parameters are simulated dynamically (i.e., "on the fly"; see [On-the-fly and just-in-time simulation](@ref)), overfitting is generally not a concern, making this form of regularisation unnecessary.

For illustration, the following code constructs a neural Bayes estimator using dropout and L₁ regularisation with penalty coefficient $\lambda = 10^{-4}$:

```
using NeuralEstimators, Flux

# Data Z|θ ~ N(θ, 1) with θ ~ N(0, 1)
d = 1     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 5     # number of independent replicates in each data set
sampler(K) = randn32(d, K)
simulator(θ, m) = [μ .+ randn32(n, m) for μ ∈ eachcol(θ)]
K = 3000  # number of training samples
θ_train = sampler(K)
θ_val   = sampler(K)
Z_train = simulator(θ_train, m)
Z_val   = simulator(θ_val, m)

# Neural network with dropout layers
w = 128
ψ = Chain(Dense(1, w, relu), Dropout(0.1), Dense(w, w, relu), Dropout(0.5))     
ϕ = Chain(Dense(w, w, relu), Dropout(0.5), Dense(w, 1))           
network = DeepSet(ψ, ϕ)

# Initialise estimator
estimator = PointEstimator(network)

# Optimiser with L₁ regularisation
optimiser = Flux.setup(OptimiserChain(SignDecay(1e-4), Adam()), estimator)

# Train the estimator
train(estimator, θ_train, θ_val, Z_train, Z_val; optimiser = optimiser)
```

## Expert summary statistics

Implicitly, neural estimators involve the learning of summary statistics. However, some summary statistics are available in closed form, simple to compute, and highly informative (e.g., sample quantiles, the empirical variogram). Often, explicitly incorporating these expert summary statistics in a neural estimator can simplify the optimisation problem, and lead to a better estimator.

The fusion of learned and expert summary statistics is facilitated by our implementation of the [`DeepSet`](@ref) framework. Note that this implementation also allows the user to construct a neural estimator using only expert summary statistics, following, for example, [Gerber and Nychka (2021)](https://onlinelibrary.wiley.com/doi/abs/10.1002/sta4.382) and [Rai et al. (2024)](https://onlinelibrary.wiley.com/doi/abs/10.1002/env.2845). Note also that the user may specify arbitrary expert summary statistics, however, for convenience several standard [User-defined summary statistics](@ref) are provided with the package, including a fast, sparse approximation of the empirical variogram.

For an example of incorporating expert summary statistics, see [Irregular spatial data](@ref), where the empirical variogram is used alongside learned graph-neural-network-based summary statistics.

## Variable sample sizes

A neural estimator in the Deep Set representation can be applied to data sets of arbitrary size. However, even when the neural Bayes estimator approximates the true Bayes estimator arbitrarily well, it is conditional on the number of replicates, $m$, and is not necessarily a Bayes estimator for $m^* \ne m$. Denote a data set comprising $m$ replicates as $\boldsymbol{Z}^{(m)} \equiv (\boldsymbol{Z}_1', \dots, \boldsymbol{Z}_m')'$. There are at least two (non-mutually exclusive) approaches one could adopt if data sets with varying $m$ are envisaged, which we describe below.

### Piecewise estimators

If data sets with varying $m$ are envisaged, one could train $l$ estimators for different sample sizes, or groups thereof (e.g., a small-sample estimator and a large-sample estimator). For example, for sample-size changepoints $m_1$, $m_2$, $\dots$, $m_{l-1}$, one could construct a piecewise neural Bayes estimator,
```math
\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(m)}; \boldsymbol{\gamma}^*)
=
\begin{cases}
\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(m)}; \boldsymbol{\gamma}^*_{\tilde{m}_1}) & m \leq m_1,\\
\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(m)}; \boldsymbol{\gamma}^*_{\tilde{m}_2}) & m_1 < m \leq m_2,\\
\quad \vdots \\
\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(m)}; \boldsymbol{\gamma}^*_{\tilde{m}_l}) & m > m_{l-1},
\end{cases}
```
where $\boldsymbol{\gamma}^* \equiv (\boldsymbol{\gamma}^*_{\tilde{m}_1}, \dots, \boldsymbol{\gamma}^*_{\tilde{m}_{l-1}})$, and $\boldsymbol{\gamma}^*_{\tilde{m}}$ are the neural-network parameters optimised for sample size $\tilde{m}$ chosen so that $\hat{\boldsymbol{\theta}}(\cdot; \boldsymbol{\gamma}^*_{\tilde{m}})$ is near-optimal over the range of sample sizes in which it is applied. This approach works well in practice and is less computationally burdensome than it first appears when used in conjunction with the technique known as pre-training (see [Sainsbury-Dale at al., 2024](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522), Sec 2.3.3), which is facilitated with [`trainx()`](@ref). 

Piecewise estimators are implemented using the type [`PiecewiseEstimator`](@ref). 

### Training with variable sample sizes

Alternatively, one could treat the sample size as a random variable, $M$, with support over a set of positive integers, $\mathcal{M}$, in which case the Bayes risk becomes
```math
\sum_{m \in \mathcal{M}}
\textrm{Pr}(M=m)\left(
\int_\Theta \int_{\mathcal{Z}^m}  L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(m)}))p(\boldsymbol{Z}^{(m)} \mid \boldsymbol{\theta})\pi(\boldsymbol{\theta}) \textrm{d}\boldsymbol{Z}^{(m)} \textrm{d} \boldsymbol{\theta}
\right).
```
This approach does not materially alter the workflow, except that one must also sample the number of replicates before simulating the data during the training phase.

The following pseudocode illustrates how one may modify a general data simulator to train under a range of sample sizes, with the distribution of $M$ defined by passing any object that can be sampled using `rand(m, K)` (e.g., an integer range like `1:30`, an integer-valued distribution from [Distributions.jl](https://juliastats.org/Distributions.jl/stable/univariate/)):

```
# Method that allows m to be an object that can be sampled from
function simulate(parameters, m)
	# Number of parameter vectors stored in parameters
	K = size(parameters, 2)

	# Generate K sample sizes from the prior distribution for M
	m̃ = rand(m, K)

	# Pseudocode for data simulation
	Z = [<simulate m̃[k] realisations from the model> for k ∈ 1:K]

	return Z
end

# Method that allows an integer to be passed for m
simulate(parameters, m::Integer) = simulate(parameters, range(m, m))
```


## Missing data

Neural networks do not naturally handle missing data, and this property can preclude their use in a broad range of applications. Here, we describe two techniques that alleviate this challenge in the context of parameter point estimation: the [masking approach](@ref "The masking approach") and the [expectation-maximisation (EM) approach](@ref "The EM approach"). 

As a running example, we consider a Gaussian process model where the data are collected over a regular grid, but where some elements of the grid are unobserved. This situation often arises in, for example, remote-sensing applications, where the presence of cloud cover prevents measurement in some places. Below, we load the packages needed in this example, and define some aspects of the model that will remain constant throughout (e.g., the prior, the spatial domain). We also define types and functions for sampling from the prior distribution and for simulating marginally from the data model.

```
using NeuralEstimators, Flux
using Distributions: Uniform
using Distances, LinearAlgebra
using Statistics: mean

# Set the prior and define the number of parameters in the statistical model
Π = (
	τ = Uniform(0, 1.0),
	ρ = Uniform(0, 0.4)
)
d = length(Π)

# Define the (gridded) spatial domain and compute the distance matrix
points = range(0, 1, 16)
S = expandgrid(points, points)
D = pairwise(Euclidean(), S, dims = 1)

# Store model information for later use
ξ = (
	Π = Π,
	S = S,
	D = D
)

# Struct for storing parameters+Cholesky factors
struct Parameters <: ParameterConfigurations
	θ
	L
end

# Constructor for above struct
function Parameters(K::Integer, ξ)

	# Sample parameters from the prior
	Π = ξ.Π
	τ = rand(Π.τ, K)
	ρ = rand(Π.ρ, K)
	ν = 1 # fixed smoothness

	# Compute Cholesky factors  
	L = maternchols(ξ.D, ρ, ν)

	# Concatenate into matrix
	θ = permutedims(hcat(τ, ρ))

	Parameters(θ, L)
end

# Marginal simulation from the data model
function simulate(parameters::Parameters, m::Integer)

	K = size(parameters, 2)
	τ = parameters.θ[1, :]
	L = parameters.L
	n = isqrt(size(L, 1))

	Z = map(1:K) do k
		z = simulategaussian(L[:, :, k], m)
		z = z + τ[k] * randn(size(z)...)
		z = Float32.(z)
		z = reshape(z, n, n, 1, :)
		z
	end

	return Z
end
```

### The masking approach

The first missing-data technique that we consider is the so-called masking approach of [Wang et al. (2024)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012184); see also the discussion by [Sainsbury-Dale et al. (2025, Sec. 2.2)](https://doi.org/10.48550/arXiv.2501.04330). The strategy involves completing the data by replacing missing values with zeros, and using auxiliary variables to encode the missingness pattern, which are also passed into the network.

Let $\boldsymbol{Z}$ denote the complete-data vector. Then, the masking approach considers inference based on $\boldsymbol{W}$, a vector of indicator variables that encode the missingness pattern (with elements equal to one or zero if the corresponding element of $\boldsymbol{Z}$ is observed or missing, respectively), and

```math
\boldsymbol{U} \equiv \boldsymbol{Z} \odot \boldsymbol{W},
```

where $\odot$ denotes elementwise multiplication and the product of a missing element and zero is defined to be zero. Irrespective of the missingness pattern, $\boldsymbol{U}$ and $\boldsymbol{W}$ have the same fixed dimensions and hence may be processed easily using a single neural network. A neural point estimator is then trained on realisations of $\{\boldsymbol{U}, \boldsymbol{W}\}$ which, by construction, do not contain any missing elements.

Since the missingness pattern $\boldsymbol{W}$ is now an input to the neural network, it must be incorporated during the training phase. When interest lies only in making inference from a single already-observed data set, $\boldsymbol{W}$ is fixed and known, and the Bayes risk remains unchanged. However, amortised inference, whereby one trains a single neural network that will be used to make inference with many data sets, requires a joint model for the data $\boldsymbol{Z}$ and the missingness pattern $\boldsymbol{W}$, which is here defined as follows:

```
# Marginal simulation from the data model and a MCAR missingness model
function simulatemissing(parameters::Parameters, m::Integer)

	Z = simulate(parameters, m)   # complete data

	UW = map(Z) do z
		prop = rand()             # sample a missingness proportion
		z = removedata(z, prop)   # randomly remove a proportion of the data
		uw = encodedata(z)        # replace missing entries with zero and encode missingness pattern
		uw
	end

	return UW
end
```

Note that the helper functions [`removedata()`](@ref) and [`encodedata()`](@ref) facilitate the construction of augmented data sets $\{\boldsymbol{U}, \boldsymbol{W}\}$.

Next, we construct and train a masked neural Bayes estimator using a CNN architecture. Here, the first convolutional layer takes two input channels, since we store the augmented data $\boldsymbol{U}$ in the first channel and the missingness pattern $\boldsymbol{W}$ in the second. We construct a point estimator, but the masking approach is applicable with any other kind of estimator (see [Estimators](@ref)):

```
# Construct DeepSet object
ψ = Chain(
	Conv((10, 10), 2 => 16,  relu),
	Conv((5, 5),  16 => 32,  relu),
	Conv((3, 3),  32 => 64, relu),
	Flux.flatten
	)
ϕ = Chain(Dense(64, 256, relu), Dense(256, d, exp))
deepset = DeepSet(ψ, ϕ)

# Initialise point estimator
θ̂ = PointEstimator(deepset)

# Train the masked neural Bayes estimator
θ̂ = train(θ̂, Parameters, simulatemissing, m = 1, ξ = ξ, K = 1000, epochs = 10)
```

Once trained, we can apply our masked neural Bayes estimator to (incomplete) observed data. The data must be encoded in the same manner that was done during training. Below, we use simulated data as a surrogate for real data, with a missingness proportion of 0.25:

```
θ = Parameters(1, ξ)     # true parameters
Z = simulate(θ, 1)[1]    # complete data
Z = removedata(Z, 0.25)  # "observed" incomplete data (i.e., with missing values)
UW = encodedata(Z)       # augmented data {U, W}
θ̂(UW)                    # point estimate
```


### The EM approach

Let $\boldsymbol{Z}_1$ and $\boldsymbol{Z}_2$ denote the observed and unobserved (i.e., missing) data, respectively, and let $\boldsymbol{Z} \equiv (\boldsymbol{Z}_1', \boldsymbol{Z}_2')'$ denote the complete data. A classical approach to facilitating inference when data are missing is the expectation-maximisation (EM) algorithm. The *neural EM algorithm* ([Sainsbury-Dale et al., 2025](https://doi.org/10.48550/arXiv.2501.04330)) is an approximate version of the conventional (Bayesian) Monte Carlo EM algorithm which, at the $l$th iteration, updates the parameter vector through
```math
\boldsymbol{\theta}^{(l)} = \underset{\boldsymbol{\theta}}{\mathrm{arg\,max}} \sum_{h = 1}^H \ell(\boldsymbol{\theta};  \boldsymbol{Z}_1,  \boldsymbol{Z}_2^{(lh)}) + \log \pi_H(\boldsymbol{\theta}),
```
where realisations of the missing-data component, $\{\boldsymbol{Z}_2^{(lh)} : h = 1, \dots, H\}$, are sampled from the probability distribution of $\boldsymbol{Z}_2$ given $\boldsymbol{Z}_1$ and $\boldsymbol{\theta}^{(l-1)}$, and where $\pi_H(\boldsymbol{\theta}) \propto \{\pi(\boldsymbol{\theta})\}^H$ is a concentrated version of the original prior density. Given the conditionally simulated data, the neural EM algorithm performs the above EM update using a neural network that returns the MAP estimate (i.e., the posterior mode) using (complete) conditionally simulated data. 

First, we construct a neural approximation of the MAP estimator. In this example, we will take $H=50$. When $H$ is taken to be reasonably large, one may lean on the [Bernstein-von Mises](https://en.wikipedia.org/wiki/Bernstein%E2%80%93von_Mises_theorem) theorem to train the neural Bayes estimator under linear or quadratic loss; otherwise, one should train the estimator under a continuous relaxation of the 0--1 loss (e.g., the [`tanhloss()`](@ref) in the limit $\kappa \to 0$). This is done as follows:

```
# Construct DeepSet object
ψ = Chain(
	Conv((10, 10), 1 => 16,  relu),
	Conv((5, 5),  16 => 32,  relu),
	Conv((3, 3),  32 => 64, relu),
	Flux.flatten
	)
ϕ = Chain(
	Dense(64, 256, relu),
	Dense(256, d, exp)
	)
deepset = DeepSet(ψ, ϕ)

# Initialise point estimator
θ̂ = PointEstimator(deepset)

# Train neural Bayes estimator
H = 50
θ̂ = train(θ̂, Parameters, simulate, m = H, ξ = ξ, K = 1000, epochs = 10)
```

Next, we define a function for conditional simulation (see [`EM`](@ref) for details on the required format of this function):

```
function simulateconditional(Z::M, θ, ξ; nsims::Integer = 1) where {M <: AbstractMatrix{Union{Missing, T}}} where T

	# Save the original dimensions
	dims = size(Z)

	# Convert to vector
	Z = vec(Z)

	# Compute the indices of the observed and missing data
	I₁ = findall(z -> !ismissing(z), Z) # indices of observed data
	I₂ = findall(z -> ismissing(z), Z)  # indices of missing data
	n₁ = length(I₁)
	n₂ = length(I₂)

	# Extract the observed data and drop Missing from the eltype of the container
	Z₁ = Z[I₁]
	Z₁ = [Z₁...]

	# Distance matrices needed for covariance matrices
	D   = ξ.D # distance matrix for all locations in the grid
	D₂₂ = D[I₂, I₂]
	D₁₁ = D[I₁, I₁]
	D₁₂ = D[I₁, I₂]

	# Extract the parameters from θ
	τ = θ[1]
	ρ = θ[2]

	# Compute covariance matrices
	ν = 1 # fixed smoothness
	Σ₂₂ = matern.(UpperTriangular(D₂₂), ρ, ν); Σ₂₂[diagind(Σ₂₂)] .+= τ^2
	Σ₁₁ = matern.(UpperTriangular(D₁₁), ρ, ν); Σ₁₁[diagind(Σ₁₁)] .+= τ^2
	Σ₁₂ = matern.(D₁₂, ρ, ν)

	# Compute the Cholesky factor of Σ₁₁ and solve the lower triangular system
	L₁₁ = cholesky(Symmetric(Σ₁₁)).L
	x = L₁₁ \ Σ₁₂

	# Conditional covariance matrix, cov(Z₂ ∣ Z₁, θ),  and its Cholesky factor
	Σ = Σ₂₂ - x'x
	L = cholesky(Symmetric(Σ)).L

	# Conditonal mean, E(Z₂ ∣ Z₁, θ)
	y = L₁₁ \ Z₁
	μ = x'y

	# Simulate from the distribution Z₂ ∣ Z₁, θ ∼ N(μ, Σ)
	z = randn(n₂, nsims)
	Z₂ = μ .+ L * z

	# Combine the observed and missing data to form the complete data
	Z = map(1:nsims) do l
		z = Vector{T}(undef, n₁ + n₂)
		z[I₁] = Z₁
		z[I₂] = Z₂[:, l]
		z
	end
	Z = stackarrays(Z, merge = false)

	# Convert Z to an array with appropriate dimensions
	Z = reshape(Z, dims..., 1, nsims)

	return Z
end
```

Now we can use the neural EM algorithm to get parameter point estimates from data containing missing values. The algorithm is implemented with the type [`EM`](@ref). Again, here we use simulated data as a surrogate for real data:

```
θ = Parameters(1, ξ)            # true parameters
Z = simulate(θ, 1)[1][:, :]     # complete data
Z = removedata(Z, 0.25)         # "observed" incomplete data (i.e., with missing values)
θ₀ = mean.([Π...])              # initial estimate, the prior mean

neuralem = EM(simulateconditional, θ̂)
neuralem(Z, θ₀, ξ = ξ, nsims = H, use_ξ_in_simulateconditional = true)
```


## Censored data

Neural estimators can be constructed to handle censored data as input, by exploiting [The masking approach](@ref) detailed above for the case of missing data. The key difference is that, unlike the data missingness pattern, the censoring pattern is assumed to be known a priori and must be user specified. For simplicity, we here describe methdology for left censored data (i.e., we observe only data that exceed some threshold), but extensions to right or interval censoring are possible. 

[Richards et al. (2024)](https://jmlr.org/papers/v25/23-1134.html) discuss neural point estimation from censored data in the context of peaks-over-threshold extreme value models, whereby artifical censoring of data is imposed to reduce estimation bias in the presence of non-extreme marginal events. In peaks-over-threshold modelling, observed data are treated as censored if they exceed their corresponding marginal $\tau$-quantile, for $\tau \in (0,1)$ close to one. We present two approaches to censoring data: a [General setting](@ref), where users specifiy their own deterministic "censoring", and [Peaks-over-threshold censoring](@ref), where users supply a (censoring) quantile level $\tau$ that can be treated as random and features in the neural network architecture.

As a running example, we consider a bivariate random scale Gaussian mixture; see [Engelke, Opitiz, and Wadsworth (2019)](https://link.springer.com/article/10.1007/s10687-019-00353-3) and [Huser and Wadsworth (2018)](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1411813). We consider the task of estimating $\boldsymbol{\theta}=(\rho,\delta)'$, where $\rho \in [0,1)$ is a correlation parameter and $\delta \in [0,1]$ is a shape parameter. Data $\boldsymbol{Z}_1,\dots,\boldsymbol{Z}_m$ are independent and identically distributed according to the random scale construction
 ```math
\boldsymbol{Z}_i = \delta R_i + (1-\delta)  \boldsymbol{X}_i,
```
where $R_i$ is a unit exponential random variable and $\boldsymbol{X}_i$ is a bivariate random vector with unit exponential margins and a Gaussian copula with correlation $\rho$. 

Below, we construct a neural point estimator for fully observed data generated from this model.
```
using NeuralEstimators, Flux, Folds
using NeuralEstimators: Symmetric
using NeuralEstimators: cholesky
using Distributions: Uniform, Normal
using CairoMakie 



m = 200     # number of independent replicates in each data set

function sample(K)
	# Sample parameters from the prior 
	ρ = rand(Uniform(0.0, 0.99),1, K)
	δ = rand(Uniform(0.0, 1.0),1, K)
	return vcat(ρ,δ)
end

function simulate(θ, m) 

	K = size(θ, 2)
	Z = Folds.map(1:K) do k
		ρ = θ[1,k]
		δ = θ[2,k]
		Σ = [1 ρ; ρ 1]
		L = cholesky(Symmetric(Σ)).L

		X = L * randn(2,m) #Standard Gaussian margins
		X = cdf.(Normal(),X)  #Uniform margins
		X = - log.(1 .- X) #Unit exponential margins

		R = -log.(1 .- rand(Uniform(0.0, 1.0),1, m)) #Unit exponential margins

		z = δ .* R .+ (1 - δ) .* X

		z
	end
	Z
end


# Hidden layer width
w = 32  

final_layer = Dense(w, 2, sigmoid)      # ρ, δ  ∈ [0,1]

# Inner and outer networks
ψ = Chain(Dense(2, w, relu), Dense(w, w, relu))    
ϕ = Chain(Dense(w, w, relu), final_layer)          

# Combine into a DeepSet
deepset = DeepSet(ψ, ϕ)

θ̂ = PointEstimator(network)

# training: full simulation on-the-fly
θ̂ = train(θ̂, sample, simulate, m = m)


θ_test = sample(1000)
Z_test = simulate(θ_test, m)
assessment = assess(θ̂, θ_test, Z_test, boot = false)     
plot(assessment)
```

### General setting

Let $\boldsymbol{Z} \equiv (\boldsymbol{Z}_1', \dots, \boldsymbol{Z}_m')'$ denote the complete-data vector. In the general censoring setting, a user must provide a function `censor_augment` that constructs, from $\boldsymbol{Z}$, augmented data $\boldsymbol{A}=(\tilde{\boldsymbol{Z}}, \boldsymbol{W}')'$. Similarly to [The masking approach](@ref), we here consider inference using a vector of indicator variables that encode the censoring pattern, denoted by $\boldsymbol{W}$; here $\boldsymbol{W}$ has elements equal to one or zero if the corresponding element of $\boldsymbol{Z}$ is left censored or observed, respectively. However, unlike typical masking, we do not set censored values to `missing`; we instead construct $\tilde{\boldsymbol{Z}}$, which comprises the vector $\boldsymbol{Z}$ with censored values set to some pre-specified constant, contained within the vector $\boldsymbol{\zeta}$, such that

```math
\tilde{\boldsymbol{Z}} \equiv \boldsymbol{Z} \odot \boldsymbol{W} + \boldsymbol{\zeta} \odot ( \boldsymbol{1} - \boldsymbol{W}),
```
where $\boldsymbol{1}$ is a vector of ones (of equivalent dimension to $\boldsymbol{W}$) and where $\odot$ denotes elementwise multiplication. Note that $\boldsymbol{\zeta}$ and the censoring pattern can differ across replicates $t=1,\dots,m$, as well as the underlying model parameter values.

The augmented data $\boldsymbol{A}$ is an input to the neural network, and the inner network $\boldsymbol{\psi}$ should be designed as to account for its dimension. The way in which concatenation of $\tilde{\boldsymbol{Z}}$ and $\boldsymbol{W}$ is performed may differ depending on the type of the first layer use in $\boldsymbol{\psi}$: if using a `Dense` layer, one can concatenate $\tilde{\boldsymbol{Z}}$ and $\boldsymbol{W}$ along the first dimension (as they are both matrices; i nthis case, with dimension ($2,m$)); if using graph layers or `Conv` layers, $\tilde{\boldsymbol{Z}}$ and $\boldsymbol{W}$ should be concatenated as if they were two separate channels in a graph/image (see [`encodedata()`](@ref)).

Note that the function `censor_augment` should be applied to data during both training and at evaluation time; any manipulation of data that is performed at train time should also be performed to data at test time!

Below, the function `censor_augment` takes in a vector of censoring levels $\boldsymbol{c}$ of the same length as $\boldsymbol{Z}_i$, and sets censored values to $\zeta_1=\dots=\zeta_m=\zeta$; in this way, the censoring mechanism and augmentation values, $\boldsymbol{\zeta}$, do not vary with the model parameter values or with the replicate index.

```
function censor_augment(z; c,  ζ=0)
    I = 1*(z .<= c)
    z=ifelse.(z .<= c,  ζ, z)
    return vcat(z, I)
end
```

Censoring is performed during training. To ensure this, we employ `censor_augment` during simulation:

```
function simulate_censored(θ, m; c, ζ) 
    
    K = size(θ, 2)
	A = Folds.map(1:K) do k
        ρ = θ[1,k]
        δ = θ[2,k]
        Σ = [1.0 ρ; ρ 1.0]
        L = cholesky(Symmetric(Σ)).L

        X = L * randn(2,m) #Standard Gaussian margins
        X = cdf.(Normal(),X)  #Uniform margins
        X = - log.(1 .- X) #Unit exponential margins

        R = -log.(1 .- rand(Uniform(0.0, 1.0),1, m)) #Unit exponential margins

        z = δ .* R .+ (1 - δ) .* X

        
        
        #censor data and create augmented datasest
        A =  mapslices(Z -> censor_augment(Z, c=c, ζ=ζ), z, dims=1)
        A # augmented dataset

	end
	A
end

# We can now generate data, with values below 0.4 censored and set to \zeta = -1.
K = 1000  # number of training samples

θ_train = sample(K)
c = [0.4, 0.4]
Z_train = simulate_censored(θ_train, m;  c = c, ζ = -1.0)
```

To adapt the point estimator architecture to handle the augmented dataset, we change the dimension of the input from two to four.
```
ψ₂ = Chain(Dense(4, w, relu), Dense(w, w, relu))   
ϕ₂ = Chain(Dense(w, w, relu), Dense(w, 2, sigmoid) )          


# Combine into a DeepSet
θ̂_cNBE = PointEstimator(DeepSet(ψ₂, ϕ₂))


# training: full simulation on-the-fly
simulator(θ, m)  = simulate_censored(θ, m; c=c, ζ=-1.0) 
θ̂_cNBE = train(θ̂_cNBE, sample, simulator, m = m)
```

We assess the estimator using the same test parameter values as for `θ̂` above (the neural estimator designed for fully observed data). We should observe increased estimation variance, as the censoring removes information from the estmation procedure.

```
Z_test = simulator(θ_test, m)
assessment_censored = assess(θ̂_cNBE, θ_test, Z_test, boot = false)   
plot(assessment_censored)
```

### Peaks-over-threshold censoring

In a peak-over-threshold modelling setup, censoring of data is determined by a user-specified quantile level $\tau$, typically taken to be close to one. During inference, artificial censoring is imposed: data that do not exceed their marginal $\tau$-quantile are treated as left censored. For example, in the running example, we censor components of $\boldsymbol{Z}_i$ below the $\tau$-quantile of the marginal distribution of the random scale mixture, i.e., $F^{-1}(\tau; \delta)$, where $F(z;\delta)$ is the ($\delta$-dependent) marginal distribution function of $\boldsymbol{Z}_i$; this has a closed form expression, see [Huser and Wadsworth (2018)](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1411813).

Peaks-over-threshold modelling, with $\tau$ fixed, can easilly be implemented by adapting the `censor_augment` in [General setting](@ref), i.e., by imposing that `c` is an evaluation of $F(\tau; \delta)$. However, [Richards et al. (2024)](https://jmlr.org/papers/v25/23-1134.html) show that one can amortise a point estimator with respect to the choice of $\tau$, by treating $\tau$ as random and allowing it to feature as an input into the outer neural network, $\boldsymbol{\phi}$. Note that, in this setting, the estimator cannot be trained via `simulation-on-the-fly`, and a finite number of $K$ data/parameter pairs must be sampled prior to training!

We also follow [Richards et al. (2024)](https://jmlr.org/papers/v25/23-1134.html) and consider inference for data on standardised margins; that is, we pre-standardise the data $\boldsymbol{Z}$ to have unit exponential margins, rather than $\delta$-dependent margins. This can help to improve the numerical stability of estimator training, as well as increasing training efficiency.

```
# Define prior for θ and τ. Here we allow $\tau$ to be Uniform(0.4,0.6).

function sample_θτ(K)

	# Sample parameters from the prior 

	ρ = rand(Uniform(0.0, 0.99),1, K)
    δ = rand(Uniform(0.0, 1.0),1, K)
	τ = rand(Uniform(0.4,0.6),1,  K)

    return vcat(ρ, δ, τ)
end


function simulate_censored_tau(θτ,  m;  ζ) 

    K = size(θτ, 2)
	A = Folds.map(1:K) do k
        ρ = θτ[1,k]
        δ = θτ[2,k]
        τ = θτ[3,k]

        Σ = [1.0 ρ; ρ 1.0]
        L = cholesky(Symmetric(Σ)).L

        X = L * randn(2,m) #Standard Gaussian margins
        X = cdf.(Normal(),X)  #Uniform margins
        X = - log.(1 .- X) #Unit exponential margins

        R = -log.(1 .- rand(Uniform(0.0, 1.0),1, m)) #Unit exponential margins

        z = δ .* R .+ (1 - δ) .* X

        #Transform to Uniform margins; see Huser and Wadsworth (2018)
        if δ == 0.5 
            z = 1 .- exp.(-2 .* z).*(1 .+ 2 .* z) 
        else 
            z = 1 .- (δ ./ (2 .* δ	.- 1)) .* exp.(- z ./ δ) .+ ((1 .- δ) ./ (2*δ .- 1 )) .* exp.(-z ./ (1-δ)) 
        end

        
        Z = -  log.(1 .- z) # Unit exponential margins; H^-1() in Richards et al., 2024
        
      
        c = -log(1 - τ)
        #censor data and create augmented datasest
        A =  mapslices(Z -> censor_augment(Z, c=c, ζ=ζ), Z, dims=1)
        A

	end
	A
end

# We then generate data used for training and validation.

K = 3000  # number of training samples
θτ_train = sample_θτ(K)
θτ_val   = sample_θτ(K)


Z_train = simulate_censored_tau(θτ_train, m;  ζ = -1.0)
Z_val   = simulate_censored_tau(θτ_val, m;  ζ = -1.0)


θ_train = θτ_train[1:2,:]
θ_val   = θτ_val[1:2,:]

τ_train = θτ_train[3:3,:]
τ_val   = θτ_val[3:3,:]
```

As $\tau$ features as an input into the outer network of the DeepSet estimator, we must increase the dimension of the input to $\boldsymbol{\phi}$.

```
# Inner and outer networks
ψ₃ = Chain(Dense(4, w, relu), Dense(w, w, relu))    
ϕ₃ = Chain(Dense(w+1, w, relu), Dense(w, 2, sigmoid) )    

# Combine into a DeepSet
θ̂_cNBE_τ = PointEstimator(DeepSet(ψ₃, ϕ₃))


# training; note we make use of set-level statistics
θ̂_cNBE_τ = train(θ̂_cNBE_τ, θ_train, θ_val, (Z_train, τ_train), (Z_val, τ_val))


# assessment with τ fixed to 0.5, but using the same test values as previously
θτ_test = vcat(θ_test,repeat([0.5],1000)')

Z_test = simulate_censored_tau(θτ_test, m;  ζ = -1.0)

assessment_tau = assess(θ̂_cNBE_τ, θ_test, (Z_test,repeat([0.5],1000)'), boot = false)    
plot(assessment_tau)


# assessment with τ fixed to 0.4
θτ_test = vcat(θ_test,repeat([0.4],1000)')
                        
                        Z_test = simulate_censored_tau(θτ_test, m;  ζ = -1.0)
                        
                        assessment_tau = assess(θ̂_cNBE_τ, θ_test, (Z_test,repeat([0.4],1000)'), boot = false)    
plot(assessment_tau)

```

