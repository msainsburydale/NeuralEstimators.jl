# Advanced usage

In this section, we discuss practical considerations on how to construct neural estimators most effectively.

## Loading pre-trained neural estimators

As training is by far the most computationally demanding part of the workflow, one typically trains an estimator and then saves it for later use. More specifically, one usually saves the *parameters* of the neural estimator (e.g., the weights and biases of the neural networks); then, to load the neural estimator at a later time, one initialises an estimator with the same architecture used during training, and then loads the saved parameters into this estimator.

If the argument `savepath` is specified, [`train`](@ref) automatically saves the neural estimator's parameters; to load them, one may use the following code, or similar:

```
using Flux: loadparams!

θ̂ = architecture()
loadparams!(θ̂, loadbestweights(savepath))
```

Above, `architecture()` is a user-defined function that returns a neural estimator with the same architecture as the estimator that we wish to load, but with randomly initialised parameters, and the function `loadparams!` loads the parameters of the best (as determined by [`loadbestweights`](@ref)) neural estimator saved in `savepath`.


## Storing expensive intermediate objects for data simulation

Parameters sampled from the prior distribution $\Omega(\cdot)$ may be stored in two ways. Most simply, they can be stored as a $p \times K$ matrix, where $p$ is the number of parameters in the model and $K$ is the number of parameter vectors sampled from the prior distribution; this is the approach taken in the example using univariate Gaussian data. Alternatively, they can be stored in a user-defined subtype of the abstract type [`ParameterConfigurations`](@ref), whose only requirement is a field `θ` that stores the $p \times K$ matrix of parameters. With this approach, one may store computationally expensive intermediate objects, such as Cholesky factors, for later use when conducting "on-the-fly" simulation, which is discussed below.

## On-the-fly and just-in-time simulation

When data simulation is (relatively) computationally inexpensive, $\mathcal{Z}_{\text{train}}$ can be simulated continuously during training, a technique coined "simulation-on-the-fly". Regularly refreshing $\mathcal{Z}_{\text{train}}$ leads to lower out-of-sample error and to a reduction in overfitting. This strategy therefore facilitates the use of larger, more representationally-powerful networks that are prone to overfitting when $\mathcal{Z}_{\text{train}}$ is fixed. Refreshing $\mathcal{Z}_{\text{train}}$ also has an additional computational benefit; data can be simulated "just-in-time", in the sense that they can be simulated from a small batch of $\vartheta_{\text{train}}$, used to train the neural estimator, and then removed from memory. This can reduce pressure on memory resources when $|\vartheta_{\text{train}}|$ is very large.

One may also regularly refresh $\vartheta_{\text{train}}$, and doing so leads to similar benefits. However, fixing $\vartheta_{\text{train}}$ allows computationally expensive terms, such as Cholesky factors when working with Gaussian process models, to be reused throughout training, which can substantially reduce the training time for some models.  

The above strategies are facilitated with various methods of [`train`](@ref).

## Combining learned and expert summary statistics

See [`DeepSetExpert`](@ref).

## Variable sample sizes

A neural estimator in the Deep Set representation can be applied to data sets of arbitrary size. However, even when the neural Bayes estimator approximates the true Bayes estimator arbitrarily well, it is conditional on the number of replicates, $m$, and is not necessarily a Bayes estimator for $m^* \ne m$. Denote a data set comprising $m$ replicates as $\boldsymbol{Z}^{(m)} \equiv (\boldsymbol{Z}_1', \dots, \boldsymbol{Z}_m')'$. There are at least two (non-mutually exclusive) approaches one could adopt if data sets with varying $m$ are envisaged, which we describe below.

### Piecewise estimators

If data sets with varying $m$ are envisaged, one could train $l$ neural Bayes estimators for different sample sizes, or groups thereof (e.g., a small-sample estimator and a large-sample estimator).
 Specifically, for sample-size changepoints $m_1$, $m_2$, $\dots$, $m_{l-1}$, one could construct a piecewise neural Bayes estimator,
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
where, here, $\boldsymbol{\gamma}^* \equiv (\boldsymbol{\gamma}^*_{\tilde{m}_1}, \dots, \boldsymbol{\gamma}^*_{\tilde{m}_{l-1}})$, and where $\boldsymbol{\gamma}^*_{\tilde{m}}$ are the neural-network parameters optimised for sample size $\tilde{m}$ chosen so that $\hat{\boldsymbol{\theta}}(\cdot; \boldsymbol{\gamma}^*_{\tilde{m}})$ is near-optimal over the range of sample sizes in which it is applied.
This approach works well in practice, and it is less computationally burdensome than it first appears when used in conjunction with pre-training.

Piecewise neural estimators are implemented with the struct, [`PiecewiseEstimator`](@ref), and their construction is facilitated with [`trainx`](@ref).  

### Training with variable sample sizes

Alternatively, one could treat the sample size as a random variable, $M$, with support over a set of positive integers, $\mathcal{M}$, in which case, for the neural Bayes estimator, the risk function becomes
```math
R(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\cdot; \boldsymbol{\gamma}))
\equiv
\sum_{m \in \mathcal{M}}
P(M=m)\left(\int_{\mathcal{S}^m}  L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(m)}; \boldsymbol{\gamma}))p(\boldsymbol{Z}^{(m)} \mid \boldsymbol{\theta}) {\text{d}} \boldsymbol{Z}^{(m)}\right).
```
 This approach does not materially alter the workflow, except that one must also sample the number of replicates before simulating the data.

 Below we define data simulation for a range of sample sizes (i.e., a range of integers) under a discrete uniform prior for ``M``, the random variable corresponding to sample size.

```
function simulate(parameters, m::R) where {R <: AbstractRange{I}} where I <: Integer

	## Number of parameter vectors stored in parameters
	K = size(parameters, 2)

	## Generate K sample sizes from the prior distribution for M
	m̃ = rand(m, K)

	## Pseudocode for data simulation
	Z = [<simulate m̃[k] iid realisations from the model> for k ∈ 1:K]

	return Z
end
```

Then, setting the argument `m` in [`train`](@ref) to be an integer range (e.g., `1:30`) will train the neural estimator with the given variable sample sizes.

## Missing data

Neural networks do not naturally handle missing data, and this property can preclude their use in a large array of applications. Here, we describe two techniques that alleviate this challenge: [The one-hot-encoding approach](@ref) and [The neural EM algorithm](@ref).

As a running example, we consider a Gaussian process model where the data are measured over a regular grid, but where some elements of the grid are unobserved. This situation may arise in, for example, a remote-sensing application, where the presence of cloud cover prevents measurement in some places. Below, we load the packages needed in this example, and define some aspects of the model that will remain constant throughout (e.g., the prior, the spatial domain, etc.).

```
using Distances: pairwise, Euclidean
using Distributions: Uniform
using Flux
using LinearAlgebra
using NeuralEstimators
import NeuralEstimators: simulate
using Statistics: mean


# Set the prior distribution
Ω = (τ = Uniform(0.01, 1.0),
		 ρ = Uniform(0.01, 0.4))

p = length(Ω)    # number of parameters in the statistical model

# Set the (gridded) spatial domain
points = range(0.0, 1.0, 16)
S = expandgrid(points, points)


# Model information that is constant (and which will be passed into later functions)
ξ = (
	Ω = Ω,
	ν = 1.0, 	# fixed smoothness
	S = S,
	D = pairwise(Euclidean(), S, S, dims = 1),
	p = p
)


# Sampler from the prior
struct Parameters <: ParameterConfigurations
	θ
	cholesky_factors
end

function Parameters(K::Integer, ξ)

	# Sample parameters from the prior
	Ω = ξ.Ω
	τ = rand(Ω.τ, K)
	ρ = rand(Ω.ρ, K)

	# Compute Cholesky factors  
	cholesky_factors = maternchols(ξ.D, ρ, ξ.ν)

	# Concatenate into a matrix
	θ = permutedims(hcat(τ, ρ))
	θ = Float32.(θ)

	Parameters(θ, cholesky_factors)
end
```

### The one-hot-encoding approach

The first missing-data technique that we consider is the so-called one-hot-encoding approach of [Wang et al. (2022)](https://www.biorxiv.org/content/10.1101/2023.01.09.523219v1). Their strategy involves completing the data by replacing missing values with zeros, and using auxiliary variables to encode the missingness pattern, which are also passed into the network.

Let $\boldsymbol{Z}$ denote the complete-data vector. Then, the one-hot-encoding approach considers inference based on $\boldsymbol{W}$, a vector of indicator variables that encode the missingness pattern (with elements equal to one or zero if the corresponding element of $\boldsymbol{Z}$ is observed or missing, respectively), and

```math
\boldsymbol{U} \equiv \boldsymbol{Z} \odot \boldsymbol{W},
```

where $\odot$ denotes elementwise multiplication. Irrespective of the missingness pattern, $\boldsymbol{U}$ and $\boldsymbol{W}$ have the same fixed dimensions and hence may be processed easily using a single neural network. A neural point estimator is then trained on realisations of $\{\boldsymbol{U}, \boldsymbol{W}\}$ which, by construction, do not contain any missing elements.

Since the missingness pattern $\boldsymbol{W}$ is now an input to the neural network, it must be incorporated during the training phase. When interest lies in making inference from a single already-observed data set, $\boldsymbol{W}$ is fixed and known, and the Bayes risk remains unchanged. Amortised inference, on the other hand, requires a model for $\boldsymbol{W}$, and the Bayes risk then also depends on this model. This can have substantial implications; a misspecified model for $\boldsymbol{W}$ will lead to a misspecified (neural) Bayes estimator and, even under correct specification, the resulting (neural) Bayes estimator may still be sub-optimal for a particular $\boldsymbol{W}$, since estimators based on average-risk optimality do not, in general, minimise the risk uniformly. Nevertheless, the one-hot-encoding approach is a flexible and computationally efficient method, and below we show how it can be incorporated in the usual workflow of `NeuralEstimators` (see also the help files for [`removedata`](@ref) and [`encodedata`](@ref)).

```
# Marginal simulation from the data model
function simulate(parameters::Parameters, m::Integer)

	K = size(parameters, 2)
	τ = parameters.θ[1, :]

	Z = map(1:K) do k
		L = parameters.cholesky_factors[:, :, k]
		z = simulategaussianprocess(L, m)
		z = z + τ[k] * randn(size(z)...)
		z = Float32.(z)
		z = reshape(z, 16, 16, 1, :)
		z
	end

	return Z
end

# Marginal simulation from the data model and a MCAR missingness model
function simulatemissing(parameters::Parameters, m::Integer)

	Z = simulate(parameters, m)   # simulate completely-observed data
	π_prior = Uniform(0.0, 1.0)   # prior for the proportion of missingness

	UW = map(Z) do z
		π = rand(π_prior) 				# sample the missingness proportion from the prior
		z = removedata(z, π)			# randomly remove a proportion π of the data
		uw = encodedata(z)				# replace missing entries with zero and encode missingness pattern
		uw
	end

	return UW
end
```

Next, we construct and train a neural estimator, here in the DeepSets representation. Note that the first convolutional layer takes two input channels, since we store the augmented data $\boldsymbol{U}$ in the first channel and the missingness pattern $\boldsymbol{W}$ in the second.

```
ψ = Chain(
	Conv((10, 10), 2 => 16,  relu),
	Conv((5, 5),  16 => 32,  relu),
	Conv((3, 3),  32 => 64, relu),
	Flux.flatten
	)
ϕ = Chain(Dense(64, 256, relu), Dense(256, p, exp))
θ̂ = DeepSet(ψ, ϕ)

θ̂ = train(θ̂, Parameters, simulatemissing, m = 1, ξ = ξ, K = 1000, epochs = 10)
```

Once trained, we can apply our neural estimator to (incomplete) observed data. The data must be encoded in the same manner that was done during training.

```
θ = Parameters(1, ξ)
Z = simulate(θ, 1)[1]
Z = removedata(Z, 0.25)				# remove 25% of the data
UW = encodedata(Z)
θ̂(UW)
```


### The neural EM algorithm

Let $\boldsymbol{Z}_1$ and $\boldsymbol{Z}_2$ denote the observed and unobserved (i.e., missing) data, respectively, and let $\boldsymbol{Z} \equiv (\boldsymbol{Z}_1', \boldsymbol{Z}_2')'$ denote the complete data. A classical approach to facilitating inference when data are missing is the expectation-maximisation (EM) algorithm. The *neural EM algorithm* is a Monte Carlo variant of the EM algorithm with $l$th iteration,

```math
\boldsymbol{\theta}^{(l)} = \argmax_{\boldsymbol{\theta}} \sum_{h = 1}^H \ell(\boldsymbol{\theta};  \boldsymbol{Z}_1,  \boldsymbol{Z}_2^{(lh)}) + \log \pi(\boldsymbol{\theta}),
```

where realisations of the missing-data component, $\{\boldsymbol{Z}_2^{(lh)} : h = 1, \dots, H\}$, are sampled from the conditional probability distribution of $\boldsymbol{Z}_2$ given $\boldsymbol{Z}_1$ and $\boldsymbol{\theta}^{(l-1)}$, and where $\pi(\boldsymbol{\theta}) \propto \{\omega(\boldsymbol{\theta})\}^H$ is a concentrated version of the original prior density. Given the conditionally sampled data, the above EM update is performed using a neural Bayes estimator that is trained to approximate the MAP estimator (i.e., the posterior mode) from a set of $H$ independent replicates of $\boldsymbol{Z}$.

First, we construct a neural approximation of the MAP estimator. When $H$ is taken to be large, we can lean on the Bernstein-von Mises theorem to train the estimator under linear or quadratic loss; otherwise, one should train the estimator under (a surrogate for) the 0--1 loss (e.g., the [`kpowerloss`](@ref) in the limit $\kappa \to 0$).

```
ψ = Chain(
	Conv((10, 10), 1 => 16,  relu),
	Conv((5, 5),  16 => 32,  relu),
	Conv((3, 3),  32 => 64, relu),
	Flux.flatten
	)
ϕ = Chain(
	Dense(64, 256, relu),
	Dense(256, p, exp)
	)
neuralMAPestimator = DeepSet(ψ, ϕ)

H = 50
neuralMAPestimator = train(neuralMAPestimator, Parameters, simulate, m = H, ξ = ξ, K = 1000, epochs = 10)
```

Next, we define a function for conditional simulation.

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
	ν = ξ.ν
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

Now we can use the neural EM algorithm to get parameter estimates from data that contain missing values. The algorithm is implemented with the type [`NeuralEM`](@ref).

```
θ = Parameters(1, ξ)
Z = simulate(θ, 1)[1][:, :]		# simulate a single gridded field
Z = removedata(Z, 0.25)				# remove 25% of the data

neuralem = NeuralEM(simulateconditional, neuralMAPestimator)
θ₀ = mean.([Ω...]) 						# initial estimate, the prior mean
neuralem(Z, θ₀, ξ = ξ, nsims = H)
```


## Censored data

Coming soon, based on the methodology presented in [Richards et al. (2023+)](https://arxiv.org/abs/2306.15642).
