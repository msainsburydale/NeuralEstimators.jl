


# Estimators {#Estimators}

The package provides several classes of neural estimator, organised within a type hierarchy rooted at the abstract supertype [`NeuralEstimator`](/API/estimators#NeuralEstimators.NeuralEstimator).
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.NeuralEstimator' href='#NeuralEstimators.NeuralEstimator'><span class="jlbinding">NeuralEstimators.NeuralEstimator</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
NeuralEstimator
```


An abstract supertype for all neural estimators.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Estimators/Estimators.jl#L1-L4" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Posterior estimators {#Posterior-estimators}

[`PosteriorEstimator`](/API/estimators#NeuralEstimators.PosteriorEstimator) approximates the posterior distribution, using a flexible, parametric family of distributions (see [Approximate distributions](/API/approximatedistributions#Approximate-distributions)).
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.PosteriorEstimator' href='#NeuralEstimators.PosteriorEstimator'><span class="jlbinding">NeuralEstimators.PosteriorEstimator</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
PosteriorEstimator <: NeuralEstimator
PosteriorEstimator(summary_network, q::ApproximateDistribution)
PosteriorEstimator(summary_network, num_parameters::Integer; num_summaries::Integer, q = NormalisingFlow, kwargs...)
```


A neural estimator that approximates the posterior distribution $p(\boldsymbol{\theta} \mid \boldsymbol{Z})$, based on a neural `summary_network` and an approximate distribution `q` (see the available in-built [Approximate distributions](/API/approximatedistributions#Approximate-distributions)).

The `summary_network` maps data $\boldsymbol{Z}$ to a vector of learned summary statistics $\boldsymbol{t} \in \mathbb{R}^{d^*}$, which are then used to condition the approximate distribution `q`. The precise way in which the summary statistics condition `q` depends on the choice of approximate distribution: for example, [`GaussianMixture`](/API/approximatedistributions#NeuralEstimators.GaussianMixture) uses an MLP to map $\boldsymbol{t}$ directly to distributional parameters, while [`NormalisingFlow`](/API/approximatedistributions#NeuralEstimators.NormalisingFlow) uses $\boldsymbol{t}$ as a conditioning input at each coupling layer.

The convenience constructor builds `q` internally given `num_parameters` and `num_summaries`, with any additional keyword arguments passed to the constructor of `q`.

**Keyword arguments**
- `num_summaries::Integer`: the number of summary statistics output by `summary_network`. Must match the output dimension of `summary_network`.
  
- `q::Type{<:ApproximateDistribution} = NormalisingFlow`: the type of approximate distribution to use.
  
- `kwargs...`: additional keyword arguments passed to the constructor of `q`.
  

**Examples**

```julia
using NeuralEstimators, Flux, CairoMakie

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ N(0, 1) and σ ~ U(0, 1)
d, m = 2, 100  # dimension of θ and number of replicates
sampler(K) = NamedMatrix(μ = randn(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(m))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

# Neural network
num_summaries = 3d
summary_network = Chain(Dense(m, 64, gelu), Dense(64, 64, gelu), Dense(64, num_summaries))

# Initialise the estimator, with q built internally
estimator = PosteriorEstimator(summary_network, d; num_summaries = num_summaries)

# Or, build q explicitly
q = NormalisingFlow(d; num_summaries = num_summaries)
estimator = PosteriorEstimator(summary_network, q)

# Train the estimator
estimator = train(estimator, sampler, simulator, K = 3000)

# Plot the risk history
plotrisk()

# Assess the estimator
θ_test = sampler(250)
Z_test = simulator(θ_test);
assessment = assess(estimator, θ_test, Z_test)
plot(assessment)

# Inference with observed data 
θ = sampler(1)
Z = simulator(θ)
sampleposterior(estimator, Z) # posterior draws
posteriormean(estimator, Z)   # point estimate
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Estimators/PosteriorEstimator.jl#L1-L55" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Ratio estimators {#Ratio-estimators}

[`RatioEstimator`](/API/estimators#NeuralEstimators.RatioEstimator) approximates the likelihood-to-evidence ratio, enabling both frequentist and Bayesian inference through various downstream algorithms.
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.RatioEstimator' href='#NeuralEstimators.RatioEstimator'><span class="jlbinding">NeuralEstimators.RatioEstimator</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
RatioEstimator <: NeuralEstimator
RatioEstimator(summary_network, num_parameters; num_summaries, kwargs...)
```


A neural estimator that estimates the likelihood-to-evidence ratio,

$$r(\boldsymbol{Z}, \boldsymbol{\theta}) \equiv p(\boldsymbol{Z} \mid \boldsymbol{\theta})/p(\boldsymbol{Z}),$$

where $p(\boldsymbol{Z} \mid \boldsymbol{\theta})$ is the likelihood and $p(\boldsymbol{Z})$ is the marginal likelihood, also known as the model evidence.

The estimator jointly summarises the data $\boldsymbol{Z}$ and parameters $\boldsymbol{\theta}$  using separate summary networks, whose outputs are concatenated and passed to an MLP inference network.  The parameter summary network maps $\boldsymbol{\theta}$ to a vector of `2 * num_parameters` summaries by default.

For numerical stability, training is done on the log-scale using the relation  $\log r(\boldsymbol{Z}, \boldsymbol{\theta}) = \text{logit}(c^*(\boldsymbol{Z}, \boldsymbol{\theta}))$,  where $c^*(\cdot, \cdot)$ denotes the Bayes classifier as described in the [methodology](/methodology#Neural-ratio-estimators) section. 

Given data `Z` and parameters `θ`, the estimated ratio can be obtained using [logratio](/API/inference#NeuralEstimators.logratio)  and can be used in various Bayesian (e.g., [Hermans et al., 2020](https://proceedings.mlr.press/v119/hermans20a.html)) or frequentist (e.g., [Walchessen et al., 2024](https://doi.org/10.1016/j.spasta.2024.100848)) inferential algorithms. For Bayesian inference, posterior samples can be obtained via simple grid-based sampling using [sampleposterior](/API/inference#NeuralEstimators.sampleposterior).

**Keyword arguments**
- `num_summaries::Integer`: the number of summaries output by `summary_network`. Must match the output dimension of `summary_network`.
  
- `num_summaries_θ::Integer = 2 * num_parameters`: the number of summaries output by the parameter summary network.
  
- `summary_network_θ_kwargs::NamedTuple = (;)`: keyword arguments passed to the MLP constructor for the parameter summary network.
  
- `kwargs...`: additional keyword arguments passed to the MLP constructor for the inference network.
  

**Examples**

```julia
using NeuralEstimators, Flux, CairoMakie

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d, m = 2, 100  # dimension of θ and number of replicates
sampler(K) = NamedMatrix(μ = rand(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(m))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

# Neural network
num_summaries = 3d
summary_network = Chain(Dense(m, 64, gelu), Dense(64, 64, gelu), Dense(64, num_summaries))

# Initialise the estimator
estimator = RatioEstimator(summary_network, d; num_summaries = num_summaries)

# Train the estimator
estimator = train(estimator, sampler, simulator, K = 1000)

# Plot the risk history
plotrisk()

# Assess the estimator
θ_test = sampler(250)
Z_test = simulator(θ_test);
θ_grid = expandgrid(0:0.01:1, 0:0.01:1)'  # fine gridding of the parameter space
assessment = assess(estimator, θ_test, Z_test; θ_grid = θ_grid)
plot(assessment)

# Generate "observed" data 
θ = sampler(1)
z = simulator(θ)

# Grid-based optimization and sampling
logratio(estimator, z, θ_grid = θ_grid)                # log of likelihood-to-evidence ratios
posteriormode(estimator, z; θ_grid = θ_grid)           # posterior mode 
sampleposterior(estimator, z; θ_grid = θ_grid)         # posterior sample
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Estimators/RatioEstimator.jl#L1-L71" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Bayes estimators {#Bayes-estimators}

Neural Bayes estimators are implemented as subtypes of [`BayesEstimator`](/API/estimators#NeuralEstimators.BayesEstimator). The general-purpose [`PointEstimator`](/API/estimators#NeuralEstimators.PointEstimator) supports user-defined loss functions (see [Loss functions](/API/lossfunctions#Loss-functions)). The types [`IntervalEstimator`](/API/estimators#NeuralEstimators.IntervalEstimator) and its generalisation [`QuantileEstimator`](/API/estimators#NeuralEstimators.QuantileEstimator) are designed for posterior quantile estimation based on user-specified probability levels, automatically configuring the quantile loss and enforcing non-crossing constraints.
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.BayesEstimator' href='#NeuralEstimators.BayesEstimator'><span class="jlbinding">NeuralEstimators.BayesEstimator</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
BayesEstimator <: NeuralEstimator
```


An abstract supertype for neural Bayes estimators.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Estimators/Estimators.jl#L7-L10" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.PointEstimator' href='#NeuralEstimators.PointEstimator'><span class="jlbinding">NeuralEstimators.PointEstimator</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
PointEstimator <: BayesEstimator
PointEstimator(network)
PointEstimator(summary_network, inference_network)
PointEstimator(summary_network, num_parameters; num_summaries, kwargs...)
```


A neural point estimator mapping data to a point summary of the posterior distribution.

The neural network can be provided in two ways:
- As a single `network` that maps data directly to the parameter space.
  
- As a `summary_network` that maps data to a vector of summary statistics, with the `inference_network` constructed internally based on `num_parameters` and `num_summaries`.
  

**Examples**

```julia
using NeuralEstimators, Flux, CairoMakie

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ N(0, 1) and σ ~ U(0, 1)
d, m = 2, 100  # dimension of θ and number of replicates
sampler(K) = NamedMatrix(μ = randn(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(m))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

# Neural network, an MLP mapping m inputs into d outputs
network = Chain(Dense(m, 64, gelu), Dense(64, 64, gelu), Dense(64, d))

# Initialise a neural point estimator
estimator = PointEstimator(network)

# Train the estimator
estimator = train(estimator, sampler, simulator)

# Plot the risk history
plotrisk()

# Assess the estimator
θ_test = sampler(1000)
Z_test = simulator(θ_test)
assessment = assess(estimator, θ_test, Z_test)
bias(assessment)
rmse(assessment)
plot(assessment)

# Apply to observed data (here, simulated as a stand-in)
θ = sampler(1)           # ground truth (not known in practice)
Z = simulator(θ)         # stand-in for real observations
estimate(estimator, Z)   # point estimate
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Estimators/PointEstimator.jl#L1-L47" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.IntervalEstimator' href='#NeuralEstimators.IntervalEstimator'><span class="jlbinding">NeuralEstimators.IntervalEstimator</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
IntervalEstimator <: BayesEstimator
IntervalEstimator(u, v = u, c::Union{Function, Compress} = identity; probs = [0.025, 0.975], g = exp)
IntervalEstimator(u, c::Union{Function, Compress}; probs = [0.025, 0.975], g = exp)
```


A neural estimator that jointly estimates marginal posterior credible intervals based on the probability levels `probs` (by default, 95% central credible intervals).

The estimator employs a representation that prevents quantile crossing. Specifically, given data $\boldsymbol{Z}$,  it constructs intervals for each parameter $\theta_i$, $i = 1, \dots, d,$  of the form,

$$[c_i(u_i(\boldsymbol{Z})), \;\; c_i(u_i(\boldsymbol{Z})) + g(v_i(\boldsymbol{Z})))],$$

where  $\boldsymbol{u}(⋅) \equiv (u_1(\cdot), \dots, u_d(\cdot))'$ and $\boldsymbol{v}(⋅) \equiv (v_1(\cdot), \dots, v_d(\cdot))'$ are neural networks that map from the sample space to $\mathbb{R}^d$; $g(\cdot)$ is a monotonically increasing function (e.g., exponential or softplus); and each $c_i(⋅)$ is a monotonically increasing function that maps its input to the prior support of $\theta_i$.

The functions $c_i(⋅)$ may be collectively defined by a $d$-dimensional [`Compress`](/API/architectures#NeuralEstimators.Compress) object, which can constrain the interval estimator&#39;s output to the prior support. If these functions are unspecified, they will be set to the identity function so that the range of the intervals will be unrestricted. If only a single neural-network architecture is provided, it will be used for both $\boldsymbol{u}(⋅)$ and $\boldsymbol{v}(⋅)$.

The return value when applied to data using [`estimate`()](/API/inference#NeuralEstimators.estimate) is a matrix with $2d$ rows, where the first and second $d$ rows correspond to the lower and upper bounds, respectively. The function [`interval()`](/API/inference#NeuralEstimators.interval) can be used to format this output in a readable $d$ × 2 matrix.  

See also [`QuantileEstimator`](/API/estimators#NeuralEstimators.QuantileEstimator).

**Examples**

```julia
using NeuralEstimators, Flux, CairoMakie

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d = 2     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 100   # number of independent replicates
sample(K) = rand32(d, K)
simulate(θ, m) = [ϑ[1] .+ ϑ[2] .* randn(n, m) for ϑ in eachcol(θ)]

# Neural network
w = 128   # width of each hidden layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, d))
u = DeepSet(ψ, ϕ)

# Initialise the estimator
estimator = IntervalEstimator(u)

# Train the estimator
estimator = train(estimator, sample, simulate, simulator_args = m, K = 3000)

# Plot the risk history
plotrisk()

# Assess the estimator
θ_test = sample(500)
Z_test = simulate(θ_test, m);
assessment = assess(estimator, θ_test, Z_test)
plot(assessment)

# Inference with "observed" data 
θ = [0.8f0; 0.1f0]
Z = simulate(θ, m)
estimate(estimator, Z) 
interval(estimator, Z)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Estimators/QuantileEstimator.jl#L1-L64" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.QuantileEstimator' href='#NeuralEstimators.QuantileEstimator'><span class="jlbinding">NeuralEstimators.QuantileEstimator</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
QuantileEstimator <: BayesEstimator
QuantileEstimator(v; probs = [0.025, 0.5, 0.975], g = Flux.softplus, i = nothing)
```


A neural estimator that jointly estimates a fixed set of marginal posterior quantiles, with probability levels $\{\tau_1, \dots, \tau_T\}$ controlled by the keyword argument `probs`. This generalises [`IntervalEstimator`](/API/estimators#NeuralEstimators.IntervalEstimator) to support an arbitrary number of probability levels. 

Given data $\boldsymbol{Z}$, by default the estimator approximates quantiles of the distributions of 

$$\theta_i \mid \boldsymbol{Z}, \quad i = 1, \dots, d, $$

for parameters $\boldsymbol{\theta} \equiv (\theta_1, \dots, \theta_d)'$. Alternatively, if initialised with `i` set to a positive integer, the estimator approximates quantiles of the full conditional distribution of  

$$\theta_i \mid \boldsymbol{Z}, \boldsymbol{\theta}_{-i},$$

where $\boldsymbol{\theta}_{-i}$ denotes the parameter vector with its $i$th element removed. 

The estimator employs a representation that prevents quantile crossing, namely,

$$\begin{aligned}
\boldsymbol{q}^{(\tau_1)}(\boldsymbol{Z}) &= \boldsymbol{v}^{(\tau_1)}(\boldsymbol{Z}),\\
\boldsymbol{q}^{(\tau_t)}(\boldsymbol{Z}) &= \boldsymbol{v}^{(\tau_1)}(\boldsymbol{Z}) + \sum_{j=2}^t g(\boldsymbol{v}^{(\tau_j)}(\boldsymbol{Z})), \quad t = 2, \dots, T,
\end{aligned}$$

where $\boldsymbol{q}^{(\tau)}(\boldsymbol{Z})$ denotes the vector of $\tau$-quantiles  for parameters $\boldsymbol{\theta} \equiv (\theta_1, \dots, \theta_d)'$;  $\boldsymbol{v}^{(\tau_t)}(\cdot)$, $t = 1, \dots, T$, are neural networks that map from the sample space to $\mathbb{R}^d$; and $g(\cdot)$ is a monotonically increasing function (e.g., exponential or softplus) applied elementwise to its arguments. If `g = nothing`, the quantiles are estimated independently through the representation

$$\boldsymbol{q}^{(\tau_t)}(\boldsymbol{Z}) = \boldsymbol{v}^{(\tau_t)}(\boldsymbol{Z}), \quad t = 1, \dots, T.$$

When the neural networks are [`DeepSet`](/API/architectures#NeuralEstimators.DeepSet) objects, two requirements must be met.  First, the number of input neurons in the first layer of the outer network must equal the number of neurons in the final layer of the inner network plus $\text{dim}(\boldsymbol{\theta}_{-i})$, where we define  $\text{dim}(\boldsymbol{\theta}_{-i}) \equiv 0$ when targetting marginal posteriors of the form $\theta_i \mid \boldsymbol{Z}$ (the default behaviour).  Second, the number of output neurons in the final layer of the outer network must equal $d - \text{dim}(\boldsymbol{\theta}_{-i})$. 

The return value is a matrix with $\{d - \text{dim}(\boldsymbol{\theta}_{-i})\} \times T$ rows, where the first $T$ rows correspond to the estimated quantiles for the first parameter, the second $T$ rows corresponds to the estimated quantiles for the second parameter, and so on.

**Examples**

```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d = 2     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 30    # number of independent replicates in each data set
sample(K) = rand32(d, K)
simulate(θ, m) = [ϑ[1] .+ ϑ[2] .* randn32(n, m) for ϑ in eachcol(θ)]

# ---- Quantiles of θᵢ ∣ 𝐙, i = 1, …, d ----

# Neural network
w = 64   # width of each hidden layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, d))
v = DeepSet(ψ, ϕ)

# Initialise the estimator
estimator = QuantileEstimator(v)

# Train the estimator
estimator = train(estimator, sample, simulate, simulator_args = m)

# Inference with "observed" data 
θ = [0.8f0; 0.1f0]
Z = simulate(θ, m)
estimate(estimator, Z) 

# ---- Quantiles of θᵢ ∣ 𝐙, θ₋ᵢ ----

# Neural network
w = 64  # width of each hidden layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w + 1, w, relu), Dense(w, d - 1))
v = DeepSet(ψ, ϕ)

# Initialise estimators respectively targetting quantiles of μ∣Z,σ and σ∣Z,μ
q₁ = QuantileEstimator(v; i = 1)
q₂ = QuantileEstimator(v; i = 2)

# Train the estimators
q₁ = train(q₁, sample, simulate, simulator_args = m)
q₂ = train(q₂, sample, simulate, simulator_args = m)

# Estimate quantiles of μ∣Z,σ with σ = 0.5 and for many data sets
θ₋ᵢ = 0.5f0
q₁(Z, θ₋ᵢ)

# Estimate quantiles of μ∣Z,σ with σ = 0.5 for a single data set
q₁(Z[1], θ₋ᵢ)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Estimators/QuantileEstimator.jl#L87-L187" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Ensembles {#Ensembles}

[`Ensemble`](/API/estimators#NeuralEstimators.Ensemble) combines multiple estimators, aggregating their individual estimates to improve accuracy.
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.Ensemble' href='#NeuralEstimators.Ensemble'><span class="jlbinding">NeuralEstimators.Ensemble</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Ensemble <: NeuralEstimator
Ensemble(estimators)
Ensemble(architecture::Function, J::Integer)
(ensemble::Ensemble)(Z; aggr = mean)
```


Defines an ensemble of `estimators` which, when applied to data `Z`, returns the mean (or another summary defined by `aggr`) of the individual estimates (see, e.g., [Sainsbury-Dale et al., 2025, Sec. S5](https://doi.org/10.48550/arXiv.2501.04330)).

The ensemble can be initialised with a collection of trained `estimators` and then applied immediately to observed data. Alternatively, the ensemble can be initialised with a collection of untrained `estimators` (or a function defining the architecture of each estimator, and the number of estimators in the ensemble), trained with `train()`, and then applied to observed data. In the latter case, where the ensemble is trained directly, if `savepath` is specified both the ensemble and component estimators will be saved.

Note that `train()` currently acts sequentially on the component estimators, using the `Adam` optimiser.

The ensemble components can be accessed by indexing the ensemble; the number of component estimators can be obtained using `length()`.

See also [`Parallel`](https://fluxml.ai/Flux.jl/stable/reference/models/layers/#Flux.Parallel), which can be used to mimic ensemble methods with an appropriately chosen `connection`. 

**Examples**

```julia
using NeuralEstimators, Flux

# Data Z|θ ~ N(θ, 1) with θ ~ N(0, 1)
d = 1     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 30    # number of independent replicates in each data set
sampler(K) = randn32(d, K)
simulator(θ, m) = [μ .+ randn32(n, m) for μ ∈ eachcol(θ)]

# Neural-network architecture of each ensemble component
function architecture()
	ψ = Chain(Dense(n, 64, relu), Dense(64, 64, relu))
	ϕ = Chain(Dense(64, 64, relu), Dense(64, d))
	network = DeepSet(ψ, ϕ)
	PointEstimator(network)
end

# Initialise ensemble with three component estimators 
ensemble = Ensemble(architecture, 3)
ensemble[1]      # access component estimators by indexing
ensemble[1:2]    # indexing with an iterable collection returns the corresponding ensemble 
length(ensemble) # number of component estimators

# Training
ensemble = train(ensemble, sampler, simulator, m = m, epochs = 5)

# Assessment
θ = sampler(1000)
Z = simulator(θ, m)
assessment = assess(ensemble, θ, Z)
rmse(assessment)

# Apply to data
ensemble(Z)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Estimators/Ensemble.jl#L1-L58" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Helper functions {#Helper-functions}

The following helper functions operate on an estimator to inspect its components or apply parts of it to data. For the main inference functions used post-training, see [Inference with observed data](/API/inference#Inference-with-observed-data).
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.summarynetwork' href='#NeuralEstimators.summarynetwork'><span class="jlbinding">NeuralEstimators.summarynetwork</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
summarynetwork(estimator::NeuralEstimator)
```


Returns the summary network of `estimator`.

See also [`summarystatistics`](/API/estimators#NeuralEstimators.summarystatistics).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Estimators/Estimators.jl#L15-L20" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.setsummarynetwork' href='#NeuralEstimators.setsummarynetwork'><span class="jlbinding">NeuralEstimators.setsummarynetwork</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
setsummarynetwork(estimator::NeuralEstimator, network)
```


Returns a new estimator identical to `estimator` but with the summary network replaced by `network`. Useful for transfer learning.

Note that [`RatioEstimator`](/API/estimators#NeuralEstimators.RatioEstimator) has a second summary network for the parameters, accessible via `estimator.summary_network_θ`, which is not affected by this function.

See also [`summarynetwork`](/API/estimators#NeuralEstimators.summarynetwork).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Estimators/Estimators.jl#L24-L31" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.summarystatistics' href='#NeuralEstimators.summarystatistics'><span class="jlbinding">NeuralEstimators.summarystatistics</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
summarystatistics(estimator::NeuralEstimator, Z; batchsize::Integer = 32, use_gpu::Bool = true)
```


Computes learned summary statistics by applying the summary network of `estimator` to data `Z`.

If `Z` is a [`DataSet`](/API/parametersdata#NeuralEstimators.DataSet) object, the learned summary statistics are concatenated with the precomputed expert summary statistics stored in `Z.S`.

See also [`summarynetwork`](/API/estimators#NeuralEstimators.summarynetwork).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Estimators/Estimators.jl#L36-L44" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.logdensity' href='#NeuralEstimators.logdensity'><span class="jlbinding">NeuralEstimators.logdensity</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
logdensity(estimator::PosteriorEstimator, θ, Z)
```


Evaluates the log-density of the approximate posterior implied by `estimator` at parameters `θ` given data `Z`,

$$\log q(\boldsymbol{\theta} \mid \boldsymbol{Z}),$$

where $q$ denotes the approximate posterior distribution.

`θ` should be a $d \times K$ matrix of parameter vectors and `Z` a collection of `K` data sets.

See also [`sampleposterior`](/API/inference#NeuralEstimators.sampleposterior).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Estimators/PosteriorEstimator.jl#L82-L93" target="_blank" rel="noreferrer">source</a></Badge>

</details>

