# Examples

## Univariate Gaussian data

Here, we develop a neural Bayes estimator for $\boldsymbol{\theta} \equiv (\mu, \sigma)'$ from data $Z_1, \dots, Z_m$ that are independent and identically distributed according to a $N(\mu, \sigma^2)$ distribution.

Before proceeding, we load the required packages:
```
using NeuralEstimators
using Flux
using Distributions
import NeuralEstimators: simulate
```

First, we sample parameters from the prior $\Omega(\cdot)$ to construct parameter sets used for training, validating, and testing the estimator. Here, we use the priors $\mu \sim N(0, 1)$ and $\sigma \sim U(0.1, 1)$, and we assume that the parameters are independent a priori. The sampled parameters are stored as $p \times K$ matrices, with $p$ the number of parameters in the model and $K$ the number of sampled parameter vectors.
```
function sample(Ω, K)
	μ = rand(Ω.μ, K)
	σ = rand(Ω.σ, K)
	θ = hcat(μ, σ)'
	return θ
end

Ω = (μ = Normal(0, 1), σ = Uniform(0.1, 1))
θ_train = sample(Ω, 10000)
θ_val   = sample(Ω, 2000)
θ_test  = sample(Ω, 1000)
```

Next, we implicitly define the statistical model via simulated data. In general, the data are stored as a `Vector{A}`, where each element of the vector is associated with one parameter vector, and where `A` depends on the representation of the neural estimator. Since our data is replicated, we will use the Deep Sets framework and, since each replicate is univariate, we will use a dense neural network (DNN) for the inner network. Since the inner network is a DNN, the data should be stored as an `Array`, with independent replicates stored in the final dimension.
```
function simulate(θ_set, m)
	Z = [rand(Normal(θ[1], θ[2]), 1, m) for θ ∈ eachcol(θ_set)]
	Z = broadcast.(Float32, Z) # convert to Float32 for computational efficiency
	return Z
end

m = 15 # number of independent replicates per parameter vector
Z_train = simulate(θ_train, m)
Z_val   = simulate(θ_val, m)
```

We now design architectures for the inner and outer neural networks, $\boldsymbol{\psi}(\cdot)$ and $\boldsymbol{\phi}(\cdot)$ respectively, in the Deep Sets framework, and initialise the neural estimator as a [`DeepSet`](@ref) object.

```
p = length(Ω)   # number of parameters in the statistical model
w = 32          # number of neurons in each layer

ψ = Chain(Dense(1, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, p))
θ̂ = DeepSet(ψ, ϕ)
```

Next, we train the neural estimator using [`train`](@ref), here using the default absolute-error loss.
```
θ̂ = train(θ̂, θ_train, θ_val, Z_train, Z_val, epochs = 30)
```

To test the accuracy of the resulting neural Bayes estimator, we use the function [`assess`](@ref), which can be used to assess the performance of the estimator (or multiple estimators) over a range of sample sizes. Note that, in this example, we trained the neural estimator using a single sample size, $m = 15$, and hence the estimator will not necessarily be optimal for other sample sizes; see [Variable sample sizes](@ref) for approaches that one could adopt if data sets with varying sample size are envisaged.
```
Z_test     = [simulate(θ_test, m) for m ∈ (5, 10, 15, 20, 30)]
assessment = assess([θ̂], θ_test, Z_test)
```

The returned object is an object of type [`Assessment`](@ref), which contains the true parameters and their corresponding estimates, and the time taken to compute the estimates for each sample size and each estimator. The risk function may be computed using the function [`risk`](@ref), and plotted against the sample size with [`plotrisk`](@ref):
```
risk(assessment)
plotrisk(assessment)
```

It is often helpful to visualise the empirical joint distribution of an estimator for a particular parameter configuration and a particular sample size. This can be done by providing [`assess`](@ref) with $J$ data sets simulated under a particular parameter configuration, and then calling [`plotdistribution`](@ref):
```
J = 100
θ = sample(Ω, 1)
Z = [simulate(θ, m, J)]
assessment = assess([θ̂], θ, Z)  
plotdistribution(assessment)
```

Once the neural Bayes estimator has been assessed, it may then be applied to observed data, with parametric/non-parametric bootstrap-based uncertainty quantification facilitated by [`bootstrap`](@ref) and [`confidenceinterval`](@ref). Below, we use simulated data as a substitute for observed data:
```
Z = simulate(θ, m)     # pretend that this is observed data
θ̂(Z)                   # point estimates from the observed data
θ̃ = bootstrap(θ̂, Z)    # non-parametric bootstrap estimates
confidenceinterval(θ̃)  # confidence interval from the bootstrap estimates
```
