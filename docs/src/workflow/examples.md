# Examples

## Univariate Gaussian data

Here, we develop a neural Bayes estimator for $\boldsymbol{\theta} \equiv (\mu, \sigma)'$ from data $Z_1, \dots, Z_m$ that are independent and identically distributed according to a $N(\mu, \sigma^2)$ distribution. We'll use the priors $\mu \sim N(0, 1)$ and $\sigma \sim U(0.1, 1)$, and we assume that the parameters are independent a priori.

Before proceeding, we load the required packages:
```
using NeuralEstimators
using Flux
using Distributions
import NeuralEstimators: simulate
```

First, we define a function to sample parameters from the prior. The sampled parameters are stored as $p \times K$ matrices, with $p$ the number of parameters in the model and $K$ the number of sampled parameter vectors.
```
function sample(K)
	μ = rand(Normal(0, 1), K)
	σ = rand(Uniform(0.1, 1), K)
	θ = hcat(μ, σ)'
	θ = Float32.(θ)
	return θ
end
```

Next, we implicitly define the statistical model with simulated data. The data are stored as a `Vector{A}`, where each element of the vector is associated with one parameter vector, and where `A` depends on the representation of the neural estimator. Since our data is replicated, we will use the Deep Sets framework and, since each replicate is univariate, we will use a dense neural network (DNN) for the inner network. Since the inner network is a DNN, `A` should be a sub-type of `AbstractArray`, with the independent replicates stored in the final dimension.
```
function simulate(parameters, m)
	Z = [θ[1] .+ θ[2] .* randn(Float32, 1, m) for θ ∈ eachcol(parameters)]
	return Z
end
```

We now design architectures for the inner and outer neural networks, $\boldsymbol{\psi}(\cdot)$ and $\boldsymbol{\phi}(\cdot)$ respectively, in the Deep Sets framework, and initialise the neural estimator as a [`PointEstimator`](@ref) object.

```
p = 2   # number of parameters in the statistical model
w = 32  # width of each layer

ψ = Chain(Dense(1, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, p))
architecture = DeepSet(ψ, ϕ)

θ̂ = PointEstimator(architecture)
```

Next, we train the neural estimator using [`train`](@ref), here using the default absolute-error loss. We'll train the estimator using 15 independent replicates per parameter configuration. Below, we pass our user-defined functions for sampling parameters and simulating data, but one may also pass parameter or data
instances, which will be held fixed during training; see [`train`](@ref).
```
m = 15
θ̂ = train(θ̂, sample, simulate, m = m, epochs = 30)
```

To test the accuracy of the resulting neural Bayes estimator, we use the function [`assess`](@ref), which can be used to assess the performance of the estimator (or multiple estimators) over a range of sample sizes. Note that, in this example, we trained the neural estimator using a single sample size, $m = 15$, and hence the estimator will not necessarily be optimal for other sample sizes; see [Variable sample sizes](@ref) for approaches that one could adopt if data sets with varying sample size are envisaged.
```
θ     = sample(1000)
Z     = [simulate(θ, m) for m ∈ (5, 10, 15, 20, 30)]
assessment = assess([θ̂], θ, Z)
```

The returned object is an object of type [`Assessment`](@ref), which contains the true parameters and their corresponding estimates, and the time taken to compute the estimates for each sample size and each estimator. The risk function may be computed using the function [`risk`](@ref):
```
risk(assessment)
```

It is often helpful to visualise the empirical sampling distribution of an estimator for a particular parameter configuration and a particular sample size. This can be done by providing [`assess`](@ref) with $J$ data sets simulated under a particular parameter configuration (below facilitated with the pre-defined method `simulate(parameters, m, J::Integer)`, which wraps the method of `simulate` that we defined earlier), and then plotting the estimates contained in the long-form `DataFrame` in the resulting [`Assessment`](@ref) object:
```
J = 100
θ = sample(1)
Z = [simulate(θ, m, J)]
assessment = assess([θ̂], θ, Z)  
```

Once the neural Bayes estimator has been assessed, it may then be applied to observed data, with parametric/non-parametric bootstrap-based uncertainty quantification facilitated by [`bootstrap`](@ref) and [`interval`](@ref). Below, we use simulated data as a substitute for observed data:
```
Z = simulate(θ, m)     # pretend that this is observed data
θ̂(Z)                   # point estimates from the observed data
θ̃ = bootstrap(θ̂, Z)    # non-parametric bootstrap estimates
interval(θ̃)  # confidence interval from the bootstrap estimates
```
