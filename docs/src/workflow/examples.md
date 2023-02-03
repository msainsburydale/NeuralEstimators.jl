# Examples

## Univariate Gaussian data

Here, we consider a classical estimation task, namely, inferring $\mu$ and $\sigma$ from $N(\mu, \sigma^2)$ data. Specifically, we will develop a neural Bayes estimator for $\mathbf{\theta} \equiv (\mu, \sigma)'$ from independent and identically distributed data, $\mathbf{Z} \equiv (Z_1, \dots, Z_m)'$, where each $Z_i \sim N(\mu, \sigma)$.

Before proceeding, we load the required packages.
```
using NeuralEstimators
using Flux
using Distributions
```

Now, the first step of the workflow is to define the prior distribution for $\mathbf{\theta}$, which we denote by $\Omega(\cdot)$. We let $\mu \sim N(0, 1)$ and $\sigma \sim U(0.1, 1)$, and we assume that the parameters are independent a priori. We also sample parameters from $\Omega(\cdot)$ to form sets of parameters used for training, validating, and testing the estimator. It does not matter how $\Omega(\cdot)$ is stored or how the parameters are sampled; the only requirement is that the sampled parameters are stored as $p \times K$ matrices, with $p$ the number of parameters in the model and $K$ the number of sampled parameter vectors.
```
# Store the prior for each parameter as a named tuple
Ω = (μ = Normal(0, 1), σ = Uniform(0.1, 1))

function sample(Ω, K)
	μ = rand(Ω.μ, K)
	σ = rand(Ω.σ, K)
	θ = hcat(μ, σ)'
	return θ
end

θ_train = sample(Ω, 10000)
θ_val   = sample(Ω, 2000)
θ_test  = sample(Ω, 1000)
```

Next, we implicitly define the statistical model via simulated data. In the following, we overload the function [`simulate`](@ref), but this is not necessary; one may simulate data however they see fit (e.g., using pre-existing functions, possibly from other programming languages). The data should be stored as a `Vector{A}`, where each element of the vector is associated with one parameter configuration, and where `A` depends on the representation of the neural estimator. Here, since our neural estimator will be based on a dense neural network (DNN), we will store the data as an `Array`, with independent replicates stored in the final dimension.
```
import NeuralEstimators: simulate

# m: number of independent replicates simulated for each parameter vector.
function simulate(θ_set, m)
	Z = [rand(Normal(θ[1], θ[2]), 1, m) for θ ∈ eachcol(θ_set)]
	Z = broadcast.(Float32, Z)
	return Z
end

m = 15
Z_train = simulate(θ_train, m)
Z_val   = simulate(θ_val, m)
```

We now design architectures for the inner and outer neural networks, $\mathbf{\psi}(\cdot)$ and $\mathbf{\phi}(\cdot)$ respectively, in the Deep Set framework, and initialise the neural estimator as a [`DeepSet`](@ref) object. Since we have univariate data, it is natural to use a dense neural network.

```
n = 1    # size of each replicate (univariate data)
w = 32   # number of neurons in each layer
p = 2    # number of parameters in the statistical model

ψ = Chain(Dense(n, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, p), Flux.flatten)
θ̂ = DeepSet(ψ, ϕ)
```

Next, we train the neural estimator using [`train`](@ref), here using the default absolute-error loss function.
```
θ̂ = train(θ̂, θ_train, θ_val, Z_train, Z_val, epochs = 30)
```

To test the accuracy of the resulting neural Bayes estimator, we use the function [`assess`](@ref). This function can be used to assess the performance of the estimator (or multiple estimators) over a range of sample sizes. Note that, in this example, we trained the neural estimator using a single sample size $m$, and hence the estimator will not necessarily be optimal for all $m$; see [Variable sample sizes](@ref) for strategies towards developing neural estimators that are optimal for a range of $m$.
```
Z_test     = [simulate(θ_test, m) for m ∈ (5, 10, 15, 20, 30)]
assessment = assess([θ̂], θ_test, Z_test)
```
The returned object is an object of type [`Assessment`](@ref), which contains the true parameters and their corresponding estimates, and the time taken to compute the estimates for each sample size and each estimator.

The risk function may then be approximated using [`risk`](@ref), and plotted against the sample size with [`plotrisk`](@ref):
```
risk(assessment)
plotrisk(assessment)
```

In addition to assessing the estimator over the entire parameter space via the risk function, it is often helpful to visualise the empirical joint distribution of an estimator for a particular parameter configuration and a particular sample size. This can be done by providing $J$ data sets simulated under a single parameter configuration.
```            
J = 100
θ = sample(Ω, 1)
Z = [simulate(θ, m, J)]
assessment = assess([θ̂], θ, Z)  
```
The empirical joint distribution may then visualised with [`plotdistribution`](@ref):
```
plotdistribution(assessment)
```


Once the neural Bayes estimator has passed our assessment, it may then be applied to observed data, with bootstrap-based uncertainty quantification facilitated by [`bootstrap`](@ref):
```            
Z = simulate(θ, m) # pretend that this is observed data
bootstrap(θ̂, θ̂(Z), m)
```


## Time series (AR1)


## Irregular spatial data
