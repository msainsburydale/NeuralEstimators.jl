# Examples



## Univariate Gaussian data

Here, we consider a very simple estimation task, namely, inferring ``\mu`` from ``N(\mu, \sigma)`` data, where ``\sigma`` is known. Specifically, we will develop a neural estimator for ``μ``, where

```math
\mu \sim N(0, 0.5), \quad \mathcal{Z} \equiv \{Z_1, \dots, Z_m\}, \; Z_i \sim N(μ, 1).
```


Before beginning, we load the required packages.
```
using NeuralEstimators
using Distributions
using Flux
```

Now we define the prior distribution, $\Omega(\cdot)$, and sample parameters from it to form sets of parameters used for training, validating, and testing the estimator. In `NeuralEstimators`, parameters are stored as $p \times K$ matrices, where $p$ is the number of parameters in the model and $K$ is the number of sampled parameter vectors.
```
Ω = Normal(0, 0.5)

p = 1
θ_train = rand(Ω, p, 10000)
θ_val   = rand(Ω, p, 2000)  
θ_test  = rand(Ω, p, 1000)  
```

Next, we implicitly define the statistical model via simulated data. In the following, we overload the function [`simulate`](@ref), but this is not necessary; one may simulate data however they see fit (e.g., using pre-existing functions, possibly from other programming languages).  Irrespective of its source, the data must be stored as a `Vector` of `Array`s, with each array associated with one parameter vector. The dimension of these array must also be amenable to `Flux` neural networks (e.g., here we simulate 3-dimensional arrays, despite the second dimension being redundant), and one typically stores the data using `Float32` precision for computational efficiency.
```
import NeuralEstimators: simulate

# m: number of independent replicates simulated for each parameter vector
function simulate(θ_set, m::Integer)
	Z = [rand(Normal(θ[1], 1), 1, 1, m) for θ ∈ eachcol(θ_set)]
	Z = broadcast.(Float32, Z)
	return Z
end

m = 15
Z_train = simulate(θ_train, m)
Z_val   = simulate(θ_val, m)
```


We then design neural network architectures for use in the Deep Set framework, and we initialise the neural estimator as a [`DeepSet`](@ref) object.
```
n = 1
w = 32
q = 16
ψ = Chain(Dense(n, w, relu), Dense(w, q, relu))
ϕ = Chain(Dense(q, w, relu), Dense(w, p), Flux.flatten)
θ̂ = DeepSet(ψ, ϕ)
```

Next, we train the neural estimator using [`train`](@ref), here using the default absolute-error loss function.
```
θ̂ = train(θ̂, θ_train, θ_val, Z_train, Z_val, epochs = 15)
```

The estimator `θ̂` now approximates the Bayes estimator under the prior distribution $\Omega(\cdot)$ and the absolute-error loss function and, hence, we refer to it as a *neural Bayes estimator*. To assess the performance of the estimator, one may use [`assess`](@ref). Below, we assess the performance over a range of sample sizes.
```
Z_test     = [simulate(θ_test, m) for m ∈ (5, 10, 15, 20, 30)]
assessment = assess([θ̂], θ_test, Z_test)
```
The returned object is of type [`Assessment`](@ref), and it contains the true parameters, estimates, and run times.  The true parameters and estimates may be merged into a convenient long-form `DataFrame` via [`merge`](@ref), and this greatly facilitates visualisation and diagnostic computation. Further, `NeuralEstimators` provides several plotting methods.
```
plotrisk(assessment)
```

Finally, it is often helpful to visualise the empirical joint distribution of an estimator for a particular parameter configuration and a particular sample size.
```            
J = 100
θ_scenario = rand(Ω, p, 1)
Z_scenario = [simulate(θ_scenario, m, J)]
assessment = assess([θ̂], θ_scenario, Z_scenario)  
```
The empirical joint distribution may then visualised as:
```
plotdistribution(assessment)
```

The estimator may then be applied to real data, with bootstrapping facilitated with...
