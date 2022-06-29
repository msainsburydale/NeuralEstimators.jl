# Simple example

Perhaps the simplest estimation task involves inferring μ from N(μ, σ) data, where σ is known, and this is the model that we consider. Specifically, we will develop a neural estimator for μ, where

```math
μ \sim N(0, 0.5), \quad \mathcal{Z} \equiv \{Z_1, \dots, Z_m\}, \; Z_i \sim N(μ, 1).
```

The first step is to define an object `ξ` that contains invariant model information. In this example, we have two invariant objects: The prior distribution of the parameters, Ω, and the standard deviation, σ.
```
using Distributions
ξ = (Ω = Normal(0, 0.5), σ = 1)
```

Next, we define a subtype of `ParameterConfigurations`, say, `Parameters` (the name is arbitrary); for the current model, `Parameters` need only stores the sampled parameters, which must be held in a field named `θ`:
```
using NeuralEstimators
struct Parameters <: ParameterConfigurations
	θ
end
```

We then define a `Parameters` constructor, returning `K` draws from `Ω`:
```
function Parameters(ξ, K::Integer)
	θ = rand(ξ.Ω, 1, K)
	Parameters(θ)
end
```

Next, we implicitly define the statistical model by overloading `simulate` as follows.
```
import NeuralEstimators: simulate
function simulate(parameters::Parameters, ξ, m::Integer)
	θ = vec(parameters.θ)
	Z = [rand(Normal(μ, ξ.σ), 1, 1, m) for μ ∈ θ]
end
```
There is some flexibility in the permitted type of the sample size `m` (e.g., `Integer`, `IntegerRange`, etc.), but `simulate` must return an `AbstractVector` of (multi-dimensional) `AbstractArrays`, where each array is associated with one parameter vector (i.e., one column of `parameters.θ`). Note also that the size of each array must be amenable to `Flux` neural networks; for instance, above we return a 3-dimensional array, even though the second dimension is redundant.

We then choose an architecture for modelling ψ(⋅) and ϕ(⋅) in the Deep Set framework,
and initialise the neural estimator as a `DeepSet` object.
```
p = 1
w = 32
q = 16
ψ = Chain(Dense(n, w, relu), Dense(w, q, relu))
ϕ = Chain(Dense(q, w, relu), Dense(w, p), flatten)
θ̂ = DeepSet(ψ, ϕ)
```

Next, we train the neural estimator using `train`. The argument `m` specifies the sample size used during training, and its type should be consistent with the `simulate` method defined above. There are two methods for `train`: Below, we provide the invariant model information `ξ` and the type `Parameters`, so that parameter configurations will be automatically and continuously sampled during training.
```
θ̂ = train(θ̂, ξ, Parameters, m = 10)
```

The estimator `θ̂` now approximates the Bayes estimator. It's usually a good idea to assess the performance of the estimator before putting it into practice. Since the performance of `θ̂` for particular values of θ may be of particular interest, `estimate` takes an instance of `Parameters`.
```
parameters = Parameters(ξ, 500)                   # test set with 500 parameters
m          = [1, 10, 30]                          # sample sizes we wish to test
estimates  = estimate(θ̂, ξ, parameters, m = m)  
```
The true parameters, estimates, and timings from this test run are returned in an `Estimates` object (each field is a `DataFrame` corresponding to the parameters, estimates, or timings). The true parameters and estimates may be merged into a convenient long-form `DataFrame`, and this greatly facilitates visualisation and diagnostic computation:
```
merged_df = merge(estimates)
```
