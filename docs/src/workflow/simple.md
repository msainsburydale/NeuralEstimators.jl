# Simple example

Perhaps the simplest estimation task involves inferring μ from N(μ, σ) data, where σ = 1 is known.

The first step is to define an object `ξ` that contains invariant model information; that is, information that does not change with time. In this example, we have two invariant objects: The prior distribution of the parameters, Ω, which we take to be a N(0, 0.5) distribution, and the standard deviation, σ.
```
ξ = (Ω = Normal(0, 0.5), σ = 1)
```

Next, we define a subtype of `ParameterConfigurations`; for this model, we need only store the, `θ`, which must be stored as a p × K matrix, where p is the dimension of θ (here, p = 1).
```
struct Parameters{T} <: ParameterConfigurations
	θ::AbstractMatrix{T, 2}
end
```

We then define a `Parameters` constructor, which returns `K` draws from `Ω`.
```
function Parameters(ξ, K::Integer)
	θ = rand(ξ.Ω, 1, K)
	Parameters(θ)
end
```

Next, we implicitly define the statistical model by overloading the `simulate` as follows.
```
import NeuralEstimators: simulate
function simulate(parameters::Parameters, ξ, m::Integer)
	n = 1
	θ = vec(parameters.θ)
	Z = [rand(Normal(μ, ξ.σ), n, 1, m) for μ ∈ θ]
end
```
There is some flexibility in the permitted type of the sample size, `m` (e.g., `Integer`, `IntegerRange`, etc.), but `simulate` must return an `AbstractVector` of (multi-dimensional) `AbstractArrays`, where each array is associated with one parameter vector (i.e., one column of `parameters.θ`). Note also that the size of each array must be amenable to `Flux` neural networks; for instance, above we return a 3-dimensional array, even though the second dimension is redundant.

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

Next, we train the neural estimator using `train`. The argument `m` specifies the sample size used during training, and its type should be consistent with the `simulate` method defined above.
```
θ̂ = train(θ̂, ξ, Ω, m = 10)
```

The estimator `θ̂` now approximates the Bayes estimator for θ. It's usually a good idea to assess the performance of the estimator before putting it into practice. Since the performance of `θ̂` for particular values of θ may be of particular interest, `estimate` takes an instance of `Parameters`.
```
parameters = Parameters(ξ, 500)      # test set with 500 parameters
m  = [1, 10, 30]                     # sample sizes we wish to test
df = estimate(θ̂, ξ, parameters, m = m)  
```
The true parameters, estimates, and timings from this test run are returned in a convenient `DataFrame`, ready for visualisation.
