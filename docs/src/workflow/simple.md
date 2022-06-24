# Simple example

Perhaps the simplest estimation task involves inferring μ from N(μ, σ) data, where σ = 1 is known.

The first step is to define an object that can be used to sample parameters and pass on information to the data simulation function. Here, we define the prior distribution, Ω, of θ, which we take to be a mean-zero Normal distribution standard deviation 0.5. In this simple example, only Ω is needed, but we wrap it in a `Tuple` for consistency with the general workflow.
```
Ω = Normal(0, 0.5)  
ξ = (Ω = Ω)
```

Next, we define a sub-type of `ParameterConfigurations` with a field `θ`, which stores parameters as a p × K matrix, where p is the dimension of θ (here, p = 1).
```
struct Parameters{T} <: ParameterConfigurations
	θ::AbstractMatrix{T, 2}
end
```
Storing parameter information in a `struct` is useful for storing intermediates objects needed for data simulation, such as Cholesky factors, and for implementing variants of on-the-fly and just-in-time simulation.

We then define a `Parameters` constructor, which returns `K` draws from `Ω`.
```
function Parameters(ξ, K::Integer)
	θ = rand(ξ.Ω, 1, K)
	Parameters(θ)
end
```

Next, we implicitly define the statistical model by providing a method `simulate()`, which defines data simulation conditional on θ. The method must take two arguments; a `Parameters` object and `m`, the sample size. There is some flexibility in the permitted type of `m` (e.g., `Integer`, `IntegerRange`, etc.), but `simulate()` must return an `AbstractVector` of (multi-dimensional) `AbstractArrays`, where each array is associated with one parameter vector.
```
import NeuralEstimators: simulate
function simulate(params::Parameters, m::Integer)
	n = 1
	σ = 1
	θ = vec(params.θ)
	Z = [rand(Normal(μ, σ), n, 1, m) for μ ∈ θ]
end
```
Note that the size of each array must be amenable to `Flux` neural networks; for instance, above we return a 3-dimensional array, even though the second dimension is redundant.

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

Next, we train the neural estimator using `train()`. This optimisation is performed with respect to an arbitrary loss function (default absolute error loss). The argument `m` specifies the sample size used during training; the type of `m` should be consistent with the `simulate()` method defined above.
```
θ̂ = train(θ̂, Ω, m = 10)
```

The estimator `θ̂` now approximates the Bayes estimator for θ. It's usually a good idea to assess the performance of the estimator before putting it into practice. Since the performance of θ̂ for particular values of θ may be of particular interest, `estimate()` takes an instance of `Parameters`.
```
parameters = Parameters(Ω, 500)      # test set with 500 parameters
m  = [1, 10, 30]                     # sample sizes we wish to test
df = estimate(θ̂, parameters, m = m)  
```
The true parameters, estimates, and timings from this test run are returned in a convenient `DataFrame`, ready for visualisation.
