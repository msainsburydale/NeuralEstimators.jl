# Workflow overview

A general overview for developing a neural estimator with `NeuralEstimators.jl` is as follows.

- Create an object `ξ` that contains the information needed to sample the p-dimensional parameter vector θ.
- Define a type `Parameters <: ParameterConfigurations` used to store information needed for data simulation. `Parameters` must contain a field `θ`, which stores K parameter vectors as a p × K matrix.
- Define a `Parameters` constructor, `Parameters(ξ, K::Integer)`.  
- Implicitly define the statistical model by providing a method `simulate(parameters::Parameters, m)` which simulates `m` independent realisations from the statistical model.
- Initialise neural networks `ψ` and `ϕ`. These will typically be `Flux.jl` networks.
- Initialise a `DeepSet` object, `θ̂ = DeepSet(ψ, ϕ)`.
- Train `θ̂` using `train()` under an arbitrary loss function.
- Test `θ̂` using `estimate()`.
- Apply `θ̂` to real-world data.

For clarity, see a [Simple example](@ref) and a [More complicated example](@ref).
