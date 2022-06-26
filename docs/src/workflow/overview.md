# Workflow overview

A general overview for developing a neural estimator with `NeuralEstimators.jl` is as follows.

- Create an object `ξ` containing invariant model information (e.g., the prior distribution for the p-dimensional parameter vector θ, spatial locations, etc.).
- Define a type `Parameters <: ParameterConfigurations`, containing a compulsory field `θ` storing K parameter vectors as a p × K matrix, and any other intermediate objects associated with the parameters (e.g., Cholesky factors) that are needed for data simulation.
- Define a `Parameters` constructor `Parameters(ξ, K::Integer)`, which draws `K` parameters from the prior.
- Implicitly define the statistical model by overloading the function `simulate`.
- Initialise neural networks `ψ` and `ϕ`, and a `DeepSet` object `θ̂ = DeepSet(ψ, ϕ)`.
- Train `θ̂` using `train` under an arbitrary loss function.
- Test `θ̂` using `estimate`.

For clarity, see a [Simple example](@ref) and a [More complicated example](@ref).
