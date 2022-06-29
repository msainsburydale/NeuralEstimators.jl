# Workflow overview

To develop a neural estimator with `NeuralEstimators.jl`,

- Create an object `ξ` containing invariant model information, that is, model information that does not depend on the parameters and hence stays constant during training (e.g, the prior distribution of the parameters, spatial locations, distance matrices, etc.).
- Define a type `Parameters <: ParameterConfigurations` containing a compulsory field `θ` storing K parameter vectors as a p × K matrix, with p the dimension of θ, as well as any other intermediate objects associated with the parameters (e.g., Cholesky factors) that are needed for data simulation.
- Define a `Parameters` constructor `Parameters(ξ, K::Integer)`, which draws `K` parameters from the prior.
- Implicitly define the statistical model by overloading the function `simulate`.
- Initialise neural networks `ψ` and `ϕ`, and a `DeepSet` object `θ̂ = DeepSet(ψ, ϕ)`.
- Train `θ̂` using `train` under an arbitrary loss function.
- Test `θ̂` using `estimate`.

For clarity, see a [Simple example](@ref) and a [More complicated example](@ref). Once familiar with the basic workflow, see [Advanced usage](@ref) for some important practical considerations and how to construct neural estimators most effectively.
