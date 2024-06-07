
# Overview

To develop a neural estimator with `NeuralEstimators`,

- Sample parameters from the prior distribution. The parameters are stored as $p \times K$ matrices, with $p$ the number of parameters in the model and $K$ the number of parameter vectors in the given parameter set (i.e., training, validation, or test set).
- Simulate data from the assumed model over the parameter sets generated above. These data are stored as a `Vector{A}`, with each element of the vector associated with one parameter configuration, and where `A` depends on the multivariate structure of the data and the representation of the neural estimator (e.g., an `Array` for CNN-based estimators, a `GNNGraph` for GNN-based estimators, etc.).
- Initialise a neural network `θ̂`.  
- Train `θ̂` under the chosen loss function using [`train()`](@ref).
- Assess `θ̂` using [`assess()`](@ref), which utilises various simulation-based empirical methods for assessing the estimator with respect to its sampling distribution.


Once the estimator `θ̂` has passed our assessments and is therefore deemed to be well calibratred, it may be applied to observed data. See the [Examples](@ref) and, once familiar with the basic workflow, see [Advanced usage](@ref) for practical considerations on how to most effectively construct neural estimators.
